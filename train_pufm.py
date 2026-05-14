import os
import torch
import sys
sys.path.append(os.getcwd())
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset, PUDataset_test
from models.diffusion import *
# from models.diffusion_v2 import *
from args.pufm_args import parse_pc_args
from args.utils import str2bool
from models.utils import *
from torch.cuda.amp import autocast, GradScaler
import copy
import argparse
from einops import rearrange, reduce
from tqdm import tqdm
import torch.distributions as dist
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
cd_module = chamfer_3DDist()
from emd_assignment import emd_module


def get_alignment_clean(aligner):
    @torch.no_grad()
    def align(noisy, clean):
        noisy = noisy.clone().transpose(1, 2).contiguous()
        clean = clean.clone().transpose(1, 2).contiguous()
        dis, alignment = aligner(noisy, clean, 0.01, 100)
        return alignment.detach()

    return align


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load training data
    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
    # load testing data
    test_dataset = PUDataset_test(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   shuffle=False,
                                                   batch_size=1, # args.batch_size
                                                   num_workers=args.num_workers)

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # create model 
    logger.info('========== Build Model ==========')
    model = PUFM_w_attn(args)
    model = model.cuda()
    if args.pretrained_path is not None:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
        print('================ successfully load pretrained model=====================')
    # get the parameter size
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
    # log
    logger.info(args)
    logger.info(repr(model))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # alignment
    aligner = emd_module.emdModule()
    emd_align = get_alignment_clean(aligner)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    best_valid_loss = np.Inf
    # train
    logger.info('========== Begin Training ==========')
    for epoch in range(args.epochs):
        logger.info('********* Epoch %d *********' % (epoch + 1))
        # epoch loss
        epoch_p2p_loss = 0.0
        epoch_loss = 0.0
        model.train()

        for i, (input_pts, gt_pts, radius, input_random) in enumerate(train_loader):
            # (b, n, 3) -> (b, 3, n)
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            input_random = rearrange(input_random, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            # query points
            mid_pts = midpoint_interpolate(args, input_random)
            # random sampling
            bs = gt_pts.shape[0]
            # cosine sampling
            t = torch.rand(size=(bs, )).cuda() 
            t = 1 - torch.cos(t * math.pi / 2) 
            alpha = t[:, None, None]
            #================ generate query points =================
            noise_pts = mid_pts + 0.01 * torch.randn_like(mid_pts)
            # EMD align
            align_idxs = emd_align(noise_pts, gt_pts)
            align_idxs = align_idxs.detach().long()
            align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
            gt_pts = torch.gather(gt_pts, -1, align_idxs)
            # linear interpolation
            query_pts = alpha * gt_pts + (1 - alpha) * noise_pts
            #============= model output =================
            pred_pts = model(query_pts, mid_pts, t)
            #============= MSE loss ================
            p2p_loss = torch.sum((pred_pts - (gt_pts - noise_pts))**2) # best one
            loss = p2p_loss

            epoch_loss += loss.item()
            epoch_p2p_loss += p2p_loss.item()

            # update parameters
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.step_ema(ema_model, model)

            # log
            writer.add_scalar('train/loss', loss, i)
            writer.add_scalar('train/p2p_loss', p2p_loss, i)
            writer.flush()
            if (i+1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, loss: %f, p2p_loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), 
                       epoch_loss / (i+1), epoch_p2p_loss / (i+1)))

        # log
        interval = time.time() - start
        logger.info("epoch: %d/%d, avg loss: %f, time: %d mins %.1f secs" %
          (epoch + 1, args.epochs, 
           epoch_loss / len(train_loader),
           interval / 60, interval % 60))

        # testing
        if (epoch + 1) % args.save_rate == 0:
            count = 0
            test_loss = 0
            model.eval()
            for i, (input_pts, gt_pts) in enumerate(test_loader):
                input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
                gt_pts = gt_pts.float().cuda()
                mid_pts = midpoint_interpolate(args, input_pts)
                # #=============================== process as whole ================================
                patches, centroid, furthest_distance = normalize_point_cloud(mid_pts)
                updated_patch = patches.clone()
                bs = patches.shape[0]
                n = gt_pts.shape[1]
                steps = 5
                with torch.no_grad():
                    for t in tqdm(range(steps), "sampling loop"):
                        alpha = t / steps * torch.ones(bs, device="cuda")
                        pred = model(updated_patch, patches, alpha)
                        updated_patch = updated_patch + (1 / steps) * pred
                updated_patch = updated_patch.clamp(-1, 1)
                updated_patch = centroid + updated_patch * furthest_distance
                # CD evaluation
                cd_p, dist, _,_ = cd_module(updated_patch.permute(0, 2, 1), gt_pts)
                dist = (cd_p + dist) / 2.0
                cd = dist.mean().detach().cpu().item()
                test_loss += cd
                count += 1
            test_loss = test_loss / count
            writer.add_scalar('test/loss', test_loss, epoch)
            logger.info("TEST epoch: %d/%d, avg loss: %.5E" %
                        (epoch + 1, args.epochs, 
                         test_loss))
        # save checkpoint
        if (epoch + 1) % args.save_rate == 0:
            if test_loss < best_valid_loss:
                model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
                ema_model_name = 'ema-ckpt-epoch-%d.pth' % (epoch+1)
                model_path = os.path.join(ckpt_dir, model_name)
                ema_model_path = os.path.join(ckpt_dir, ema_model_name)
                torch.save(model.state_dict(), model_path)
                # torch.save(ema_model.state_dict(), ema_model_path)
                best_valid_loss = test_loss


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--global_sigma', default=1.5, type=float, help='global sampling rate')
    parser.add_argument('--grad_lambda', default=0.1, type=float, help='gradient weights')
    parser.add_argument('--epochs', default=500, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--print_rate', default=50, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=1, type=int, help='model save frequency')
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
    parser.add_argument('--pretrained_path', default=None, type=str, help='the pretrained path')

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['pu1k', 'pugan']

    model_args = parse_pc_args()

    reset_model_args(train_args, model_args)
    if not hasattr(model_args, "up_rate"): model_args.up_rate = 4
    if not hasattr(model_args, "num_points"): model_args.num_points = 256

    train(model_args)
