import torch
import numpy as np
from glob import glob
import os
import open3d
from models.utils import *
from models.diffusion import *
from einops import rearrange
from time import time
from args.pufm_args import parse_pc_args
from args.utils import str2bool
from tqdm import tqdm
import argparse
import plyfile
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
cd_module = chamfer_3DDist()


def numpy_to_pc(points):
    pc = open3d.geometry.PointCloud()
    points = open3d.utility.Vector3dVector(points)
    pc.points = points
    return pc

def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def pcd_update_patch(args, model, interpolated_pcd):
    # interpolated_pcd: (b, 3, n)
    pcd_pts_num = interpolated_pcd.shape[-1]
    # 1024
    patch_pts_num = args.num_points * 4
    # extract patch
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    # FPS: (b, 3, fps_pts_num), ensure seeds have a good coverage
    seed = FPS(interpolated_pcd, sample_num)
    # (b*fps_pts_num, 3, patch_pts_num)
    patches = extract_knn_patch(patch_pts_num, interpolated_pcd, seed)
    # normalize each patch
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    # fix the parameters of model while updating the patches
    for param in model.parameters():
        param.requires_grad = False

    updated_patch = patches.clone() 
    bs = patches.shape[0]
    steps = 5
    with torch.no_grad():
        for t in tqdm(range(steps), "sampling loop"):
            # 1st order sampling
            alpha = t / steps * torch.ones(bs, device="cuda")
            pred = model(updated_patch, patches, alpha)
            updated_patch = updated_patch + (1 / steps) * pred

    # transform to original scale and merge patches
    updated_patch = updated_patch.clamp(-1, 1)
    updated_patch = centroid + updated_patch * furthest_distance
    # (3, m)
    updated_pcd = rearrange(updated_patch, 'b c n -> c (b n)').contiguous()
    # post process: (1, 3, n)
    output_pts_num = interpolated_pcd.shape[-1]
    updated_pcd = FPS(updated_pcd.unsqueeze(0), output_pts_num)

    return updated_pcd


def pcd_update_whole(args, model, interpolated_pcd):
    # fix the parameters of model while updating the patches
    for param in model.parameters():
        param.requires_grad = False

    updated_pcd = interpolated_pcd.clone() 
    bs = updated_pcd.shape[0]
    steps = 5
    with torch.no_grad():
        for t in tqdm(range(steps), "sampling loop"):
            # 1st order sampling
            alpha = t / steps * torch.ones(bs, device="cuda")
            pred = model(updated_pcd, interpolated_pcd, alpha)
            updated_pcd = updated_pcd + (1 / steps) * pred

    # transform to original scale and merge patches
    updated_pcd = updated_pcd.clamp(-1, 1)
    # (3, m)
    updated_pcd = rearrange(updated_pcd, 'b c n -> c (b n)').contiguous()

    return updated_pcd

def pcd_upsample(args, model, input_pcd):
    # interpolate: (b, 3, m)
    interpolated_pcd = midpoint_interpolate(args, input_pcd)
    # update: (b, 3, m)
    if args.sr_method == 'patch':
        updated_pcd = pcd_update_patch(args, model, interpolated_pcd)
    elif args.sr_method == 'whole':
        updated_pcd = pcd_update_whole(args, model, interpolated_pcd)

    return updated_pcd


def test(args):
    # load model
    if args.model == 'pufm':
        model = PUFM(args).cuda()
    elif args.model == 'pufm_w_attn':
        model = PUFM_w_attn(args).cuda()

    model_path = os.path.join(args.ckpt_folder, args.model+'.pth')    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # log
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save upsampled point cloud
    pcd_dir = os.path.join(save_dir, 'pcd')
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)

    # test
    pcd = open3d.io.read_point_cloud(args.test_input_path)
    pcd_name = args.test_input_path.split('/')[-1]
    input_pcd = np.array(pcd.points)
    input_pcd = torch.from_numpy(input_pcd).float().cuda()
    # (n, 3) -> (3, n)
    input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
    # (3, n) -> (1, 3, n)
    input_pcd = input_pcd.unsqueeze(0)
    output_pts_num = input_pcd.shape[2] * args.up_rate
    # normalize input
    # input_pcd = input_pcd + 0.01*torch.randn_like(input_pcd)
    input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
    # upsample
    upsampled_pcd = pcd_upsample(args, model, input_pcd)
    # double upsampling
    upsampled_pcd = pcd_upsample(args, model, upsampled_pcd.unsqueeze(0))
    # FPS sampling
    upsampled_pcd = FPS(upsampled_pcd.unsqueeze(0), output_pts_num)
    upsampled_pcd = centroid + upsampled_pcd * furthest_distance
        
    # (b, 3, n) -> (n, 3)
    upsampled_pcd = rearrange(upsampled_pcd.squeeze(0), 'c n -> n c').contiguous()
    upsampled_pcd = upsampled_pcd.detach().cpu().numpy()
    # save path
    save_path = os.path.join(pcd_dir, pcd_name)
    upsampled_pcd = numpy_to_pc(upsampled_pcd)
    open3d.io.write_point_cloud(filename=save_path, pointcloud=upsampled_pcd)


def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--test_input_path', default='/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/input_2048/camel.xyz', type=str, help='the test input pc path')
    parser.add_argument('--model', default='pufm_w_attn', type=str, help='the pretrained model [pufm, pufm_w_attn]')
    parser.add_argument('--sr_method', default='whole', type=str, help='the upsampling approach, either patch based upsampling for larger PC, or take the whole PC [patch, whole]')
    parser.add_argument('--ckpt_folder', default='pretrained_model', type=str, help='the pretrained model folder')
    parser.add_argument('--save_dir', default='output', type=str, help='save upsampled point cloud')
    parser.add_argument('--truncate_distance', default=True, type=str2bool, help='whether truncate distance')
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = parse_test_args()

    test(test_args)
