import torch
import torch.utils.data as data
from dataset.utils import *
import random
import transforms3d
import copy
import math
import glob
import open3d
import os

def augment_cloud(input, gt, input_rand=None, pc_augm_scale=1.2, pc_augm_rot=True, 
                    pc_rot_scale=90, pc_augm_mirror_prob=0.5, 
                    translation_magnitude=0.1, pc_augm_jitter=False):
    """" Augmentation on XYZ and jittering of everything """
    # Ps is a list of point clouds

    M = transforms3d.zooms.zfdir2mat(1) # M is 3*3 identity matrix
    # scale
    if pc_augm_scale > 1:
        s = random.uniform(1/pc_augm_scale, pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)

    # rotation
    if pc_augm_rot:
        scale = pc_rot_scale # we assume the scale is given in degrees
        # should range from 0 to 180
        if scale > 0:
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) 
    
    # mirror
    if pc_augm_mirror_prob > 0: # mirroring x&z, not y
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)

    # translation
    translation_sigma = translation_magnitude
    translation_sigma = max(pc_augm_scale, 1) * translation_sigma
    if translation_sigma > 0:
        noise = np.random.normal(scale=translation_sigma, size=(1, 3))
        # noise = noise.astype(Ps[0].dtype)
        
    input[:,:3] = np.dot(input[:,:3], M.T)
    gt[:,:3] = np.dot(gt[:,:3], M.T)
    if input_rand is not None:
        input_rand[:,:3] = np.dot(input_rand[:,:3], M.T)

    if translation_sigma > 0:
        input[:,:3] = input[:,:3] + noise
        gt[:,:3] = gt[:,:3] + noise
        if input_rand is not None:
            input_rand[:,:3] = input_rand[:,:3] + noise

    if pc_augm_jitter:
        sigma = 0.02
        input = input + sigma * np.random.randn(*input.shape).astype(np.float32)
        # gt = gt + np.clip(sigma * np.random.randn(*gt.shape), -1*clip, clip).astype(np.float32)
        if input_rand is not None:
            input_rand = input_rand + sigma * np.random.randn(*input.shape).astype(np.float32)

    return input, gt, input_rand



class PUDataset(data.Dataset):
    def __init__(self, args):
        super(PUDataset, self).__init__()

        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.input_data, self.gt_data, self.radius_data = load_h5_data(args)

    
    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        # (n, 3)
        radius = self.radius_data[index]
        input = copy.deepcopy(self.input_data[index])
        gt = copy.deepcopy(self.gt_data[index])
        # radius = radius * scale
        # augmentation
        # sample_lst = np.random.choice(input.shape[0], input.shape[0], replace=False)
        # input = input[sample_lst, :]

        permutation = np.arange(gt.shape[0])
        np.random.shuffle(permutation)
        input_random = gt[permutation[:input.shape[0]]]
        # permutation = np.arange(gt.shape[0])
        # np.random.shuffle(permutation)
        # gt = gt[permutation]

        input, gt, input_random = augment_cloud(input, gt, input_random, pc_augm_jitter=False)
        # to tensor
        input = torch.from_numpy(input)
        input_random = torch.from_numpy(input_random)
        gt = torch.from_numpy(gt)
        radius = torch.from_numpy(radius)


        return input, gt, radius, input_random
    

class PUDataset_test(data.Dataset):
    def __init__(self, args):
        super(PUDataset_test, self).__init__()

        self.args = args
        self.input_path = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/input_2048"
        self.gt_path = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/gt_8192"

        # ---- input ----
        plys = glob.glob(os.path.join(self.input_path, "*.xyz"))
        input_data = []
        for ply in plys:
            pc = open3d.io.read_point_cloud(ply)
            points = np.asarray(pc.points, dtype=np.float32)
            input_data.append(points)
        self.input_data = np.stack(input_data, axis=0)
        # ---- input ----

        # ---- gt ----
        plys = glob.glob(os.path.join(self.gt_path, "*.xyz"))
        gt_data = []
        for ply in plys:
            pc = open3d.io.read_point_cloud(ply)
            points = np.asarray(pc.points, dtype=np.float32)
            gt_data.append(points)
        self.gt_data = np.stack(gt_data, axis=0)
        # ---- gt ----

        # ---- name ----
        self.plys = [ply.split("/")[-1][:-4] for ply in plys]
        # ---- name ----

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        # (n, 3)
        input = copy.deepcopy(self.input_data[index])
        gt = copy.deepcopy(self.gt_data[index])
        # radius = radius * scale
        # to tensor
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)

        return input, gt