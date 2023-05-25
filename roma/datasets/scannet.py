import os
import random
from PIL import Image
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset)

import torchvision.transforms.functional as tvf
import kornia.augmentation as K
import os.path as osp
import matplotlib.pyplot as plt
import roma
from roma.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
from roma.utils.transforms import GeometricSequential
from tqdm import tqdm

class ScanNetScene:
    def __init__(self, data_root, scene_info, ht = 384, wt = 512, min_overlap=0., shake_t = 0, rot_prob=0.,use_horizontal_flip_aug = False,
) -> None:
        self.scene_root = osp.join(data_root,"scans","scans_train")
        self.data_names = scene_info['name']
        self.overlaps = scene_info['score']
        # Only sample 10s
        valid = (self.data_names[:,-2:] % 10).sum(axis=-1) == 0
        self.overlaps = self.overlaps[valid]
        self.data_names = self.data_names[valid]
        if len(self.data_names) > 10000:
            pairinds = np.random.choice(np.arange(0,len(self.data_names)),10000,replace=False)
            self.data_names = self.data_names[pairinds]
            self.overlaps = self.overlaps[pairinds]
        self.im_transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=True)
        self.depth_transform_ops = get_depth_tuple_transform_ops(resize=(ht, wt), normalize=False)
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.H_generator = GeometricSequential(K.RandomAffine(degrees=90, p=rot_prob))
        self.use_horizontal_flip_aug = use_horizontal_flip_aug

    def load_im(self, im_B, crop=None):
        im = Image.open(im_B)
        return im
    
    def load_depth(self, depth_ref, crop=None):
        depth = cv2.imread(str(depth_ref), cv2.IMREAD_UNCHANGED)
        depth = depth / 1000
        depth = torch.from_numpy(depth).float()  # (h, w)
        return depth

    def __len__(self):
        return len(self.data_names)
    
    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht /  hi
        sK = torch.tensor([[sx, 0, 0],
                        [0, sy, 0],
                        [0, 0, 1]])
        return sK@K

    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
    def read_scannet_pose(self,path):
        """ Read ScanNet's Camera2World pose and transform it to World2Camera.
        
        Returns:
            pose_w2c (np.ndarray): (4, 4)
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        world2cam = np.linalg.inv(cam2world)
        return world2cam


    def read_scannet_intrinsic(self,path):
        """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
        """
        intrinsic = np.loadtxt(path, delimiter=' ')
        return torch.tensor(intrinsic[:-1, :-1], dtype = torch.float)

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        data_name = self.data_names[pair_idx]
        scene_name, scene_sub_name, stem_name_1, stem_name_2 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        
        # read the intrinsic of depthmap
        K1 = K2 =  self.read_scannet_intrinsic(osp.join(self.scene_root,
                       scene_name,
                       'intrinsic', 'intrinsic_color.txt'))#the depth K is not the same, but doesnt really matter
        # read and compute relative poses
        T1 =  self.read_scannet_pose(osp.join(self.scene_root,
                       scene_name,
                       'pose', f'{stem_name_1}.txt'))
        T2 =  self.read_scannet_pose(osp.join(self.scene_root,
                       scene_name,
                       'pose', f'{stem_name_2}.txt'))
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4]  # (4, 4)

        # Load positive pair data
        im_A_ref = os.path.join(self.scene_root, scene_name, 'color', f'{stem_name_1}.jpg')
        im_B_ref = os.path.join(self.scene_root, scene_name, 'color', f'{stem_name_2}.jpg')
        depth_A_ref = os.path.join(self.scene_root, scene_name, 'depth', f'{stem_name_1}.png')
        depth_B_ref = os.path.join(self.scene_root, scene_name, 'depth', f'{stem_name_2}.png')

        im_A = self.load_im(im_A_ref)
        im_B = self.load_im(im_B_ref)
        depth_A = self.load_depth(depth_A_ref)
        depth_B = self.load_depth(depth_B_ref)

        # Recompute camera intrinsic matrix due to the resize
        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)
        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops((depth_A[None,None], depth_B[None,None]))
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)

        data_dict = {'im_A': im_A,
                    'im_B': im_B,
                    'im_A_depth': depth_A[0,0],
                    'im_B_depth': depth_B[0,0],
                    'K1': K1,
                    'K2': K2,
                    'T_1to2':T_1to2,
                    }
        return data_dict


class ScanNetBuilder:
    def __init__(self, data_root = 'data/scannet') -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root,'scannet_indices')
        self.all_scenes = os.listdir(self.scene_info_root)
        
    def build_scenes(self, split = 'train', min_overlap=0., **kwargs):
        # Note: split doesn't matter here as we always use same scannet_train scenes
        scene_names = self.all_scenes
        scenes = []
        for scene_name in tqdm(scene_names, disable = roma.RANK > 0):
            scene_info = np.load(os.path.join(self.scene_info_root,scene_name), allow_pickle=True)
            scenes.append(ScanNetScene(self.data_root, scene_info, min_overlap=min_overlap, **kwargs))
        return scenes
    
    def weight_scenes(self, concat_dataset, alpha=.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n)/n**alpha for n in ns])
        return ws
