import numpy as np
import os
from glob import glob
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from PIL import Image
import torch
import json
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R
from dataLoader.utils import intrinsic_to_fov, KMean

class Instant3DObjsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(Instant3DObjsDataset, self).__init__()
        self.data_root = cfg.data_root

        self.img_size = np.array(cfg.img_size)

        scenes_name = np.array([f for f in sorted(os.listdir(self.data_root)) if f.endswith('png')])
        self.scenes_name =  scenes_name
        print(len(self.scenes_name))

        self.build_camera()
        self.bg_color = 1.0

    def build_camera(self):
        scene_info = {'c2ws':[],'w2cs':[],'ixts':[]}
        json_info = json.load(open(os.path.join(self.data_root, f'opencv_cameras.json')))

        for i in range(4):
            frame = json_info['frames'][i]
            w2c = np.array(frame['w2c'])
            c2w = np.linalg.inv(w2c)
            c2w[:3,3] /= 1.7
            w2c = np.linalg.inv(c2w)
            scene_info['c2ws'].append(c2w)
            scene_info['w2cs'].append(w2c)
            
            ixt = np.eye(3)
            ixt[[0,1],[0,1]] = np.array([frame['fx'],frame['fy']])
            ixt[[0,1],[2,2]] = np.array([frame['cx'],frame['cy']])
            scene_info['ixts'].append(ixt)
        
        scene_info['c2ws'] = np.stack(scene_info['c2ws']).astype(np.float32)
        scene_info['w2cs'] = np.stack(scene_info['w2cs']).astype(np.float32)
        scene_info['ixts'] = np.stack(scene_info['ixts']).astype(np.float32)
        
        self.scene_info = scene_info

    def __getitem__(self, index):


        scenes_name = self.scenes_name[index]
        # src_view_id = list(range(4))
        # tar_views = src_view_id + list(range(4))
        
        #np.random.rand(3)
        tar_img = self.read_image(scenes_name)
        tar_c2ws = self.scene_info['c2ws']
        tar_w2cs = self.scene_info['w2cs']
        tar_ixts = self.scene_info['ixts']
        
        # align cameras using first view
        # no inver operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        tar_c2ws = transform_mats @ tar_c2ws.copy()

        fov_x, fov_y = intrinsic_to_fov(tar_ixts[0],w=512,h=512)
        
        ret = {'fovx':fov_x, 
               'fovy':fov_y,
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img.transpose(1,0,2,3).reshape(H,4*W,3),
                    'transform_mats': transform_mats
                    })
        near_far = np.array([r-1.0, r+1.0]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene':scenes_name,f'tar_h': int(H), f'tar_w': int(W)}})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret


    def read_image(self, scenes_name):

        img = imageio.imread(f'{self.data_root}/{scenes_name}')
        img = img.astype(np.float32) / 255.
        if img.shape[-1] == 4:
            img = (img[..., :3] * img[..., -1:] + self.bg_color*(1 - img[..., -1:])).astype(np.float32)

        # split images
        row_chunks = np.array_split(img, 2)
        imgs = np.stack([np.array_split(chunk, 2, axis=1) for chunk in row_chunks]).reshape(4,512,512,-1)
        return imgs


    def __len__(self):
        return len(self.scenes_name)