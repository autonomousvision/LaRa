import numpy as np
import os
from glob import glob
import imageio
from tqdm import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from PIL import Image
import torch
import json
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R
from dataLoader.utils import intrinsic_to_fov, KMean, read_pfm

class GoogleObjsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(GoogleObjsDataset, self).__init__()
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        self.img_downscale = self.img_size/512

        scenes_name = np.array([f for f in sorted(os.listdir(self.data_root)) if os.path.isdir(f'{self.data_root}/{f}')])
        i_test = np.arange(len(scenes_name))[::10][:cfg.n_scenes]
        i_train = np.array([i for i in np.arange(len(scenes_name)) if
                        (i not in i_test)])[:cfg.n_scenes]
        self.scenes_name =  scenes_name#[i_train] if self.split=='train' else scenes_name[i_test]
        
        self.n_group = cfg.n_group
        self.build_metas()

    def build_metas(self):

        self.scene_infos = {}

        for scene in tqdm(self.scenes_name):
                
            json_info = json.load(open(os.path.join(self.data_root, scene,f'transforms.json')))
            b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_info = {'ixts': [], 'c2ws': [], 'w2cs':[], 'img_paths': [], 'depth_paths': [], 'fovx': [], 'fovy':[]}
            positions = []
            for idx, frame in enumerate(json_info['frames']):
                c2w = np.array(frame['transform_matrix'])
                c2w = c2w @ b2c
                # ext = np.linalg.inv(c2w)
                ixt = np.array(frame['intrinsic_matrix'])
                fov_x, fov_y = intrinsic_to_fov(ixt)
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['c2ws'].append(c2w.astype(np.float32))
                scene_info['w2cs'].append(np.linalg.inv(c2w.astype(np.float32)))
                img_name = os.path.basename(frame['file_path'])
                img_path = os.path.join(self.data_root, scene, f'r_{idx:03d}.png')
                depth_path = os.path.join(self.data_root, scene, f'depth/r_{idx:03d}.pfm')
                scene_info['img_paths'].append(img_path)
                scene_info['fovx'].append(fov_x)
                scene_info['fovy'].append(fov_y)
                scene_info['depth_paths'].append(depth_path)
                positions.append(c2w[:3,3])

            groups = KMean(np.stack(positions), n_clusters=self.n_group)
            scene_info['groups'] = groups
            
            groups_4 = KMean(np.stack(positions), n_clusters=4)
            scene_info['groups_4'] = groups_4
            
            self.scene_infos[scene] = scene_info
        

    def __getitem__(self, index):

        # index = 5
        # index = np.random.randint(0, len(self.scenes_name))
            
        scene_name = self.scenes_name[index]
        scene_info = self.scene_infos[scene_name]
        
        if self.split=='train':
            src_view_id = [random.choices(scene_info['groups'][i])[0] for i in torch.randperm(self.n_group).tolist()]
            tar_views = src_view_id + [random.choices(scene_info['groups'][i])[0] for i in torch.randperm(self.n_group).tolist()]
        else:
            src_view_id = [scene_info['groups'][i][0] for i in range(self.n_group)]
            tar_views = src_view_id + [scene_info['groups_4'][i][-1] for i in range(4)]
        
        bg_color = np.ones(3).astype(np.float32)

        tar_img, tar_dep, tar_msks, tar_c2ws, tar_w2cs, tar_ixts = self.read_views(scene_info, tar_views, bg_color)
        
        # align cameras using first view
        # no inver operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        tar_c2ws = transform_mats @ tar_c2ws.copy()

        ret = {'fovx':scene_info['fovx'][tar_views[0]], 
               'fovy':scene_info['fovy'][tar_views[0]],
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'tar_dep': tar_dep,
                    'tar_msk': tar_msks,
                    'bg_color': bg_color[None].repeat(len(tar_views),0),
                    'transform_mats': transform_mats
                    })
        near_far = np.array([0.5, 2.5]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene_name, 'tar_view': tar_views, 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret

    def read_view(self, scene, idx):
        img, mask = self.read_image(scene, idx)
        ixt, ext = self.read_cam(scene, idx)
        return img, mask, ext, ixt
    
    def read_views(self, scene, src_views, bg_color):
        src_ids = src_views
        ixts, exts, w2cs, imgs, msks, deps = [], [], [], [], [], []
        for idx in src_ids:
            img, mask, depth = self.read_image(scene, idx, bg_color)
            imgs.append(img)
            ixt, ext, w2c = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
            deps.append(depth)
        return np.stack(imgs), np.stack(deps), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)

    def read_cam(self, scene, view_idx):
        ext = scene['c2ws'][view_idx]
        w2c = scene['w2cs'][view_idx]
        ixt = scene['ixts'][view_idx].copy()
        ixt[:2] =  ixt[:2] * self.img_downscale.reshape(2,1)
        return ixt, ext, w2c

    def read_image(self, scene, view_idx, bg_color):
        img_path = scene['img_paths'][view_idx]
        img = imageio.imread(img_path)
        if self.img_downscale[0]!=1 or self.img_downscale[1]!=1:
            img = cv2.resize(img, self.img_size)
        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)
        depth = self.read_depth(scene['depth_paths'][view_idx])
        return img, mask, depth

    def read_depth(self, depth_path):
        depth,_ = read_pfm(depth_path)
        return depth

    def __len__(self):
        return len(self.scene_infos)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

