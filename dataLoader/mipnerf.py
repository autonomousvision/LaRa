import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from dataLoader.utils import intrinsic_to_fov
from dataLoader.utils import build_rays

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class MipNeRF360Dataset(Dataset):
    def __init__(self, cfg , split='train', hold_every=8):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = cfg.data_root
        self.split = split
        self.hold_every = hold_every
        self.is_stack = False if 'train' == split else True
        self.downsample = cfg.get(f'downsample_{self.split}', 4.0)
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()


    def read_meta(self):

        print(self.root_dir)
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_4/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        scene_info = {'ixts': [], 'c2ws': [], 'w2cs':[], 'imgs': [], 'fovx': [], 'fovy':[]}
        for i in img_list:
            image_path = self.image_paths[i]
            
            c2w = torch.eye(4)
            c2w[:3] = torch.FloatTensor(self.poses[i]).float()
            c2w[:3,3] /= 2.0

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.permute(1, 2, 0)  # (h, w, 3) RGB

            H, W, focal = hwf[i]
            focal = [focal * self.img_wh[0] / W, focal * self.img_wh[1] / H]
            ixt = torch.tensor([[focal[0],0,self.img_wh[0]/2],[0,focal[1],self.img_wh[1]/2],[0,0,1.0]])
            fovx, fovy = intrinsic_to_fov(ixt, self.img_wh[0], self.img_wh[1])

            scene_info['ixts'].append(ixt.float())
            scene_info['c2ws'].append(c2w.float())
            scene_info['w2cs'].append(torch.inverse(c2w).float())

            scene_info['imgs'].append(img.float())
            scene_info['fovx'].append(fovx.float())
            scene_info['fovy'].append(fovy.float())

        for item in scene_info:
            scene_info[item] = torch.stack(scene_info[item]) 
        self.scene_info = scene_info

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return 1000#len(scene_info[''])

    def __getitem__(self, index):

        # # align cameras using first view
        # # no inver operation 
        # r = np.linalg.norm(tar_c2ws[0,:3,3])
        # ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        # ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        # ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        # transform_mats = ref_c2w @ tar_w2cs[:1]
        # tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        # tar_c2ws = transform_mats @ tar_c2ws.copy()
        
        view_id = torch.randperm(len(self.scene_info[f'c2ws'])).tolist()[:4]
        
        ret = {'fovx': self.scene_info[f'fovx'][view_id], 
               'fovy': self.scene_info[f'fovy'][view_id], 
               }
        W, H = self.img_wh

        ret.update({'tar_c2w': self.scene_info[f'c2ws'][view_id],
                    'tar_w2c': self.scene_info[f'w2cs'][view_id],
                    'tar_ixt': self.scene_info[f'ixts'][view_id],
                    'tar_rgb': self.scene_info[f'imgs'][view_id].permute(1,0,2,3).reshape(H,len(view_id)*W,3),
                    'tar_msk': torch.ones(H,len(view_id)*W)
                    })
        
        near_far = np.array([np.min(self.near_fars), np.max(self.near_fars)]).astype(np.float32)
        ret.update({'near_far': near_far})
        ret['meta'] = {f'tar_h': int(H), f'tar_w': int(W)}

        rays = build_rays(ret['tar_c2w'].numpy(), ret['tar_ixt'].numpy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(ret['tar_c2w'].numpy(), ret['tar_ixt'].numpy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret