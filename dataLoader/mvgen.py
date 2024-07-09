import numpy as np
import os
from glob import glob
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2, math
import random
from PIL import Image
import torch
import json

from scipy.spatial.transform import Rotation as R
from dataLoader.utils import intrinsic_to_fov, build_rays, build_rays_torch

import rembg


from third_party.image_generator.scripts.sampling.simple_video_sample import build_sv3d_model
from third_party.image_generator.scripts.sampling.simple_video_sample import sample as sv3d_pipe

image_extensions = ['*.jpg', '*.jpeg', '*.png']

class MVGenDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(MVGenDataset, self).__init__()

        self.img_size = np.array(cfg.img_size)
        self.generator_type = cfg.get("generator_type", "sv3d")
        
        self.model = self.init_model(self.generator_type, device='cuda')
        self.prompts = cfg.get("prompts", [])
        self.image_pathes = cfg.get("image_pathes", [])

        if len(self.image_pathes) and os.path.isdir(self.image_pathes):
            image_pathes = []
            for extension in image_extensions:
                search_pattern = os.path.join(self.image_pathes, extension)
                image_pathes.extend(glob(search_pattern))
            self.image_pathes = image_pathes

        self.bg_color = 1.0

    def init_model(self, generator_type, device):
        
        if generator_type == 'sv3d':
            sv3d_model = build_sv3d_model(
                                num_steps=30,
                                device=device,
                                )
            return sv3d_model
                
        elif generator_type == 'zero123plus-v1.1':
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
            zero123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
                local_files_only=False,
            )
            zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123.scheduler.config, timestep_spacing='trailing'
            )
            zero123.to(device)
            return zero123

        elif generator_type == 'zero123plus-v1.2':
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
            zero123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
                local_files_only=False,
            )
            zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123.scheduler.config, timestep_spacing='trailing'
            )
            zero123.to(device)
            return zero123
              
        else:
            raise NotImplementedError

    def gen(self, index):
                
        if self.generator_type == 'zero123plus-v1.1':
            assert len(self.image_pathes) > index, 'zero123plus-v1.1 model shoule be provided images.'
            
            image_path = self.image_pathes[index]
            return zero123plus_v11(
                    self.model,
                    image_path=image_path,
                    num_steps=30,
                    )
                
        elif self.generator_type == 'zero123plus-v1.2':
            assert len(self.image_pathes) > index, 'zero123plus-v1.2 model shoule be provided images.'
            
            image_path = self.image_pathes[index]
            return zero123plus_v12(
                    self.model,
                    image_path=image_path,
                    num_steps=30,
                    )
                
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        
        images, c2ws, fxfycxcy, name = self.gen(index)
        N_view = images.shape[0]
    
        fxfycxcy[...,[0,2]] = fxfycxcy[...,[0,2]] * self.img_size[0]
        fxfycxcy[...,[1,3]] = fxfycxcy[...,[1,3]] * self.img_size[1]
        ixts = torch.eye(3)[None].repeat(N_view,1,1).to(fxfycxcy)
        ixts[:,[0,1,0,1],[0,1,2,2]] = fxfycxcy

        fov_x, fov_y = intrinsic_to_fov(ixts[0].cpu().numpy(),w=self.img_size[0],h=self.img_size[1])
        ret = {'fovx':fov_x, 'fovy':fov_y}
        H, W = self.img_size

        # scale to our bbox 
        c2ws[...,:3,3] /= 1.7
        w2cs = torch.inverse(c2ws)

        # align cameras using first view
        # no inver operation 
        dist = torch.norm(c2ws[0,:3,3])
        ref_c2w = torch.eye(4).reshape(1,4,4).to(c2ws)
        ref_w2c = torch.eye(4).reshape(1,4,4).to(c2ws)
        ref_c2w[...,2,3], ref_w2c[...,2,3] = -dist, dist
        transform_mats = ref_c2w @ w2cs[:1]
        w2cs = w2cs.clone() @ c2ws[:1] @ ref_w2c
        c2ws = transform_mats @ c2ws.clone()

        bg_color = np.ones(3).astype(np.float32)

        ret.update({'tar_c2w': c2ws,
                    'tar_w2c': w2cs,
                    'tar_ixt': ixts,
                    'tar_rgb': images,
                    'bg_color': bg_color,
                    'transform_mats': transform_mats
                    })

        near_far = torch.tensor([dist-1.0, dist+1.0]).to(c2ws)
        ret.update({'near_far':near_far})
        ret.update({'meta': {'scene':name,f'tar_h': int(H), f'tar_w': int(W)}})


        rays = build_rays_torch(c2ws, ixts.clone(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays_torch(c2ws, ixts.clone(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
            
        return ret

    def __len__(self):
        return max(len(self.prompts),len(self.image_pathes))

def pad_image_to_square(image_path):
    # Open the original image
    original_image = Image.open(image_path)
    
    # Get dimensions
    width, height = original_image.size
    
    # Determine new size for a square image
    new_size = max(width, height)
    
    # Create a new image with the required size and the color of the top-left corner pixel as the background
    new_image = Image.new('RGB', (new_size, new_size), original_image.getpixel((0, 0)))
    
    # Compute the position to paste the original image on the new canvas
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    
    # Paste the original image onto the new canvas
    new_image.paste(original_image, paste_position)
    
    # Return the padded image
    return new_image

def zero123plus_v11(
          zero123_model,
          image_path,
          input_res=(512,512),
          num_steps=30,
          device='cuda:0'
          ):
    cond = pad_image_to_square(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    mv_images = []
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        image_fg = cv2.resize(image_fg, input_res)
        mv_images.append(image_fg) 

    # normalize
    images = np.stack(mv_images, axis=0)
    # images = (images - 0.5)*2
    images = torch.tensor(images).float().to(device)
    # 1, V, C, H, W
    # images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[30, 225+30], [30, 225+150], [30, 225+270], [-20, 225+330]], fov=50)
    fxfycxcy = (fxfycxcy.unsqueeze(0)).repeat(c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    
    return images, c2ws, fxfycxcy, name

def zero123plus_v12(
          zero123_model,
          image_path,
          input_res=(512,512),
          num_steps=30,
          device='cuda:0'
          ):
    cond = pad_image_to_square(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    mv_images = [] 
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        # image_fg = pad_image_to_fit_fov((image_fg*255).astype(np.uint8), 50, 30)
        image_fg = cv2.resize(image_fg, input_res)
        # image_fg = image_fg / 255
        mv_images.append(image_fg)

    # normalize
    images = np.stack(mv_images, axis=0)
    images = torch.tensor(images).float().to(device)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225+30], [20, 225+150], [20, 225+270], [-10, 225+330]], fov=30)
    fxfycxcy = (fxfycxcy.unsqueeze(0)).repeat(c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    # sample = prepare_output(images, c2ws, fxfycxcy, name, img_size=input_res)

    torch.cuda.empty_cache()
    return images, c2ws, fxfycxcy, name

def sv3d(
          sv3d_model,
          image_path,
          input_res=(512,512),
          num_steps=30,
          device='cuda:0'
          ):

    video = sv3d_pipe(model=sv3d_model,
                input_path=image_path,
                version='sv3d_p',
                elevations_deg=20.0,
                azimuths_deg=[0,10,30,50,90,110,130,150,180,200,220,240,270,280,290,300,310,320,330,340,350],
                output_folder=f'outputs/sv3d')
    torch.cuda.empty_cache()

    mv_images = video[[0, 4, 8, 12]]
    
    mv_images = [ cv2.resize(image, input_res) for image in mv_images]

    # normalize
    images = np.stack(mv_images, axis=0)
    images = torch.tensor(images).float().to(device)
    # 1, V, C, H, W
    # images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225], [20, 225+90], [20, 225+180], [20, 225+270]], fov=33.8)
    fxfycxcy = (fxfycxcy.unsqueeze(0)).repeat(c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    
    torch.cuda.empty_cache()
    return images, c2ws, fxfycxcy, name

    
def generate_input_camera(r, poses, device='cuda:0', fov=50):
    def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    poses = np.deg2rad(poses)
    poses = torch.tensor(poses).float()
    pitch = poses[:, 0]
    yaw = poses[:, 1]

    z = r*torch.sin(pitch)
    x = r*torch.cos(pitch)*torch.cos(yaw)
    y = r*torch.cos(pitch)*torch.sin(yaw)
    cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                                        device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                        dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                        dim=-1))
    rotate = torch.stack(
                    (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix

    fx = 0.5/np.tan(np.deg2rad(fov/2))
    fxfycxcy = torch.tensor([fx, fx, 0.5, 0.5], dtype=rotate.dtype, device=device)

    return cam2world, fxfycxcy

def pad_image_to_fit_fov(image, new_fov, old_fov):
    img = Image.fromarray(image)

    scale_factor = math.tan(np.deg2rad(new_fov/2)) / math.tan(np.deg2rad(old_fov/2))

    # Calculate the new size
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))

    # Calculate padding
    pad_width = (new_size[0]-img.size[0]) // 2
    pad_height = (new_size[1] - img.size[1]) // 2

    # Create padding
    padding = (pad_width, pad_height, pad_width+img.size[0], pad_height+img.size[1])

    # Pad the image
    img_padded = Image.new(img.mode, (new_size[0], new_size[1]), color='white')
    img_padded.paste(img, padding)
    img_padded = np.array(img_padded)
    return img_padded

