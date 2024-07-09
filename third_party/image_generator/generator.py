
import os
import random
import rembg
import sys
from PIL import Image
import numpy as np
import torch
import imageio
import math
import cv2

from dataLoader.utils import intrinsic_to_fov, build_rays_torch

from dataLoader.image_generator.scripts.sampling.simple_video_sample import build_sv3d_model
from dataLoader.image_generator.scripts.sampling.simple_video_sample import sample as sv3d_pipe

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

def prepare_output(images, c2ws, fxfycxcy, name, img_size=(512,512)):
    
    N_view = images.shape[1]
    images = images.permute(0,1,3,4,2) # [B,N,H,W,C]
    
    fxfycxcy[...,[0,2]] = fxfycxcy[...,[0,2]] * img_size[0]
    fxfycxcy[...,[1,3]] = fxfycxcy[...,[1,3]] * img_size[1]
    ixts = torch.eye(3)[None].repeat(*fxfycxcy.shape[:2],1,1).to(fxfycxcy)
    ixts[:,:,[0,1,0,1],[0,1,2,2]] = fxfycxcy
    
    fov_x, fov_y = intrinsic_to_fov(ixts[0,0].cpu().numpy(),w=img_size[0],h=img_size[1])
    ret = {'fovx':fov_x, 'fovy':fov_y, }
    H, W = img_size
    
    w2cs = torch.inverse(c2ws)

    # align cameras using first view
    # no inver operation 
    dist = torch.norm(c2ws[0,0,:3,3])
    ref_c2w = torch.eye(4).reshape(1,1,4,4).to(c2ws)
    ref_w2c = torch.eye(4).reshape(1,1,4,4).to(c2ws)
    ref_c2w[...,2,3], ref_w2c[...,2,3] = -dist, dist
    transform_mats = ref_c2w @ w2cs[:1,:1]
    w2cs = w2cs.clone() @ c2ws[:1,:1] @ ref_w2c
    c2ws = transform_mats @ c2ws.clone()
    
    ret.update({'tar_c2w': c2ws,
                'tar_w2c': w2cs,
                'tar_ixt': ixts,
                'tar_rgb': images.permute(0,2,1,3,4).reshape(1,H,N_view*W,3),
                'transform_mats': transform_mats
                })
    near_far = torch.tensor([dist-1.0, dist+1.0]).to(c2ws).unsqueeze()
    ret.update({'near_far':near_far})
    ret.update({'meta': {'scene':name,f'tar_h': int(H), f'tar_w': int(W)}})

    rays = build_rays_torch(c2ws[0], ixts[0].clone(), H, W, 1.0).unsqueeze(0)
    ret.update({f'tar_rays': rays})
    rays_down = build_rays_torch(c2ws[0], ixts[0].clone(), H, W, 1.0/16).unsqueeze(0)
    ret.update({f'tar_rays_down': rays_down})
        
    return ret
    
    
def zero123plus_v11(
          zero123_model,
          grm_model_cfg,
          image_path,
          num_steps=30,
          device='cuda:0'
          ):
    cond = Image.open(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = []
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        image_fg = cv2.resize(image_fg, (input_size, input_size))
        mv_images.append(image_fg) 

    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[30, 225+30], [30, 225+150], [30, 225+270], [-20, 225+330]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]

def zero123plus_v12(
          zero123_model,
          grm_model_cfg,
          image_path,
          num_steps=30,
          device='cuda:0'
          ):
    cond = Image.open(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = [] 
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        image_fg = pad_image_to_fit_fov((image_fg*255).astype(np.uint8), 50, 30)
        image_fg = cv2.resize(image_fg, (input_size, input_size))
        image_fg = image_fg / 255
        mv_images.append(image_fg)

    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225+30], [20, 225+150], [20, 225+270], [-10, 225+330]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]


def sv3d(
          sv3d_model,
          grm_model_cfg,
          image_path,
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

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = video[[0, 4, 8, 12]]
    
    mv_images = [ cv2.resize(pad_image_to_fit_fov(image, 50, 33.8), (input_size, input_size)) for image in mv_images]


    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images/255 - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225], [20, 225+90], [20, 225+180], [20, 225+270]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]

def generate(prompts=None,
             image_pathes=None,
             generator_type: str='zero123plus-v1.1',
             device='cuda'):
    
    if generator_type == 'sv3d':
        assert image_pathes is not None, 'sv3d model shoule be provided images.'
        sv3d_model = build_sv3d_model(
                             num_steps=30,
                             device=device,
                             )
        
        for image_path in image_pathes:
            sv3d(
                  sv3d_model=sv3d_model,
                  image_path=image_path,
                  num_steps=30,
                  )
            
    elif generator_type == 'zero123plus-v1.1':
        assert image_pathes is not None, 'zero123plus-v1.1 model shoule be provided images.'
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        zero123 = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
            zero123.scheduler.config, timestep_spacing='trailing'
        )
        zero123.to(device)
        
        for image_path in image_pathes:
            zero123plus_v11(
                zero123,
                image_path=image_path,
                num_steps=30,
                )
            
    elif generator_type == 'zero123plus-v1.2':
        assert image_pathes is not None, 'zero123plus-v1.2 model shoule be provided images.'
        zero123 = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
            zero123.scheduler.config, timestep_spacing='trailing'
        )
        zero123.to(device)
        
        for image_path in image_pathes:
            zero123plus_v12(
                zero123,
                image_path=image_path,
                num_steps=30,
                )
            
    else:
        raise NotImplementedError
    
    return samples
            

            
