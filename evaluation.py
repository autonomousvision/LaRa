import os

n_thread = 1
os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
os.environ["OMP_NUM_THREADS"] = f"4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 


from omegaconf import OmegaConf
import os, torch, math, imageio, cv2
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys,json
from lightning.system import system
from torch.utils.data import DataLoader
import pytorch_lightning as L
from dataLoader import dataset_dict

from pytorch_msssim import ssim
from tools.gen_video_path import uni_video_path,uni_mesh_path

import lpips
import torch.nn.functional as F
from tools.depth import acc_threshold,abs_error

@torch.no_grad()
def main(cfg):

    torch.set_float32_matmul_precision('medium')

    # data loader
    dataset = dataset_dict[cfg.infer.dataset.dataset_name]
    loader = DataLoader(dataset(cfg.infer.dataset), 
                              batch_size=cfg.infer.dataset.batch_size,
                              num_workers=cfg.infer.dataset.num_workers, 
                              shuffle=False,
                              pin_memory=False)
    loader_iter = iter(loader)

    device = 'cuda'
    my_system = system.load_from_checkpoint(cfg.infer.ckpt_path, cfg=cfg, map_location=device)

    # metrics
    lpips_vgg_fun = lpips.LPIPS(net='vgg').to(device)
    lpips_alex_fun = lpips.LPIPS(net='alex').to(device)

    names, depth_accs = [], []
    psnrs,ssims, lpips_vggs, lpips_alexs = [],[],[],[]
    os.makedirs(cfg.infer.save_folder, exist_ok=True)
    for i in tqdm(range(len(loader))):#len(loader)
        sample = next(loader_iter)
        sample = {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in sample.items()} 

        my_system.net.eval()
        
        return_buffer = cfg.infer.video_frames > 0 or cfg.infer.save_mesh
        output = my_system.net(sample, with_fine=True, return_buffer=return_buffer)
        
        name = sample['meta']['scene'][0].split('.')[0]
        images = output['image_fine'][0]
        img_gt = sample['tar_rgb'][0].permute(1,0,2,3).reshape(images.shape)
        alpha = output['acc_map'][0][...,None]
        normal_white = ((output['rend_normal_fine'][0]*alpha+1-alpha) + 1)/2

        n_view = cfg.n_views

        
        if i<100:
            cv2.imwrite(os.path.join(cfg.infer.save_folder, name + '.jpg'), torch.cat((img_gt,images,normal_white),dim=0).detach().cpu().numpy()[...,::-1]*255)
        
        if cfg.infer.eval_novel_view_only:
            width = sample['meta']['tar_w']
            images = images.permute(2,0,1)[None][...,width*n_view:]
            img_gt = img_gt.permute(2,0,1)[None][...,width*n_view:]
        else:
            images = images.permute(2,0,1)[None]
            img_gt = img_gt.permute(2,0,1)[None]
        
        if images.shape[-1] > 0:
            color_loss_all = (images-img_gt)**2
            psnr = -10. * torch.log(color_loss_all.mean()) / torch.log(torch.tensor([10.]).to(device))
        
            ssim_val = ssim(images, img_gt, data_range=1.0, size_average=False)
            
            lpips_vgg = lpips_vgg_fun(img_gt*2-1,images*2-1)
            lpips_alex = lpips_alex_fun(img_gt*2-1,images*2-1)
            
            psnrs.append(psnr.item())
            ssims.append(ssim_val.item())
            lpips_vggs.append(lpips_vgg.item())
            lpips_alexs.append(lpips_alex.item())
       
        if len(cfg.infer.eval_depth):
            B,N,H,W = sample['tar_msk'].shape
            mask = sample['tar_msk'].permute(0,2,1,3).reshape(B,H,N*W)
            mask = mask.cpu().detach().bool().numpy()
            depth_gt = sample['tar_dep'].permute(0,2,1,3).reshape(B,H,N*W)
            depth_gt = depth_gt.cpu().detach().numpy()
            
            depth_pred = output['depth_fine'].cpu().squeeze(-1).detach().numpy()

            depth_acc = []
            errors = abs_error(depth_pred, depth_gt, mask).mean().item()
            depth_acc.append(errors)
            for threshold in cfg.infer.eval_depth:
                depth_acc.append(acc_threshold(depth_pred, depth_gt, mask, threshold=threshold).mean())
            depth_accs.append(depth_acc)
            
        names.append(name)


        fov = [sample['fovx'],sample['fovy']]

        if cfg.infer.video_frames > 0:

            cams =  uni_video_path(cfg.infer.video_frames, cfg.infer.dataset, sample, fov=fov)
            
            gs_params = output['render_pkg'][1] # fine ouputs
            _centers, _shs, _opacity, _scaling, _rotation, mask = gs_params
        
            imgs,normal_blks,normal_whites = [],[],[]
            for cam in cams:
                cam.to_device(device)
                rays = cam.get_rays().to(device)
                output_img = my_system.net.gs_render.render_img(cam, rays, _centers, _shs, _opacity[mask], _scaling[mask], _rotation[mask], device)
                img, normal = output_img['image'], output_img['rend_normal']
                img = np.round(img.cpu().detach().numpy()*255).astype('uint8')
                

                alpha = output_img['acc_map'][...,None]
                normal_white = np.round((((normal*alpha+1-alpha) + 1)/2).cpu().detach().numpy()*255).astype('uint8')
        
                imgs.append(img)
                normal_whites.append(normal_white)

            imageio.mimwrite(f'{cfg.infer.save_folder}/{name}.mp4', imgs, fps=30, quality=10)
            imageio.mimwrite(f'{cfg.infer.save_folder}/{name}_nrm.mp4', normal_whites, fps=30, quality=10)

        
        if cfg.infer.save_mesh:
            from tools.meshExtractor import MeshExtractor
            aabb = cfg.infer.aabb
            gs_params = output['render_pkg'][1] # fine ouputs
            meshExtractor = MeshExtractor(gs_params, my_system.net.gs_render, aabb=aabb)
            meshExtractor.extract(f'{cfg.infer.save_folder}/{name}.obj', cfg.infer.dataset, sample=sample,fov=fov)
            
            if cfg.infer.mesh_video_frames > 0:
                from tools.meshRender import render_mesh
                cams =  uni_video_path(cfg.infer.video_frames, cfg.infer.dataset, sample=sample, fov=fov)
                mesh_imgs = render_mesh(cams, f'{cfg.infer.save_folder}/{name}.obj')[...,:3]
                imageio.mimwrite(f'{cfg.infer.save_folder}/{name}_mesh.mp4', mesh_imgs, fps=30, quality=10)
                
        del sample
    
    if len(cfg.infer.eval_depth):
        mean_depth_acc = np.mean( np.stack(depth_accs),axis=0).tolist()
    else:
        mean_depth_acc = 0.0

    if len(psnrs) and cfg.infer.metric_path is not None:
        print(f'evaluation score, psnr: {np.mean(psnrs)} ssim: {np.mean(ssims)}, lpips_vgg:{np.mean(lpips_vggs)}, lpips_alex: {np.mean(lpips_alexs)}, depth_acc:{mean_depth_acc}')
        
        scores = {'name':names, 'psnr':psnrs, 'ssim':ssims, \
                'lpips_vgg':lpips_vggs,'lpips_alex':lpips_alexs, \
                'depth_acc': depth_accs}
        scores.update({'psnr_mean':np.mean(psnrs), 'ssim_mean':np.mean(ssims), 
                    'lpips_vgg_mean':np.mean(lpips_vggs),'lpips_alex_mean':np.mean(lpips_alexs),
                    'depth_acc': mean_depth_acc})
        
        os.makedirs(os.path.dirname(cfg.infer.metric_path), exist_ok=True)
        with open(cfg.infer.metric_path, 'w') as f:
            json.dump(scores, f, indent=4)
        
if __name__ == '__main__':
    
    base_conf = OmegaConf.load('configs/base.yaml')
    path_config = sys.argv[1]
    cli_conf = OmegaConf.from_cli()
    second_conf = OmegaConf.load(path_config)
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)

    main(cfg)