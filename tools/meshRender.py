import os,sys
import mitsuba as mi
from tqdm import tqdm
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import numpy as np

def render_mesh(cams, mesh_path, spp = 512, white_bg=True):
    
    image_width = cams[0].image_width
    image_height = cams[0].image_height

    mesh_type = os.path.splitext(mesh_path)[1][1:]
    sdf_scene = mi.load_file("configs/render/scene.xml", resx=image_width, resy=image_height, mesh_path=mesh_path, mesh_type=mesh_type,
                            integrator_file="configs/render/integrator_path.xml", update_scene=False, spp=spp, max_depth=8)

    imgs = []
    pbar = tqdm(total=len(cams), desc='Files', position=0)
    b2c = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    for cam in cams:
        c2w, fov = cam.view_world_transform.numpy(), cam.FoVx
        fov = np.degrees(fov)
        image_width = cam.image_width
        image_height = cam.image_height
        
        to_world = c2w @ b2c
        to_world_transform = mi.ScalarTransform4f(to_world.tolist())
        
        sensor = mi.load_dict({
            'type': 'perspective',
                    'fov': fov, 'sampler': {'type': 'independent'},
                    'film': {'type': 'hdrfilm', 'width': image_width, 'height': image_height,
                            'pixel_filter': {'type': 'gaussian'}, 'pixel_format': 'rgba'},
                    'to_world': to_world_transform
                    })

        img = mi.render(sdf_scene, sensor=sensor, spp=spp)
        img = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)
        # img.write(f'123.png')
        img = np.array(img, copy=False)
        if white_bg:
            img = img.astype(np.float32)/255
            img = img[...,:3]*img[...,3:] + (1.0-img[...,3:])*np.array([0.722,0.376,0.161])
            img = np.round(img*255).astype('uint8')
            
        imgs.append(img)
        pbar.update(1)
        pbar.set_description("Mesh extraction Done. Rendering *_mesh.mp4: ")
        
    return np.stack(imgs)