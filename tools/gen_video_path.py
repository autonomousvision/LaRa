import torch, math
from scipy.spatial.transform import Rotation as R
from tools.camera import MiniCam
from tools.camera_utils import get_interpolated_poses_many


def generate_gobjverse_frames(N, config, sample=None, elevation=0, fov=None):
    frames = []

    width, height = config.img_size
    znear, zfar = 0.5, 2.5
    if fov is None:
        fovx, fovy = 0.75,0.75
    else:
        fovx, fovy = fov[0].item(), fov[1].item()
    fovx, fovy = 0.75,0.75
    
    elevation_rot = torch.eye(4)
    elevation_rot[:3,:3] = torch.tensor(R.from_euler('y', elevation/180.0*math.pi).as_matrix())
    
    transform_mats = torch.eye(4) if sample is None else sample['transform_mats'][0].cpu().squeeze(0)
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor([[0,1.0,0.0],[0.4515947,0.0,-0.8922232],[-0.8922232,0,-0.4515947]]).t()
    c2w[:3,3] = torch.tensor([1.70006549,0.0,0.8604804])

    c2w = elevation_rot @ c2w
    cam = MiniCam(transform_mats @ c2w, width, height, fovy, fovx, znear, zfar)
    frames.append(cam)
    
    rot_step = torch.eye(4)
    rot_step[:3,:3] = torch.tensor(R.from_euler('z', math.pi*2/N).as_matrix())
    
    for i in range(N-1):
        c2w = rot_step @ c2w
        cam = MiniCam(transform_mats @ c2w, width, height, fovy, fovx, znear, zfar)
        frames.append(cam)
    
    return frames


def generate_instant3d_frames(N, config, sample=None, elevation=0, fov=None):
    frames = []
    
    width, height = config.img_size
    znear, zfar = 1.0, 3.0
    if fov is None:
        fovx, fovy = 0.7,0.7
    else:
        fovx, fovy = fov[0].item(), fov[1].item()
    
    elevation_rot = torch.eye(4)
    elevation_rot[:3,:3] = torch.tensor(R.from_euler('x', elevation/180.0*math.pi).as_matrix())
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor([[-7.0710677e-01,  2.4184476e-01, -6.6446304e-01],
                                [7.0710677e-01,  2.4184476e-01, -6.6446304e-01],
                                [-5.2163419e-17, -9.3969262e-01, -3.4202015e-01]])
    c2w[:3,3] = torch.tensor([1.328926,1.328926,6.8404031e-01])
    c2w = elevation_rot @ c2w
    
    if sample is None:
        transform_mats = torch.tensor([[-7.0710677e-01,  7.0710677e-01,  7.8504622e-17,  0.0000000e+00],
            [ 2.4184476e-01,  2.4184476e-01, -9.3969262e-01,  0.0000000e+00],
            [-6.6446304e-01, -6.6446304e-01, -3.4202015e-01,  0.0000000e+00],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
    else:
        transform_mats = sample['transform_mats'][0].cpu().squeeze(0)
    cam = MiniCam(transform_mats @ c2w, width, height, fovy, fovx, znear, zfar)
    frames.append(cam)
    
    rot_step = torch.eye(4)
    rot_step[:3,:3] = torch.tensor(R.from_euler('z', math.pi*2/N).as_matrix())
    
    for i in range(N-1):
        c2w =  rot_step @ c2w
        cam = MiniCam(transform_mats @ c2w, width, height, fovy, fovx, znear, zfar)
        frames.append(cam)
    
    return frames

def generate_unposed_frames(N, config, sample=None, elevation=0, fov=None):
    frames = []
    
    width, height = config.img_size
    znear, zfar = 1.0, 3.0
    if fov is None:
        fovx, fovy = 0.7,0.7
    else:
        fovx, fovy = fov[0].item(), fov[1].item()
    
    c2ws, ixt = sample['tar_c2w'][0,:,:3].clone().detach(), sample['tar_ixt'][0].clone().detach()
    traj, k_interp = get_interpolated_poses_many(c2ws, ixt, steps_per_transition=N//len(c2ws), order_poses=True)
    
    elevation_rot = torch.eye(4)
    elevation_rot[:3,:3] = torch.tensor(R.from_euler('x', elevation/180.0*math.pi).as_matrix())
    
    _c2w = torch.eye(4)
    for c2w in traj:
        _c2w[:3] = elevation_rot @ c2w
        cam = MiniCam(_c2w, width, height, fovy, fovx, znear, zfar)
        frames.append(cam)
    
    return frames


def uni_video_path(N, data, sample=None, fov=None):
    
    if data.dataset_name in ['gobjeverse','GSO']:
        pathes =  generate_gobjverse_frames(N, data, sample, fov=fov)
    elif data.dataset_name in ['instant3d','mvgen']:
        pathes =  generate_instant3d_frames(N, data, sample, fov=fov)
    elif data.dataset_name in ['unposed']:
        pathes =  generate_unposed_frames(N, data, sample, fov=fov)
    return pathes
    
def uni_mesh_path(N, data, sample, fov=None):
    
    # transform_mats = torch.eye(4) if transform_mats is None else transform_mats
    
    pathes = []
    for elevation in [0,-30,30]:
        if data.dataset_name in ['gobjeverse','GSO']:
            pathes.extend(generate_gobjverse_frames(N, data, sample, elevation,fov=fov))
        elif data.dataset_name in ['instant3d','co3d','mvgen']:
            pathes.extend(generate_instant3d_frames(N, data, sample, elevation,fov=fov))
        elif data.dataset_name in ['unposed']:
            pathes.extend(generate_unposed_frames(N, data, sample, elevation,fov=fov))
    return pathes