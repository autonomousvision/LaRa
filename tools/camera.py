import math, torch
from dataLoader.utils import build_rays, fov_to_ixt

def getProjectionMatrix(znear, zfar, fovX, fovY):

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar


        w2c = torch.inverse(c2w)

        # rectify...
        # w2c[1:3, :3] *= -1
        # w2c[:3, 3] *= -1

        self.view_world_transform = c2w
        self.world_view_transform = w2c.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0, 1)

        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).to(torch.float32)
        self.camera_center = -c2w[:3, 3]

    def to_device(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.camera_center = self.camera_center.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        
    def get_rays(self):
        ixt = fov_to_ixt(torch.tensor((self.FoVx,self.FoVy)), torch.tensor((self.image_width,self.image_height)))
        rays = build_rays(self.view_world_transform.cpu().numpy()[None], ixt[None], self.image_height, self.image_width)
        return torch.from_numpy(rays)