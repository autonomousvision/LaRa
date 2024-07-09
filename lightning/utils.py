import torch, os, json, math
import numpy as np
from torch.optim.lr_scheduler import LRScheduler

def getProjectionMatrix(znear, zfar, fovX, fovY):

    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, device):
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

        self.world_view_transform = w2c.transpose(0, 1).to(device)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(device)
        )
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).float()
        self.camera_center = -c2w[:3, 3].to(device)


def rotation_matrix_to_quaternion(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return torch.stack([qw, qx, qy, qz], dim=1)

def rotate_quaternions(q, R):
    # Convert quaternions to rotation matrices
    q = torch.cat([q[:, :1], -q[:, 1:]], dim=1)
    q = torch.cat([q[:, :3], q[:, 3:] * -1], dim=1)
    rotated_R = torch.matmul(torch.matmul(q, R), q.inverse())
    
    # Convert the rotated rotation matrices back to quaternions
    return rotation_matrix_to_quaternion(rotated_R)

# this function is borrowed from OpenLRM
class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, initial_lr: float = 1e-10, last_iter: int = -1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_iter)

    def get_lr(self):

        if self._step_count <= self.warmup_iters:
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self._step_count / self.warmup_iters
                for base_lr in self.base_lrs]
        else:
            cos_iter = self._step_count - self.warmup_iters
            cos_max_iter = self.max_iters - self.warmup_iters
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [base_lr * (1 + math.cos(cos_theta)) / 2 for base_lr in self.base_lrs]
            return cos_lr