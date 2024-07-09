import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
from tools.gen_video_path import uni_mesh_path

def storePly(path, xyz, normal):
    from plyfile import PlyData, PlyElement
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    rgb = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normal, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

from functools import partial
class MeshExtractor(object):
    def __init__(self, gs_params, render, aabb, bg_color=(1.0,1.0,1.0)):

        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if aabb is not None:
            self.aabb = np.array(aabb).reshape(2,3)*1.1
        else:
            self.aabb = None

        self.gs_params = gs_params
        self.render = render 
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.points = []
    
    @torch.no_grad()
    def extract(self, save_mesh_path, dataset_name, voxel_size=2/256, sdf_trunc=0.08, alpha_thres=0.08, depth_trunc=10, sample=None, fov=None, device='cuda'):
        import open3d as o3d
        import copy
        if self.aabb is not None:
            center = self.aabb.mean(0)
            radius = np.linalg.norm(self.aabb[1] - self.aabb[0]) * 0.5
            voxel_size = radius / 256
            sdf_trunc = voxel_size * 2
            print("using aabb")

        print("running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')
        print(f'alpha_thres: {alpha_thres}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        cams = uni_mesh_path(16, dataset_name, sample, fov)
        _centers, _shs, _opacity, _scaling, _rotation, mask = self.gs_params
        for cam in tqdm(cams):
            cam.to_device(device)
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width=cam.image_width, 
                    height=cam.image_height, 
                    cx = cam.image_width/2,
                    cy = cam.image_height/2,
                    fx = cam.image_width / (2 * math.tan(cam.FoVx / 2.)),
                    fy = cam.image_height / (2 * math.tan(cam.FoVy / 2.)))
     
            rays = cam.get_rays().squeeze(0).to(device)
            render_pkg = self.render.render_img(cam, rays, _centers, _shs, _opacity[mask], _scaling[mask], _rotation[mask], device)
            
            depth = render_pkg['depth']
            alpha = render_pkg['acc_map']
            rgb = render_pkg['image']
            
            # if viewpoint_cam.gt_alpha_mask is not None:
            #     depth[(viewpoint_cam.gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < alpha_thres)] = 0
            if self.aabb is not None:
                campos = cam.camera_center.cpu().numpy()
                depth_trunc = np.linalg.norm(campos - center, axis=-1) + radius
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, 
                        intrinsic=intrinsic, 
                        extrinsic=np.asarray((cam.world_view_transform.T).cpu().numpy()))

        mesh = volume.extract_triangle_mesh()
        # write mesh
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # o3d.io.write_triangle_mesh(path, mesh)
        mesh_0 = copy.deepcopy(mesh)

        if self.aabb is not None:
            vert_mask = ~((np.asarray(mesh_0.vertices) >= self.aabb[0]).all(-1) & (np.asarray(mesh_0.vertices) <= self.aabb[1]).all(-1))
            triangles_to_remove = vert_mask[np.array(mesh_0.triangles)].any(axis=-1)
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        # postprocessing
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        largest_cluster_idx = cluster_n_triangles.argmax()
        cluster_to_keep = min(len(cluster_n_triangles),10)
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(save_mesh_path, mesh_0)
        # print("num vertices raw {}".format(len(mesh.vertices)))
        # print("num vertices post {}".format(len(mesh_0.vertices)))
        # print("save tsdf mesh into {}".format(path))

