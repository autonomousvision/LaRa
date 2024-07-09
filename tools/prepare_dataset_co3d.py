import torch
import torchvision
import numpy as np

import math
import os,h5py
import tqdm
from PIL import Image
from omegaconf import DictConfig

from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields

from multiprocessing import Process, Lock
from pytorch3d.vis import plotly_vis
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.camera_utils import join_cameras_as_batch


CO3D_RAW_ROOT = '/mnt/anpei/Code/splatter-image/dataset/Co3D/downloads' # change to where your CO3D data resides
CO3D_OUT_ROOT = '/mnt/anpei/Code/splatter-image/dataset/Co3D/' # change to your folder here

assert CO3D_RAW_ROOT is not None, "Change CO3D_RAW_ROOT to where your raw CO3D data resides"
assert CO3D_OUT_ROOT is not None, "Change CO3D_OUT_ROOT to where you want to save the processed CO3D data"

"""
Taken from https://github.com/szymanowiczs/viewset-diffusion
"""

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)

from sklearn.cluster import KMeans
def KMean(xyz, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init=10)
    kmeans.fit(xyz)
    labels = kmeans.labels_
    
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels==i)[0]
        clusters.append(idx.astype(np.uint8))

    return clusters

def normalize_sequence(dataset, sequence_name, volume_side_length, vis=False):
    """
    Normalizes the sequence using the point cloud information. Takes 3 steps to normalize
    the cameras so that the point clouds are aligned across sequences.
    1. Normalize translation: shift cameras and point cloud so that COM is at origin
    2. Normalize rotation: using photographer's bias, align the point cloud with y-axis
    3. Normalize scale so that point cloud fits in a cube of side length volume_side_length
    """
    needs_checking = False
    frame_idx_gen = dataset.sequence_indices_in_order(sequence_name)
    frame_idxs = []
    while True:
        try:
            frame_idx = next(frame_idx_gen)
            frame_idxs.append(frame_idx)
        except StopIteration:
            break

    cameras_start = []
    for frame_idx in frame_idxs:
        cameras_start.append(dataset[frame_idx].camera)
    cameras_start = join_cameras_as_batch(cameras_start)
    cameras = cameras_start.clone()

    # ===== Translation normalization
    point_cloud_pts = dataset[frame_idxs[0]].sequence_point_cloud.points_list()[0].clone()
    # find the center of mass
    com = torch.mean(point_cloud_pts, dim=0)
    # center the point cloud
    point_cloud_pts = point_cloud_pts - com
    # shift the cameras accordingly
    cameras.T = torch.matmul(com, cameras.R) + cameras.T
    
    # ===== Rotation normalization
    # Estimate the world 'up' direction assuming that yaw is small
    # and running SVD on the x-vectors of the cameras
    x_vectors = cameras.R.transpose(1, 2)[:, 0, :].clone()
    x_vectors -= x_vectors.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x_vectors)
    V = Vh.mH
    # vector with the smallest variation is to the normal to
    # the plane of x-vectors (assume this to be the up direction)
    if S[0] / S[1] > S[1] / S[2]:
        print('Warning: unexpected singular values in sequence {}: {}'.format(sequence_name, S))
        needs_checking = True
    estimated_world_up = V[:, 2:]
    # check all cameras have the same y-direction
    for camera_idx in range(len(cameras.T)):
        if torch.sign(torch.dot(estimated_world_up[:, 0],
                                cameras.R[0].transpose(0,1)[1, :])) != torch.sign(torch.dot(estimated_world_up[:, 0],
                                    cameras.R[camera_idx].transpose(0,1)[1, :])):
            print("Some cameras appear to be flipped in sequence {}".format(sequence_name) )
            needs_checking = True
    flip = torch.sign(torch.dot(estimated_world_up[:, 0], cameras.R[0].transpose(0,1)[1, :])) < 0
    if flip:
        estimated_world_up = V[:, 2:] * -1
    # build the target coordinate basis using the estimated world up
    target_coordinate_basis = torch.cat([V[:, :1],
                                        estimated_world_up,
                                        torch.linalg.cross(V[:, :1], estimated_world_up, dim=0)],
                                        dim=1)
    cameras.R = torch.matmul(target_coordinate_basis.T, cameras.R)
    point_cloud_pts = torch.bmm(point_cloud_pts.unsqueeze(1),
                                target_coordinate_basis.unsqueeze(0).expand(len(point_cloud_pts),
                                                                            3, 3)).squeeze(1)
    
    # ===== Scale normalization
    # align the center along the longest axis to the origin
    ranges = torch.max(point_cloud_pts, dim=0)[0] - torch.min(point_cloud_pts, dim=0)[0]
    max_range_index = 1 # torch.argmax(ranges)
    aligned_com_dist = torch.max(point_cloud_pts, dim=0)[0][max_range_index] - ranges[max_range_index] / 2
    aligned_com = torch.zeros(3)
    aligned_com[max_range_index] = aligned_com_dist
    # shift cameras and point cloud
    cameras.T = torch.matmul(aligned_com, cameras.R) + cameras.T
    point_cloud_pts = point_cloud_pts - aligned_com

    max_point_cloud = torch.max(torch.abs(point_cloud_pts))

    scaling_factor = volume_side_length * 0.95 / (2 * max_point_cloud )
    point_cloud_pts = point_cloud_pts * scaling_factor
    cameras.T = cameras.T * scaling_factor

    normalized_point_cloud = Pointclouds([point_cloud_pts])
    maximum_distance = torch.max(torch.norm(cameras.T, dim=1))
    minimum_distance = torch.min(torch.norm(cameras.T, dim=1))

    if vis:
        x_axis_points = torch.tensor([[x, 0, 0] for x in torch.linspace(0, 10, 100)])
        y_axis_points = torch.tensor([[0, y, 0] for y in torch.linspace(0, 10, 100)])
        z_axis_points = torch.tensor([[0, 0, z] for z in torch.linspace(0, 10, 100)])

        axis_dict = {"x_axis": Pointclouds([x_axis_points]),
                    "y_axis": Pointclouds([y_axis_points]), 
                    "z_axis": Pointclouds([z_axis_points]),
        }

        fig = plotly_vis.plot_scene({
            sequence_name:{
                # "point cloud before": dataset[frame_idxs[0]].sequence_point_cloud,
                "point cloud after": normalized_point_cloud,
                # "cameras before": cameras_start,
                "cameras after": cameras,
                **axis_dict
            }
        },
        axis_args=plotly_vis.AxisArgs(showgrid=True))
        fig.show()

    return cameras, minimum_distance, maximum_distance, normalized_point_cloud, needs_checking

def update_scores(top_scores, top_names, new_score, new_name):
    for sc_idx, sc in enumerate(top_scores):
        if new_score > sc:
            # shift scores and names to the right, start from the end
            for sc_idx_next in range(len(top_scores)-1, sc_idx, -1):
                top_scores[sc_idx_next] = top_scores[sc_idx_next - 1]
                top_names[sc_idx_next] = top_names[sc_idx_next - 1]
            top_scores[sc_idx] = new_score
            top_names[sc_idx] = new_name
            break
    return top_scores, top_names


def crop_image_at_non_integer_locations(img, 
                                        max_half_side: float, 
                                        principal_point_x: float, 
                                        principal_point_y: float):
    """
    Crops the image so that its center is at the principal point.
    The boundaries are specified by half of the image side. 
    """
    # number of pixels that the image spans. We don't want to resize
    # at this stage. However, the boundaries might be such that
    # the crop side is not an integer. Therefore there will be
    # minimal resizing, but it's extent will be sub-pixel.
    # We don't apply low-pass filtering at this stage and cropping is
    # done with bilinear sampling 
    max_pixel_number = math.floor(2 * max_half_side)
    half_pixel_side = 0.5 / max_pixel_number
    x_locations = torch.linspace(principal_point_x - max_half_side + half_pixel_side,
                                 principal_point_x + max_half_side - half_pixel_side,
                                 max_pixel_number)
    y_locations = torch.linspace(principal_point_y - max_half_side + half_pixel_side,
                                 principal_point_y + max_half_side - half_pixel_side,
                                 max_pixel_number)
    grid_locations = torch.stack(torch.meshgrid(x_locations, y_locations, indexing='ij'), dim=-1).transpose(0, 1)
    grid_locations[:, :, 1] = ( grid_locations[:, :, 1] - img.shape[1] / 2 ) / ( img.shape[1] / 2 )
    grid_locations[:, :, 0] = ( grid_locations[:, :, 0] - img.shape[2] / 2 ) / ( img.shape[2] / 2 )
    image_crop = torch.nn.functional.grid_sample(img.unsqueeze(0), grid_locations.unsqueeze(0))
    return image_crop.squeeze(0)

def write_one_scene(hdf5_file_path, created_dataset, sequence_names, split):

    with h5py.File(hdf5_file_path, 'a') as hdf5_file:
        names = []
        for k,sequence_name in enumerate(sequence_names):
            print(f'{k} of {len(sequence_names)}')
            
            positions, rgbs, c2ws,fovs = [],[],[],[]
            frame_idx_gen = created_dataset.sequence_indices_in_order(sequence_name)
            frame_idxs = []

            # Preprocess cameras with Viewset Diffusion protocol
            normalized_cameras, _, _, _, _ = normalize_sequence(created_dataset, sequence_name, 1.0)
                
            R = normalized_cameras.R
            T = normalized_cameras.T

            while True:
                try:
                    frame_idx = next(frame_idx_gen)
                    frame_idxs.append(frame_idx)
                except StopIteration:
                    break
        
            
            camera_transform_matrix = np.eye(4)
            camera_transform_matrix[0, 0] *= -1
            camera_transform_matrix[1, 1] *= -1
            
            idx = 0
            
            # Preprocess images
            for i, frame_idx in enumerate(frame_idxs):
                # Read the original uncropped image
                frame = created_dataset[frame_idx]
                rgb_image = torchvision.transforms.functional.pil_to_tensor(
                    Image.open(frame.image_path)).float() / 255.0
                # ============= Foreground mask =================
                # Initialise the foreground mask at the original resolution
                fg_probability = torch.zeros_like(rgb_image)[:1, ...]
                # Find size of the valid region in the 800x800 image (non-padded)
                resized_image_mask_boundary_y = torch.where(frame.mask_crop > 0)[1].max() + 1
                resized_image_mask_boundary_x = torch.where(frame.mask_crop > 0)[2].max() + 1
                # Resize the foreground mask to the original scale
                x0, y0, box_w, box_h = frame.crop_bbox_xywh
                resized_mask = torchvision.transforms.functional.resize(
                    frame.fg_probability[:, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x],
                    (box_h, box_w),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    )
                # Fill in the foreground mask at the original scale in the correct location based
                # on where it was cropped
                fg_probability[:, y0:y0+box_h, x0:x0+box_w] = resized_mask

                # ============== Crop around principal point ================
                # compute location of principal point in Pytorch3D NDC coordinate system in pixels
                # scaling * 0.5 is due to the NDC min and max range being +- 1
                principal_point_cropped = frame.camera.principal_point * 0.5 * frame.image_rgb.shape[1]
                # compute location of principal point from top left corer, i.e. in image grid coords
                scaling_factor = max(box_h, box_w) / 800
                principal_point_x = (frame.image_rgb.shape[2] * 0.5 - principal_point_cropped[0, 0]) * scaling_factor + x0
                principal_point_y = (frame.image_rgb.shape[1] * 0.5 - principal_point_cropped[0, 1]) * scaling_factor + y0
                # Get the largest center-crop that fits in the foreground
                max_half_side = get_max_box_side(
                    frame.image_size_hw, principal_point_x, principal_point_y)
                # After this transformation principal point is at (0, 0)
                
                rgb = crop_image_at_non_integer_locations(rgb_image, max_half_side, 
                                                            principal_point_x, principal_point_y)          
                fg_probability_cc = crop_image_at_non_integer_locations(fg_probability, max_half_side, 
                                                            principal_point_x, principal_point_y)
                assert frame.image_rgb.shape[1] == frame.image_rgb.shape[2], "Expected square images"
                breakpoint()
                # bad segmentation
                if torch.sum(fg_probability_cc>0.5) < 0.1*fg_probability_cc.shape[-1]*fg_probability_cc.shape[-2]:
                    continue
                
                # =============== Resize and save =======================
                # Resize raw rgb
                img_width = 512
                pil_rgb = torchvision.transforms.functional.to_pil_image(rgb)
                pil_rgb = torchvision.transforms.functional.resize(pil_rgb,
                                                    img_width,
                                                    interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
                rgb = torchvision.transforms.functional.pil_to_tensor(pil_rgb) / 255.0
                
                
                # Resize mask
                fg_probability_cc = torchvision.transforms.functional.resize(fg_probability_cc,
                                                    img_width,
                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                rgba = torch.cat((rgb,fg_probability_cc),dim=0).permute(1,2,0).contiguous().float().numpy()
                rgba = np.round(rgba*255).astype('uint8')
                rgbs.append(rgba)
                
                
                w2c = np.eye(4)
                w2c[:3,:3],w2c[3:,:3] = R[i],T[i]

                w2c = np.transpose(np.matmul(w2c, camera_transform_matrix))
                
                # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                # T = w2c[:3, 3]
        
                # w2c = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
                # c2w =getView2World(R, T).astype(np.float32)
                c2w = np.linalg.inv(w2c)
                
                c2ws.append(c2w)
                positions.append(c2w[:3,3])
                
                # ============== Intrinsics transformation =================
                # Transform focal length according to the crop
                # Focal length is in NDC conversion so we do not need to change it when resizing
                # We should transform focal length to non-cropped image and then back to cropped but
                # the scaling factor of the full non-cropped image cancels out.
                focal_lengths_ndc = frame.camera.focal_length * max(box_h, box_w) / (2 * max_half_side)
                focal_lengths_px = (focal_lengths_ndc * img_width / 2).view(-1).tolist()
      
                FovY = focal2fov(focal_lengths_px[0], img_width) 
                FovX = focal2fov(focal_lengths_px[1], img_width)
                fov = np.array([FovX,FovY]).astype(np.float32)
                # intrinsic[[0,1],[0,1]] = focal_lengths_px
                # intrinsic[:2,2] = img_width / 2
                
                fovs.append(fov)
                
                idx += 1

            print(sequence_name, len(rgbs),len(c2ws),len(fovs),len(positions))
            if len(rgbs)==len(c2ws)==len(fovs)==len(positions) and len(rgbs)>10:
                if sequence_name in hdf5_file:
                    del hdf5_file[sequence_name]
                grp = hdf5_file.create_group(sequence_name)
            
                for idx in range(len(rgbs)):
                    grp.create_dataset(f'image_{idx}', data=rgbs[idx], compression='gzip', compression_opts=4)
                    grp.create_dataset(f'c2w_{idx}', data=c2ws[idx])
                    grp.create_dataset(f'fov_{idx}', data=fovs[idx])
                group_id = grp.create_group('groups')
                for n_groups in [2,3,4,5,6]:

                    g_id = KMean(np.stack(positions), n_clusters=n_groups)
                    for i in range(n_groups):
                        group_id.create_dataset(f'groups_{n_groups}_{i}', data=g_id[i])
                names.append(sequence_name)
        
        if 'splits' not in hdf5_file:
            grp = hdf5_file.create_group('splits')
        else:
            grp = hdf5_file['splits']
        grp.create_dataset(split, data=names)


def merge_h5_files(save_path, hdf5_file_path, N, splits):
    
    def copy_group(source_group, dest_group):
        """Recursively copy all items from source_group to dest_group"""
        for name, item in source_group.items():
     
            if isinstance(item, h5py.Dataset):
                # Copy dataset from source to destination
                if name in dest_group:
                    continue
                dest_group.copy(item, name)
            elif isinstance(item, h5py.Group):
                # Create new group in destination and copy contents
                if name in dest_group:
                    del dest_group[name]
                    continue
                new_group = dest_group.create_group(name)
                copy_group(item, new_group)
                
    with h5py.File(save_path, 'w') as dest_h5:
        for i in range(N):
            with h5py.File( f'{hdf5_file_path}_{i:02d}.hdf5','r')  as source_h5:
                copy_group(source_h5, dest_h5)
        
        grp = dest_h5.create_group('splits')
        for name,item in splits.items():
            grp.create_dataset(name, data=item)
                        
def main(category):

    subset_name = "fewview_dev"

    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_root=CO3D_RAW_ROOT,
        dataset_JsonIndexDataset_args=DictConfig(
            {"remove_empty_masks": False, "load_point_clouds": True}
        ),
    ).get_dataset_map()

    splits = {}
    for split in ["train"]:# "train",

        created_dataset = dataset_map[split]

        sequence_names = [k for k in created_dataset.seq_annots.keys()]
        max_len = max(len(s) for s in sequence_names)
        str_dtype = 'S{}'.format(max_len)
        # splits[split] = np.array(sequence_names, dtype=str_dtype)

        """
        Convert folders of images and camera poses to an HDF5 file.
        """

        # lock = Lock()
        processes = []
        num_processes = 1  # For example, create 5 processes

        file_chunks = np.array_split(sequence_names, num_processes)

        hdf5_file_path = os.path.join(CO3D_OUT_ROOT, "co3d_{}".format(category))
        # os.makedirs(hdf5_file_path, exist_ok=True)
        
        # # Initialize and start multiple processes
        # for i in range(num_processes):
        #     p = Process(target=write_one_scene, args=(f'{hdf5_file_path}_{i:02d}.hdf5',created_dataset, file_chunks[i]))
        #     p.start()
        #     processes.append(p)

        # # Wait for all processes to finish
        # for p in processes:
        #     p.join()
        
        write_one_scene(f'{hdf5_file_path}.hdf5', created_dataset, sequence_names, split)
        # splits[split] = names

    # save_path = os.path.join(CO3D_OUT_ROOT, "co3d_{}.hdf5".format(category))
    # merge_h5_files(save_path, hdf5_file_path, num_processes, splits)
        

def get_max_box_side(hw, principal_point_x, principal_point_y):
    # assume images are always padded on the right - find where the image ends
    # find the largest center crop we can make
    max_x = hw[1] # x-coord of the rightmost boundary
    min_x = 0.0 # x-coord of the leftmost boundary
    max_y = hw[0] # y-coord of the top boundary
    min_y = 0.0 # y-coord of the bottom boundary

    max_half_w = min(principal_point_x - min_x, max_x - principal_point_x) 
    max_half_h = min(principal_point_y - min_y, max_y - principal_point_y) 
    max_half_side = min(max_half_h, max_half_w)

    return max_half_side


if __name__ == "__main__":
    for category in ["teddybear", "hydrant"]:#
        bad_sequences_val = main(category)

