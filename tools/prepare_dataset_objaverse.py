
import os,io
import h5py
import numpy as np
from PIL import Image
import json,shutil
from tqdm import tqdm
from sklearn.cluster import KMeans
from multiprocessing import Process, Lock
import  tarfile
import imageio.v2 as imageio

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def KMean(xyz, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init=10)
    kmeans.fit(xyz)
    labels = kmeans.labels_
    
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels==i)[0]
        clusters.append(idx.astype(np.uint8))

    return clusters

def check_camera_poses(folder_path):
    for i in range(40):
        if i==25 or i==26:
            continue
        file_path =f'{folder_path}/{i:05d}/{i:05d}.json'
        if not os.path.exists(file_path):
            return False
    
    return True

def load_camera_poses(my_tarfile, name):
    """
    Load camera poses from a file. Modify this function based on the format of your camera poses file.
    """
    # Example: Load camera poses from a JSON file
    # You need to replace this with the appropriate method to load your camera poses
    poses = []
    for i in range(40):
        if i==25 or i==26:
            continue
        try:
            pose = {}
            c2w = np.eye(4).astype('float32')
            file_path =f'{name}/campos_512_v4/{i:05d}/{i:05d}.json'
            my_tarfile.extract(file_path, path=f'temp')
            
            with open(f'temp/{file_path}', 'r') as file:
                temp = json.load(file)
            pose['fov'] = np.array([temp['x_fov'],temp['y_fov']],dtype=np.float32)
            c2w[:3,:3] = np.stack((temp['x'],temp['y'],temp['z']),axis=1)
            c2w[:3,3] = np.array(temp['origin'],dtype=np.float32)
            pose['bbox'] = np.array(temp['bbox'],dtype=np.float32)
            pose['c2w'] = c2w
            poses.append(pose)
        except:
            return
    return poses

def check_images_from_folder(folder_path):
    for i in range(40):
        if i==25 or i==26:
            continue
        img_path =f'{folder_path}/{i:05d}/{i:05d}.png'

        if not os.path.exists(img_path):
            return False
    
    return True

def load_images_from_folder(my_tarfile, name, load_normal=False):
    """
    Load all images in a folder into a list of numpy arrays.
    """
    images = []
    normals = []

    for i in range(40):
        if i==25 or i==26:
            continue
        img_path =f'{name}/campos_512_v4/{i:05d}/{i:05d}.png'
        normal_path =f'{name}/campos_512_v4/{i:05d}/{i:05d}_nd.exr'

        try:
            img_data = io.BytesIO(my_tarfile.extractfile(img_path).read())
            img = Image.open(img_data)
            images.append(np.array(img, dtype=np.uint8))
            # images.append(np.array(my_tarfile.extractfile(img_path).read(), dtype=np.uint8))
            # with Image.open(img_path) as img:
            #     # images.append(np.array(img.resize((size, size)), dtype=np.uint8))
            #     images.append(np.array(img, dtype=np.uint8))
            #     # print(images[-1].shape)
            if load_normal:
                
                my_tarfile.extract(normal_path, path=f'temp')
                normald = cv2.imread(f'temp/{normal_path}',-1)
                # normald = np.array(my_tarfile.extractfile(normal_path).read())
                normal = normald[...,:3]
                normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
                normal = normal / normal_norm
                normal = normal[...,[2,0,1]]
                normal[...,[0,1]] = -normal[...,[0,1]]
                normals.append(((normal+1)/2*255).astype('uint8'))
        except:
            return None, None
    return images, normals

def write_one_folder(folders, hdf5_file_path, dele_folder=False):
    
    for k, folder in enumerate(list(folders)):
        name = os.path.basename(folder).split('.')[0]
        
        my_tarfile = tarfile.open(folder)
        images, normals = load_images_from_folder(my_tarfile, name, True)
        poses = load_camera_poses(my_tarfile, name) # Adjust the file name as needed
        
        if os.path.exists(f'temp/{name}'):
            shutil.rmtree(f'temp/{name}')
        
        if images is None or poses is None or normals is None:
            print(folder)
            continue

        positions = [pos[f'c2w'][:3,3] for pos in poses]
            
        # with lock:
        with h5py.File(hdf5_file_path, 'a') as hdf5_file:

            # Check if the group already exists
            if name in hdf5_file:
                # Delete the existing group
                del hdf5_file[name]
        
            grp = hdf5_file.create_group(name)
            for i in range(38):
                grp.create_dataset(f'image_{i}', data=images[i], compression='gzip', compression_opts=4)
                grp.create_dataset(f'normal_{i}', data=normals[i], compression='gzip', compression_opts=4)
                grp.create_dataset(f'c2w_{i}', data=poses[i]['c2w'])
                grp.create_dataset(f'fov_{i}', data=poses[i]['fov'])
            grp.create_dataset(f'bbox', data=poses[0]['bbox'])

            group_id = grp.create_group('groups')
            for n_groups in [2,3,4,5,6]:
                g_id = KMean(np.stack(positions), n_clusters=n_groups)
                for i in range(n_groups):
                    group_id.create_dataset(f'groups_{n_groups}_{i}', data=g_id[i])

            if dele_folder:
                os.remove(folder)
                
        print(f'{k} of {len(folders)}')
            
def convert_to_hdf5(folders, hdf5_file_path, dele_folder=False):
    """
    Convert folders of images and camera poses to an HDF5 file.
    """
    
    # lock = Lock()
    processes = []
    num_processes = 1  # For example, create 5 processes
    
    file_chunks = np.array_split(folders, num_processes)
    
    # Initialize and start multiple processes
    for i in range(num_processes):
        p = Process(target=write_one_folder, args=(file_chunks[i], f'{hdf5_file_path}_{i:02d}.hdf5', dele_folder))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
   
    # write_one_folder(lock, folders, hdf5_file_path, dele_folder)
    
def check_complete(folders):
    """
    Convert folders of images and camera poses to an HDF5 file.
    """
    missed_folders = []
    for folder in tqdm(folders):
        folder_path = os.path.join(*folder.split('/')[-2:])

        complete = check_images_from_folder(folder)
        if not complete:
            missed_folders.append(folder_path)
            continue
        complete = check_camera_poses(os.path.join(folder)) # Adjust the file name as needed
        if not complete:
            missed_folders.append(folder_path)
            continue
    return missed_folders

def get_all_folders(root):
    all_folders = []
    categrey = os.listdir(root)
    for item in categrey:
        if not os.path.isdir(f'{root}/{item}'):
            continue
        folders = os.listdir(f'{root}/{item}')
        all_folders += [f'{root}/{item}/{folder}' for folder in folders]
    return all_folders

def merge_h5_files(save_path, hdf5_file_path, N):
    
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
                
size = 512
source_root = 'path/to/gobjaverse/folder'
output_path = 'path/to/target'
folders = sorted(get_all_folders(source_root))[:10]

# Example usage
convert_to_hdf5(folders, output_path, dele_folder=False)

