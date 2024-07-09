from huggingface_hub import hf_hub_download
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

def download_folder(repo_id, folder, local_dir, files, repo_type="dataset"):# model, dataset, or space.

    def download_file(file):
        cache_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            subfolder=folder,
            # repo_type=repo_type,
            cache_dir=f'{local_dir}/{folder}/_temp',
        )
        
        target_path = f'{local_dir}/{folder}/{file}'
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.system(f'mv {os.path.realpath(cache_file_path)} {target_path}')
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            futures.append(executor.submit(download_file, file))
        for future in futures:
            future.result()


# Example usage
repo_id = "apchen/LaRa"  # Replace with your repository ID
folder_path = "dataset"  # Replace with the path to the folder in the repository
local_dir = "."  # Replace with your local destination directory

gso_list = ['GSO.zip']
co3d_list = ['Co3D/co3d_hydrant.h5','Co3D/co3d_teddybear.h5']
gobjaverse_list = [f'gobjaverse/gobjaverse_part_{i+1:02d}.h5' for i in range(32)] + ['gobjaverse/gobjaverse.h5']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download files.")
    parser.add_argument("dtype", type=str, default="gso", help="one of [gso,co3d,objaverse,all]")
    
    args = parser.parse_args()
    
    if "gso" == args.dtype:
        # download_folder(repo_id, folder_path, local_dir,gso_list)
        os.system(f'unzip {local_dir}/{folder_path}/{gso_list[0]} -d {local_dir}/{folder_path}')
        os.system(f'rm {local_dir}/{folder_path}/{gso_list[0]}')
    elif "co3d" == args.dtype:
        download_folder(repo_id, folder_path, local_dir,co3d_list)
    elif "objaverse" == args.dtype:
        download_folder(repo_id, folder_path, local_dir, gobjaverse_list)
    elif "all" == args.dtype:
        download_folder(repo_id, folder_path, local_dir,gso_list+co3d_list+gobjaverse_list)
        os.system(f'unzip {local_dir}/{folder_path}/{gso_list[0]} -d {local_dir}/{folder_path}')
        os.system(f'rm {local_dir}/{folder_path}/{gso_list[0]}')
        
    # shutil.rmtree(f'{local_dir}/{folder_path}/_temp')
