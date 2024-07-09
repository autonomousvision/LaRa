# Copyright (c) Alibaba, Inc. and its affiliates.

import os, sys, json
from multiprocessing import Pool

def download_url(item):
    global save_dir
    oss_full_dir = os.path.join("https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse_tar", item+".tar")
    os.system("wget -P {} {}".format(os.path.join(save_dir, item.split("/")[0]), oss_full_dir))

def get_all_folders(root):
    all_folders = []
    categrey = os.listdir(root)
    for item in categrey:
        if not os.path.isdir(f'{root}/{item}'):
            continue
        folders = os.listdir(f'{root}/{item}')
        all_folders += [f'{root}/{item}/{folder}' for folder in folders]
    return all_folders

def folder_to_json(exist_files):
    files = []
    for item in exist_files:
        split = item.split('/')[-2:]
        files.append(f'{split[0]}/{split[1][:-4]}')
    return files
    
def filterout_existing(json, exist_files):
    for item in exist_files:
        json.remove(item)
    return json
    
if __name__=="__main__":
    # download_gobjaverse_280k index file
    # wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k.json   
    assert len(sys.argv) == 4, "eg: python download_objaverse.py ./data /path/to/json_file 10"
    save_dir = str(sys.argv[1])
    json_file = str(sys.argv[2])
    n_threads = int(sys.argv[3])

    data = json.load(open(json_file))[:100]
    
    exist_files = get_all_folders(save_dir)
    exist_files = folder_to_json(exist_files)
    
    print(len(data))
    data = filterout_existing(data, exist_files)
    print(len(data))
    
    p = Pool(n_threads)
    p.map(download_url, data)
