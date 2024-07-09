import h5py
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def split_hdf5_file(input_file, output_prefix, num_splits):
    with h5py.File(input_file, 'r') as f:
        keys = sorted(list(f.keys()))
        chunk_size = len(keys) // num_splits
        
        def write_chunk(i, keys_chunk):
            output_file = f"{output_prefix}_part_{i+1}.h5"
            with h5py.File(output_file, 'w') as out_f:
                for key in keys_chunk:
                    f.copy(key, out_f)

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(16):
                keys_chunk = keys[i*chunk_size: (i+1)*chunk_size]
                futures.append(executor.submit(write_chunk, i, keys_chunk))
            for future in futures:
                future.result()
                
    print(f"Split into {num_splits} files with prefix '{output_prefix}'.")

def merge_hdf5_files(output_file, input_files):
    with h5py.File(output_file, 'w') as out_f:
        def copy_data(input_file):
            with h5py.File(input_file, 'r') as in_f:
                for key in in_f.keys():
                    in_f.copy(key, out_f)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(copy_data, input_file) for input_file in input_files]
            for future in futures:
                future.result()

    print(f"Merged files into '{output_file}'.")

def get_absolute_paths(directory, prefix):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and merge HDF5 files.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    split_parser = subparsers.add_parser("split", help="Split an HDF5 file into multiple files.")
    split_parser.add_argument("input_file", type=str, help="Input HDF5 file to split.")
    split_parser.add_argument("output_prefix", type=str, help="Output prefix for split files.")
    split_parser.add_argument("num_splits", type=int, help="Number of splits.")
    
    merge_parser = subparsers.add_parser("merge", help="Merge multiple HDF5 files into one file.")
    merge_parser.add_argument("output_file", type=str, help="Output HDF5 file to create.")
    merge_parser.add_argument("file_prefix", type=str, help="Input HDF5 files to merge.")
    
    args = parser.parse_args()
    
    if args.command == "split":
        split_hdf5_file(args.input_file, args.output_prefix, args.num_splits)
    elif args.command == "merge":
        input_files = get_absolute_paths(args.input_directory, args.file_prefix)
        merge_hdf5_files(args.output_file, input_files)
