import argparse
import os
import shutil
import subprocess
from gen_dataset_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='XLA Op fusion dataset generator.')
    parser.add_argument('--freezed_graph_path', required=True, type=str, 
                        help='Path to the freezed graph proto.')
    parser.add_argument('--result_dir', required=True, type=str, 
                        help='Path to save the resulting dataset.')
    parser.add_argument('--num_samples', default=200, type=int, 
                        help='Number of samples to generate.')
    parser.add_argument('--debug', action='store_true',
                        help='If set, output additional debug info.')
    parser.add_argument('--gpu_benchmark_cmd', required=True, type=str, 
                        help='Executable to generate GPU benchmark statistics.')
    parser.add_argument('--min_subgraph_level', default=1, type=int, 
                        help='Lower bound of generated subgraph\'s level.')
    parser.add_argument('--max_subgraph_level', default=8, type=int, 
                        help='Upper bound of generated subgraph\'s level.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    print("Reading profiles for entire graph.")
    op_time_fn = os.path.join(args.result_dir, "op_running_times.pickle")
    with open(op_time_fn, "rb") as f:
        op_time_dict = pickle.load(f)
    gen_dataset(args.freezed_graph_path, op_time_dict, args.gpu_benchmark_cmd, 
                args.result_dir, num_samples=args.num_samples, 
                min_subgraph_level=args.min_subgraph_level,
                max_subgraph_level=args.max_subgraph_level)
    
