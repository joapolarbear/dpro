import argparse
import os
import numpy as np
from cost_model_xla.xlatools import compile_to_hlo, replay_hlo
import subprocess
import re

parser = argparse.ArgumentParser(description="Compile and execute subgraphs in XLA.")

parser.add_argument("dataset_folder_path", type=str, 
					help="Path to the debug folder.")

parser.add_argument("sample_id", type=int,
					help="Run this sample.")

parser.add_argument("n_runs", type=int,
					help="Run this sample n times.")

args_ = parser.parse_args()

dataset_dir = args_.dataset_folder_path
debug_dir = os.path.join(dataset_dir, "debug")

compile_to_hlo(os.path.join(debug_dir,"{}.pbtxt".format(args_.sample_id)), os.path.join(debug_dir,"{}_config.pbtxt".format(args_.sample_id)), os.path.join(debug_dir,"tmp_unopt.txt"), os.path.join(debug_dir,"tmp_opt.txt"))
replay_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu"
opt_1 = "--num_runs={}".format(args_.n_runs)
opt_2 = "--use_fake_data=true"
opt_3 = "--print_result=false"
process = subprocess.run([replay_exec, opt_1, opt_2, opt_3, os.path.join(debug_dir,"tmp_unopt.txt")], capture_output=True)
output = process.stderr.decode("ascii")
times = [float(line.split()[3][:-2]) for line in re.findall("Done executing in .*s:", output)]
for t in times:
    print(t * 1e6)
