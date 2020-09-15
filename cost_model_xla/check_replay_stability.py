import argparse
import os
import numpy as np
import re
from collections import defaultdict
from cost_model_xla.xlatools import compile_to_hlo, replay_hlo

parser = argparse.ArgumentParser(description="Compile and execute subgraphs in XLA.")

parser.add_argument("dataset_folder_path", type=str, 
					help="Path to the debug folder.")

parser.add_argument("start_from", type=int, 
					help="Start from this sample.")
parser.add_argument("end_at", type=int, 
					help="End at this sample.")

args_ = parser.parse_args()

dataset_dir = args_.dataset_folder_path
debug_dir = os.path.join(dataset_dir, "debug")

samples = []
with open(os.path.join(dataset_dir, "running_time.txt"), "r") as f:
    for line in f:
        sample_id = int(line.split(":")[0])
        time = float(line.split(":")[1])
        samples.append((sample_id,time))

if args_.end_at > len(samples):
    print("At most {} samples exist.".format(len(samples)))
    exit(0)

diff_all = []
diff_else = []
diff_matmul = []
diff_not_matmul = []
diff_mul = []
diff_not_mul = []
diff_matmul_and_mul = []
diff_matmul_but_not_mul = []
diff_mul_but_not_matmul = []
replayed = []
orig = []
sample_ids_printed = []
for i in range(args_.start_from-1, args_.end_at-1):
    sample_id, recorded_time = samples[i]
    compile_to_hlo(os.path.join(debug_dir,"{}.pbtxt".format(sample_id)), os.path.join(debug_dir,"{}_config.pbtxt".format(sample_id)), os.path.join(debug_dir,"tmp_unopt.txt"), os.path.join(debug_dir,"tmp_opt.txt"))
    time = replay_hlo(os.path.join(debug_dir,"tmp_unopt.txt"))
    print("replayed time: {}, recorded time: {}".format(time*1e6, recorded_time))
    diff = np.abs(time*1e6 - recorded_time) / recorded_time
    diff_all.append(diff)
    with open(os.path.join(debug_dir,"{}.pbtxt".format(sample_id)), "r") as f:
        all_op_types = [raw_str.split("\"")[1] for raw_str in re.findall("op: \".*\"", f.read())]

    stat_counter = defaultdict(int)

    for op_type in all_op_types:
        stat_counter[op_type] += 1

    # if np.abs(time*1e6 - recorded_time) / recorded_time > 1:
    #     replayed.append(time*1e6)
    #     orig.append(recorded_time)
    #     sample_ids_printed.append(sample_id)
    if "BatchMatMulV2" in stat_counter:
        diff_matmul.append(diff)
        if "Mul" not in stat_counter:
            diff_matmul_but_not_mul.append(diff)
        else:
            diff_matmul_and_mul.append(diff)
    else:
        diff_not_matmul.append(diff)
    if "Mul" in stat_counter:
        diff_mul.append(diff)
        if "BatchMatMulV2" not in stat_counter:
            diff_mul_but_not_matmul.append(diff)
    else:
        diff_not_mul.append(diff)
    if "BatchMatMulV2" not in stat_counter and "Mul" not in stat_counter:
        diff_else.append(diff)

# print("="*20)
# print("Sampleid: {}".format(sample_ids_printed))
# print("Replayed: {}".format(replayed))
# print("Original: {}".format(orig))
print("="*20)
print("Overall Difference: {} (len: {})".format(np.average(diff_all), len(diff_all)))
print("Difference w MatMul: {} (len: {})".format(np.average(diff_matmul), len(diff_matmul)))
print("Difference wo MatMul: {} (len: {})".format(np.average(diff_not_matmul), len(diff_not_matmul)))
print("Difference w Mul: {} (len: {})".format(np.average(diff_mul), len(diff_mul)))
print("Difference wo Mul: {} (len: {})".format(np.average(diff_not_mul), len(diff_not_mul)))
print("Difference w MatMul and Mul: {} (len: {})".format(np.average(diff_matmul_and_mul), len(diff_matmul_and_mul)))
print("Difference w MatMul but not Mul: {} (len: {})".format(np.average(diff_matmul_but_not_mul), len(diff_matmul_but_not_mul)))
print("Difference w Mul but not MatMul: {} (len: {})".format(np.average(diff_mul_but_not_matmul), len(diff_mul_but_not_matmul)))
print("Difference wo both: {} (len: {})".format(np.average(diff_else), len(diff_else)))