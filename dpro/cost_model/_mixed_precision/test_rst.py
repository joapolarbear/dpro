''' This script is used to collect TF AMP strategy and test the mixed precision search resutls'''
import re
import os
import json
import argparse

parser = argparse.ArgumentParser(description="AMP Parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--option", type=str, required=True, help="option.")
parser.add_argument("--env", type=str, default="", help="environment.")
parser.add_argument("--cmd", type=str, default=None, help="command.")

parser.add_argument("--amp_rst_path", type=str, default=None, help="amp_rst_path.")
parser.add_argument("--search_rst_path", type=str, default=None, help="amp_rst_path.")
parser.add_argument("--timeline_path", type=str, default=None, help="timeline_path.")
parser.add_argument("--target_path", type=str, default=None, help="target_path.")

args = parser.parse_args()

def read_amp_fp16_ops(_path):
    with open(_path, "r") as f:
        fp16_ops = json.load(f)
        fp16_ops = fp16_ops['names']
        if "DT_HALF" in fp16_ops:
            fp16_ops = [l.split("node ")[1].split(" to DT_HALF")[0] for l in fp16_ops]
        return fp16_ops

def read_search_fp16_ops(_path):
    with open(_path, "r") as f:
        fp16_ops = json.load(f)
        fp16_ops = fp16_ops['best_strategy']
        fp16_ops = [n[1].split("->")[1].split(".")[1] for n in fp16_ops]
        return fp16_ops

if args.option == "parse":
    os.system("rm nohup.out")
    env = ""
    if len(args.env) > 0:
        env = " ".join(args.env.split(","))
    if args.cmd is None:
        cmd = "python3 /opt/tiger/horovod_examples/tensorflow_synthetic_benchmark.py --num-warmup-batches 1 --num-batches-per-iter 1 --num-iters 1 --amp"
    else:
        cmd = args.cmd
    
    print(env + " TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MIN_VLOG_LEVEL=2 nohup {}".format(cmd))
    os.system(env + " TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MIN_VLOG_LEVEL=2 nohup {}".format(cmd))
    
    with open("nohup.out", 'r') as f:
        result = f.read()

    ret = {}
    lines = re.findall("Converted [0-9]+/[0-9]+ nodes to "
        "float16 precision using [0-9]+ cast\(s\) to "
        "float16 \(excluding Const and Variable casts\)", result)
    if len(lines) > 0:
        print(lines[0])
        ret["info"] = lines[0]

    lines = re.findall("Changing type .+ of "
                    ".+ node .+ to DT_HALF", result)
    print("check change {} nodes type".format(len(lines)))

    ret["names"] = [l.split("node ")[1].split(" to DT_HALF")[0] for l in lines]
    with open("amp_result.json", "w") as f:
        json.dump(ret, f)
    os.system("rm nohup.out")
elif args.option == "paint":
    with open(args.timeline_path, "r") as f:
        traces = json.load(f)

    fp16_ops_list = [read_amp_fp16_ops(args.amp_rst_path), read_search_fp16_ops(args.search_rst_path)]

    rst_traces = []
    one_pid = None
    for trace in traces:
        if "Comm" in trace["name"]:
            continue
        if one_pid is None:
            one_pid = trace["pid"]
        elif one_pid != trace["pid"]:
            continue

        if trace["args"]["step"] != 0:
            continue
            
        raw_name = trace["name"].split(".")[1]

        is_fp16 = [False, False]

        new_trace = trace.copy()
        if raw_name in fp16_ops_list[0]:
            new_trace["name"] = "Single float16"
            is_fp16[0] = True
        else:
            new_trace["name"] = "Double float32"
        new_trace["pid"] = "TF AMP"
        new_trace["tid"] ="default"
        rst_traces.append(new_trace)

        new_trace = trace.copy()
        if raw_name in fp16_ops_list[1]:
            new_trace["name"] = "Single float16"
            is_fp16[1] = True
        else:
            new_trace["name"] = "Double float32"
        new_trace["pid"] = "Search Result"
        new_trace["tid"] ="default"
        rst_traces.append(new_trace)

        if is_fp16[0] != is_fp16[1]:
            print("{} - TF_AMP:{}, Search Result:{}".format(raw_name,
                    "float16" if is_fp16[0] else "float32",
                    "float16" if is_fp16[1] else "float32"))

    with open(args.target_path, "w") as f:
        json.dump(rst_traces, f)
    
elif args.option == "show":
    if (args.amp_rst_path is not None and args.search_rst_path is not None) or (args.amp_rst_path is None and args.search_rst_path is None):
        raise ValueError("Please input one and only one path")

    if args.amp_rst_path is not None:
        fp16_ops = read_amp_fp16_ops(args.amp_rst_path)
        print("TF AMP: ", ",".join(fp16_ops))

    if args.search_rst_path is not None:
        fp16_ops = read_search_fp16_ops(args.search_rst_path)
        print("Search Results: ", ",".join(fp16_ops))

    if args.target_path is not None:
        with open(args.target_path, 'w') as f:
            for op in fp16_ops:
                f.write(op + "\n")


