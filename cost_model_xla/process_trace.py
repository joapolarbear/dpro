import gzip
import json
import os
import time

TRACE_SUFFIX = "trace.json.gz"
XLA_DUMP_SUFFIX = "after_optimizations.txt"

def search_for_file(profile_dir, suffix):
    for dir_path, dir_names, file_names in os.walk(profile_dir):
        for fn in file_names:
            if fn.endswith(suffix):
                return os.path.join(dir_path, fn)
    return None

def wait_for_file(profile_dir, suffix):
    for i in range(20):
        file_path = search_for_file(profile_dir, suffix)
        if file_path is not None:
            return file_path
        else:
            # sleep 10 ms
            time.sleep(0.01)
    file_path = search_for_file(profile_dir, suffix)
    if file_path is None:
        print("[WARNING] Cannot find file with suffix {} in dir {}.".format(suffix, profile_dir))
        return None
    else:
        return file_path

def search_for_trace(profile_dir):
    return wait_for_file(profile_dir, TRACE_SUFFIX)

def search_for_hlo(xla_dir):
    return wait_for_file(xla_dir, XLA_DUMP_SUFFIX)

def get_execution_time_for_whole_graph(trace_path):
    with gzip.open(trace_path, "rb") as f:
        trace_data = json.loads(f.read().decode("ascii"))
        events = trace_data["traceEvents"]
    time_dict = {}
    for ev in events:
        if "args" not in ev:
            continue
        if "long_name" not in ev["args"]:
            continue
        long_name = ev["args"]["long_name"].split(":")[0]
        if long_name not in time_dict:
            time_dict[long_name] = (0, 0)
        time, count = time_dict[long_name]
        time_dict[long_name] = (time + ev["dur"], count + 1)
    for name, (time, count) in time_dict.items():
        time_dict[name] = (time / count, count)
    return time_dict

def get_execution_time_from_temp_trace(trace_path):
    one_pid = None
    with open(trace_path, "r") as f:
        trace = json.load(f)
    if isinstance(trace, dict):
        trace = trace["traceEvents"]
    for tr in trace:
        if tr["ph"] == "M" and tr["name"] == "process_name":
            if "args" in tr and "name" in tr["args"]:
                if "device:GPU" in tr["args"]["name"] and "Compute" in tr["args"]["name"] and "replica" in tr["args"]["name"]:
                    one_pid = tr["pid"]
                    break
    time_dict = {}
    for tr in trace:
        try:
            if tr["ph"] == "X" and tr["pid"] == one_pid:
                op_name = tr["args"]["name"]
                if op_name not in time_dict:
                    time_dict[op_name] = []
                time_dict[op_name].append(tr["dur"])
        except:
            pass
    for key, durs in time_dict.items():
        time_dict[key] = (sum(durs) / len(durs), len(durs))
    return time_dict

def get_execution_time_from_trace(trace_path):
    with gzip.open(trace_path, "rb") as f:
        trace_data = json.loads(f.read().decode("ascii"))
    events = trace_data["traceEvents"]
    time = 0
    count = 0
    for ev in events:
        try:
            if ev["ph"] == "X" and ev["name"] == "_XlaRun":
                time += ev["dur"]
                count += 1
        except:
            pass
    if count == 0:
        # cannot compile
        return 0, 0
    return time/count, count

def get_execution_time_from_uncompiled_trace(trace_path):
    with gzip.open(trace_path, "rb") as f:
        trace_data = json.loads(f.read().decode("ascii"))
    events = trace_data["traceEvents"]
    time = 0
    count = 0
    for ev in events:
        try:
            if ev["ph"] == "X" and ev["name"] == "SessionRun":
                time += ev["dur"]
                count += 1
        except:
            pass
    if count == 0:
        # cannot compile
        return 0, 0
    return time/count, count
