import json

with open("/Users/chenyu/Downloads/20210127_02_hvd_tf_vgg16_rdma_apply_xla_no_tensor_fusion/combined.json", "r") as f:
    trace = json.load(f)

# one_pid = -1
pids = set()
for ev in trace["traceEvents"]:
    # if ev["ph"] == "M":
    #     if ev["name"] == "process_name" and "name" in ev["args"]:
    #         # if "/job:localhost/replica:0/task:0/device:GPU:" in ev["args"]["name"] \
    #         print(ev["args"]["name"])
    #         if "stream:all" in ev["args"]["name"] \
    #             and "Compute" in ev["args"]["name"]:
    #             one_pid = ev["pid"]
    pids.add(ev["pid"])

evs = sorted(trace["traceEvents"], key=lambda x: x["ts"])

iter_times_pid = {}

for pid in pids:
    source_sts = []
    dep_eds = []
    started = False
    last_assign = -1
    for ev in evs:
        if ev["ph"] == "X" and ev["pid"] == pid:
            # if "args" in ev and ev["args"]["name"] == "_SOURCE":
            if "args" in ev and "ncclAllReduceRingLLKernel" in ev["args"]["name"] and started == False:
                source_sts.append(ev["ts"])
                if last_assign != -1:
                    dep_eds.append(last_assign)
                    last_assign = -1
                started = True
            # elif "args" in ev and ev["args"]["name"] == "group_deps_1":
            elif "args" in ev and "ncclAllReduceRingLLKernel" in ev["args"]["name"]:
                # started = False
                last_assign = ev["ts"] + ev["dur"]
            elif "args" in ev and "GradientDescent" == ev["args"]["name"]:
                started = False
    if last_assign != -1:
        dep_eds.append(last_assign)

    source_sts = sorted(source_sts)
    dep_eds = sorted(dep_eds)

    iter_times = []
    for i in range(len(source_sts)):
        iter_times.append((dep_eds[i] - source_sts[i]) / 1000)
    iter_times_pid[pid] = iter_times

avg = []
avg_per_iter = [float("inf")] * 10
for pid, iter_times in iter_times_pid.items():
    print("PID {}: {}".format(pid, iter_times))
    avg += iter_times
    for idx, time in enumerate(iter_times):
        avg_per_iter[idx] = min(time, avg_per_iter[idx])

print("Average: {}".format(sum(avg) / len(avg)))
print("Average min per iter: {}, details: {}".format(sum(avg_per_iter)/len(avg_per_iter), avg_per_iter))
