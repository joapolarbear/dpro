import json

with open("/Users/chenyu/temp_1_NONE.json", "r") as f:
    trace = json.load(f)

one_pid = -1
for ev in trace["traceEvents"]:
    if ev["ph"] == "M":
        if ev["name"] == "process_name" and "name" in ev["args"]:
            if "/job:localhost/replica:0/task:0/device:GPU:" in ev["args"]["name"] \
                and "Compute" in ev["args"]["name"]:
                one_pid = ev["pid"]

assert one_pid != -1
print("one_pid: {}".format(one_pid))

source_sts = []
dep_eds = []
for ev in trace["traceEvents"]:
    if ev["ph"] == "X" and ev["pid"] == one_pid:
        if "args" in ev and ev["args"]["name"] == "_SOURCE":
            source_sts.append(ev["ts"])
        elif "args" in ev and ev["args"]["name"] == "group_deps_1":
            dep_eds.append(ev["ts"] + ev["dur"])

source_sts = sorted(source_sts)
dep_eds = sorted(dep_eds)

iter_times = []
for i in range(len(source_sts)):
    iter_times.append((dep_eds[i] - source_sts[i]) / 1000)

for time in iter_times:
    print(time, end=" ")
print()

