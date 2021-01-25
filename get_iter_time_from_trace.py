import json

with open("/Users/chenyu/temp_1_OPT.json", "r") as f:
    trace = json.load(f)

one_pid = -1
for ev in trace["traceEvents"]:
    # if ev["ph"] == "M":
    #     if ev["name"] == "process_name" and "name" in ev["args"]:
    #         # if "/job:localhost/replica:0/task:0/device:GPU:" in ev["args"]["name"] \
    #         print(ev["args"]["name"])
    #         if "stream:all" in ev["args"]["name"] \
    #             and "Compute" in ev["args"]["name"]:
    #             one_pid = ev["pid"]
    one_pid = ev["pid"]
    break

assert one_pid != -1
print("one_pid: {}".format(one_pid))

evs = sorted(trace["traceEvents"], key=lambda x: x["ts"])

source_sts = []
dep_eds = []
started = False
last_assign = -1
for ev in evs:
    if ev["ph"] == "X" and ev["pid"] == one_pid:
        # if "args" in ev and ev["args"]["name"] == "_SOURCE":
        if "args" in ev and "random_uniform" in ev["args"]["name"] and started == False:
            source_sts.append(ev["ts"])
            if last_assign != -1:
                dep_eds.append(last_assign)
                last_assign = -1
            started = True
        # elif "args" in ev and ev["args"]["name"] == "group_deps_1":
        elif "args" in ev and "GradientDescent" == ev["args"]["name"]:
            started = False
            last_assign = ev["ts"] + ev["dur"]
if last_assign != -1:
    dep_eds.append(last_assign)

source_sts = sorted(source_sts)
dep_eds = sorted(dep_eds)

# import code
# code.interact(local=locals())

iter_times = []
for i in range(len(source_sts)):
    iter_times.append((dep_eds[i] - source_sts[i]) / 1000)

for time in iter_times:
    print(time, end=" ")
print()

