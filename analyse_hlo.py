import os
import code
import pickle
import subprocess
import re

hlo_stats = {}
fns = os.listdir("/root/dataset_no_small_graphs/dataset/hlos/")
sample_ids = []
has_gemm_counter = 0
has_rng_counter = 0
for fn in fns:
    sample_id = int(fn.strip(".txt").strip("hlo_"))
    hlo_stats[sample_id] = {}
    hlo_stats[sample_id]["custom_call"] = False
    hlo_stats[sample_id]["rng"] = False
    with open(os.path.join("/root/dataset_no_small_graphs/dataset/hlos/", fn), "r") as f:
        for line in f:
            if "custom-call" in line:
                if hlo_stats[sample_id]["custom_call"] == False:
                    hlo_stats[sample_id]["custom_call"] = True
                    has_gemm_counter += 1
            if "rng-get-and-update-state" in line:
                if hlo_stats[sample_id]["rng"] == False:
                    hlo_stats[sample_id]["rng"] = True
                    has_rng_counter += 1
            if hlo_stats[sample_id]["custom_call"] and hlo_stats[sample_id]["rng"]:
                break
    sample_ids.append(int(fn.strip(".txt").strip("hlo_")))

with open("/root/hlo_stats.pickle", "wb") as f:
    pickle.dump((sample_ids, hlo_stats), f)

print("Have custom calls: {}, total: {}".format(has_gemm_counter, len(sample_ids)))
print("Have rng: {}, total: {}".format(has_rng_counter, len(sample_ids)))

for fn in fns:
    full_path = os.path.join("/root/dataset_no_small_graphs/dataset/hlos/", fn)
    analyse_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/byteprofile_xlatools/analyse_hlo"
    process = subprocess.run([analyse_exec, full_path], capture_output=True)
    output = process.stderr.decode("ascii")
    regex = re.compile("Target String: .*$", re.MULTILINE)
    names = [line.split()[2] for line in regex.findall(output)]
    for name in names:
        if name != "__cublas$gemm":
            print(fn + ": " + name)

code.interact(local=locals())

