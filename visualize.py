import os

fns = os.listdir("/root/dataset_no_small_graphs/dataset/feature_vecs")
exec_times = {}
with open("/root/dataset_no_small_graphs/running_time.txt", "r") as f:
        for line in f:
                sid, time = line.split(":")
                sid = int(sid)
                time = float(time)
                exec_times[sid] = time
feature_vecs = []
labels = []

for fn in fns:
        sid = int(fn.split(".txt")[0])
        if sid in exec_times:
                fv = []
                with open(os.path.join("/root/dataset_no_small_graphs/dataset/feature_vecs", fn), "r") as f:
                        for line in f:
                                fv.append(float(line))
                feature_vecs.append(fv)
                labels.append(str(sid) + ":" + str(exec_times[sid]))

with open("/root/vectors.tsv", "w") as f:
        for fv in feature_vecs:
                for value in fv:
                        f.write(str(value))
                        f.write("\t")
                f.write("\n")

with open("/root/vectors_meta.tsv", "w") as f:
        for fv in labels:
                f.write(fv)
                f.write("\n")