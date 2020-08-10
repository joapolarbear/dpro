from cost_model import *
from gen_dataset_utils import run_gpu_profile
import json
import pickle
import re
from sklearn.model_selection import train_test_split
import code

print("Loading dataset.")
dataset = XlaDataset("/root/dataset_1")
print("Loaded dataset.")
dataset.dump_dataset("/root/dataset_1/dataset.pickle")
print("Dataset dumped.")

dataset_loaded = XlaDataset("/root/dataset_1/dataset.pickle")
print("Dataset loaded from dump.")
cost_model = FusionCostModel("/root/dataset_1/cost_model_tmp")
print("Costmodel setting feature mask.")
cost_model.set_feature_mask(dataset_loaded.feature_mask)
print("Costmodel setting op time dict.")
cost_model.set_op_time_dict(dataset_loaded.op_time_dict)
print("Costmodel running gpu profile.")
gflops, gbps = run_gpu_profile("/root/byteprofile-analysis/3rdparty/mixbench/build/mixbench-cuda-alt")
cost_model.set_gflops(gflops)
cost_model.set_gbps(gbps)

cost_model.set_graph_def_from_json("/root/capture_file_tf/run_0/traces_1/0/cleaned_graph.json")
print("Costmodel training on dataset.")

X_train, X_test, y_train, y_test = train_test_split(dataset_loaded.X, dataset_loaded.y, test_size=0.1, random_state=42)
with open("/root/dataset_1/test_data.pickle", "wb") as f:
    pickle.dump([X_test, y_test], f)
with open("/root/dataset_1/train_data.pickle", "wb") as f:
    pickle.dump([X_train, y_train], f)

cost_model.train(X_train, y_train)
print("Costmodel dumping.")
cost_model.dump("/root/dataset_1/cost_model.pickle")
print("Costmodel loading from pickle")
cost_model_new = FusionCostModel("/root/dataset_1/cost_model_tmp_new")
cost_model_new.load("/root/dataset_1/cost_model.pickle")
print("Predicting.")

# op_names = []
# r = re.compile("generated[0-9]*_")
# with open("./op_names_in_def.txt", "r") as f:
#     for line in f:
#         op_name = line.strip()
#         n = r.sub("", op_name)
#         op_names.append(n)

# time = cost_model_new.predict(op_names)
# print("predicted time: {}".format(time))

predicted = cost_model_new.predict_from_feature(X_test)
residual = []
for i in range(len(predicted)):
    residual.append(np.abs(predicted[i]-y_test[i]) / y_test[i])

print("Average: {}, Median: {}".format(np.average(residual), np.median(residual)))

code.interact(local=locals())

