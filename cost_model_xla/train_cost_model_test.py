from cost_model import *
from gen_dataset_utils import run_gpu_profile
import json
import pickle
import re
import os
from sklearn.model_selection import train_test_split
import code

DATASET_ROOT = "/root/dataset_warmup"

print("Loading dataset.")
dataset = XlaDataset(DATASET_ROOT)
print("Loaded dataset.")
dataset_pickle_path = os.path.join(DATASET_ROOT, "dataset.pickle")
dataset.dump_dataset(dataset_pickle_path)
print("Dataset dumped.")

dataset_loaded = XlaDataset(dataset_pickle_path)
print("Dataset loaded from dump.")
cost_model = FusionCostModel(os.path.join(DATASET_ROOT, "cost_model_tmp"))
cost_model.init_from_dataset(dataset_loaded, "/root/capture_file_tf/run_0/traces_1/0/cleaned_graph.json")

print("Costmodel training on dataset.")

X_train, X_test, y_train, y_test = train_test_split(dataset_loaded.X, dataset_loaded.y, test_size=0.1, random_state=42)
with open(os.path.join(DATASET_ROOT, "test_data.pickle"), "wb") as f:
    pickle.dump([X_test, y_test], f)
with open(os.path.join(DATASET_ROOT, "train_data.pickle"), "wb") as f:
    pickle.dump([X_train, y_train], f)

cost_model.train(X_train, y_train)
print("Costmodel dumping.")
cost_model.dump(os.path.join(DATASET_ROOT, "cost_model.pickle"))
print("Costmodel loading from pickle")
cost_model_new = FusionCostModel(os.path.join(DATASET_ROOT, "/root/dataset_1/cost_model_tmp_new"))
cost_model_new.load(os.path.join(DATASET_ROOT, "cost_model.pickle"))
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

