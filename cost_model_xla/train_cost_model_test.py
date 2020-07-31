from cost_model import *
from gen_dataset_utils import run_gpu_profile
import tensorflow as tf
import official.nlp.bert as bert
from official.nlp.bert import bert_models
import json
import pickle
import re
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from sklearn.model_selection import train_test_split
import code

# print("Loading dataset.")
# dataset = XlaDataset("/root/bert_dataset")
# print("Loaded dataset.")
# dataset.dump_dataset("/root/cost_model_test/dataset.pickle")
# print("Dataset dumped.")

dataset_loaded = XlaDataset("/root/cost_model_test/dataset.pickle")
print("Dataset loaded from dump.")
cost_model = FusionCostModel("/root/cost_model_test/cost_model_tmp")
print("Costmodel setting feature mask.")
cost_model.set_feature_mask(dataset_loaded.feature_mask)
print("Costmodel setting op time dict.")
cost_model.set_op_time_dict(dataset_loaded.op_time_dict)
print("Costmodel running gpu profile.")
gflops, gbps = run_gpu_profile("/root/mixbench/build/mixbench-cuda-alt")
cost_model.set_gflops(gflops)
cost_model.set_gbps(gbps)


# gen freezed bert graph
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)
bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
bert_config = bert.configs.BertConfig.from_dict(config_dict)
max_seq_length = 128
max_predictions_per_seq = 20
use_next_sentence_label = True

pretrain_model, core_model = bert_models.pretrain_model(bert_config, max_seq_length, max_predictions_per_seq,
                                                    use_next_sentence_label=use_next_sentence_label)

func = tf.function(core_model).get_concrete_function([tf.TensorSpec(shape=[32,128], dtype=tf.int32), tf.TensorSpec(shape=[32,128], dtype=tf.int32), tf.TensorSpec(shape=[32,128], dtype=tf.int32)])
frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func)

cost_model.set_graph_def(graph_def)
print("Costmodel training on dataset.")

X_train, X_test, y_train, y_test = train_test_split(dataset_loaded.X, dataset_loaded.y, test_size=0.33, random_state=42)
with open("/root/cost_model_test/test_data.pickle", "wb") as f:
    pickle.dump([X_test, y_test], f)
with open("/root/cost_model_test/train_data.pickle", "wb") as f:
    pickle.dump([X_train, y_train], f)

cost_model.train(X_train, y_train)
print("Costmodel dumping.")
cost_model.dump("/root/cost_model_test/cost_model.pickle")
print("Costmodel loading from pickle")
cost_model_new = FusionCostModel("/root/cost_model_test/cost_model_tmp_new")
cost_model_new.load("/root/cost_model_test/cost_model.pickle")
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

