import argparse
import os
import shutil
import subprocess
import tensorflow as tf
import official.nlp.bert as bert
from official.nlp.bert import bert_models
import json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from gen_samples import SampleGenerator
from execute_graph import execute_graph_def
from process_trace import *
from gen_dataset_utils import *
from tqdm import trange
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate graph profile for BERT model.')
    parser.add_argument('--result_dir', required=True, type=str, 
                        help='Path to save the resulting dataset.')
    args = parser.parse_args()
    # check
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
        print("Result dir not exist. Created result dir at {}.".format(args.result_dir))
    return args

if __name__ == "__main__":
    args = parse_arguments()
    profile_dir = os.path.join(args.result_dir, "whole_graph_profile")
    if not os.path.isdir(profile_dir):
        os.makedirs(profile_dir)
    # gen freezed bter graph
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

    sample_generator = SampleGenerator(freezed_graph_def=graph_def)
    print("Profiling entire graph.")
    op_time_dict = profile_entire_graph(sample_generator, profile_dir)
    output_fn = os.path.join(args.result_dir, "op_running_times.pickle")
    with open(output_fn, "wb") as f:
        pickle.dump(op_time_dict, f)
    # if os.path.isdir(profile_dir):
        # os.rmdir(profile_dir)
    print("Dataset generation complete.")
    
