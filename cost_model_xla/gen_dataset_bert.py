import argparse
import os
import shutil
import subprocess
import tensorflow as tf
import official.nlp.bert as bert
from official.nlp.bert import bert_models
from google.protobuf import text_format
import json
import pickle
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from gen_samples import SampleGenerator
from execute_graph import execute_graph_def
from process_trace import *
from gen_dataset_utils import *
from tqdm import trange

def parse_arguments():
    parser = argparse.ArgumentParser(description='XLA Op fusion dataset generator.')
    parser.add_argument('--result_dir', required=True, type=str, 
                        help='Path to save the resulting dataset.')
    parser.add_argument('--num_samples', default=200, type=int, 
                        help='Number of samples to generate.')
    parser.add_argument('--debug', action='store_true',
                        help='If set, output additional debug info.')
    parser.add_argument('--min_subgraph_level', default=1, type=int, 
                        help='Lower bound of generated subgraph\'s level.')
    parser.add_argument('--max_subgraph_level', default=8, type=int, 
                        help='Upper bound of generated subgraph\'s level.')
    parser.add_argument('--gpu_benchmark_cmd', required=True, type=str, 
                        help='Executable to generate GPU benchmark statistics.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
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

    print("Reading profiles for entire graph.")
    op_time_fn = os.path.join(args.result_dir, "op_running_times.pickle")
    with open(op_time_fn, "rb") as f:
        op_time_dict = pickle.load(f)

    gen_dataset(graph_def, op_time_dict, args.gpu_benchmark_cmd, args.result_dir, 
                num_samples=args.num_samples, min_subgraph_level=args.min_subgraph_level,
                max_subgraph_level=args.max_subgraph_level)
    
