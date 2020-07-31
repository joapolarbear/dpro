import argparse
import os
import shutil
import subprocess
import tensorflow as tf
import official.nlp.bert as bert
from official.nlp.bert import bert_models
import json
import pickle
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from gen_samples import SampleGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description='XLA Op fusion dataset generator.')
    parser.add_argument('--result_dir', required=True, type=str, 
                        help='Path to save the resulting dataset.')
    parser.add_argument('--min_subgraph_level', default=1, type=int, 
                        help='Lower bound of generated subgraph\'s level.')
    parser.add_argument('--max_subgraph_level', default=8, type=int, 
                        help='Upper bound of generated subgraph\'s level.')
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

    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
        print("Result dir not exist. Created result dir at {}.".format(args.result_dir))
    sample_generator = SampleGenerator(freezed_graph_def=graph_def)
    subgraph_def, input_defs, output_def, sample_id = sample_generator.gen_random_subgraph(min_levels=args.min_subgraph_level, max_levels=args.max_subgraph_level, debug_dir=args.result_dir)