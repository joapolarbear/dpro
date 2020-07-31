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
from gen_dataset_utils import log_op_names
from google.protobuf import text_format

if __name__ == "__main__":
    # gen freezed bert graph
    with tf.compat.v1.gfile.FastGFile("/root/bert_subgraph/0.pbtxt", "r") as f:
        print("Reading graph...")
        graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())
    log_op_names(graph_def)