import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.client import timeline
import random
import code
import pickle
import os

def get_shape_from_placeholder(placeholder_op):
    dim_protos = placeholder_op.get_attr("shape").dim
    return tf.TensorShape([d.size for d in dim_protos])

def get_dtype_from_placeholder(placeholder_op):
    return placeholder_op.get_attr("dtype")

def get_output_tensors_from_graph(graph):
    output_tensors = []
    for op in graph.get_operations():
        output_tensors.append(op.outputs[0])
    return output_tensors

def execute_graph_def(graph_def, input_node_defs, profile_result_dir, tf2xla_config_path=None, num_runs=20):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    input_nodes = []
    for node_def in input_node_defs:
        node = graph.get_operation_by_name(node_def.name)
        input_nodes.append(node)

    feed_dict = {}
    for input_node in input_nodes:
        shape = get_shape_from_placeholder(input_node)
        dtype = get_dtype_from_placeholder(input_node)
        if dtype == tf.int8 or dtype == tf.int32 or dtype == tf.int64:
            feed_dict[input_node.outputs[0]] = tf.random.uniform(shape=shape, maxval=5, dtype=dtype).numpy()
        else:
            feed_dict[input_node.outputs[0]] = tf.random.uniform(shape=shape, dtype=dtype).numpy()
        print("feed_dict added : {} with shape {} and dtype {}".format(input_node.outputs[0], shape, dtype))
    with tf.profiler.experimental.Profile(profile_result_dir):
        with tf.compat.v1.Session(graph=graph) as sess:
            for i in range(num_runs):
                sess.run(get_output_tensors_from_graph(graph), feed_dict)
