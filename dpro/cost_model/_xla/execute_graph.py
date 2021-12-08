import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import os
import json
from tensorflow.python.client import timeline

def get_shape_from_placeholder(placeholder_op):
    dim_protos = placeholder_op.get_attr("shape").dim
    return [d.size for d in dim_protos]

def get_dtype_from_placeholder(placeholder_op):
    return str(placeholder_op.get_attr("dtype")).split("\'")[1]

def get_output_tensors_from_graph(graph):
    output_tensors = []
    for op in graph.get_operations():
        output_tensors.append(op.outputs[0])
    return output_tensors

def execute_graph_def(graph_def, input_node_defs, fetches, profile_result_dir, tf2xla_config_path=None, num_runs=20, trace_start=10, trace_end=20):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    input_nodes = []
    for node_def in input_node_defs:
        node = graph.get_operation_by_name(node_def.name)
        input_nodes.append(node)
    output_tensors = []
    for node_def in fetches:
        node = graph.get_operation_by_name(node_def.name)
        output_tensors.append(node.outputs[0])

    feed_dict = {}
    for input_node in input_nodes:
        shape = get_shape_from_placeholder(input_node)
        dtype = get_dtype_from_placeholder(input_node)
        print(dtype)
        feed_dict[input_node.outputs[0]] = np.random.rand(*shape).astype(dtype)
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    traces = {"traceEvents":[]}
    fetch = output_tensors
    with tf.compat.v1.Session(graph=graph) as sess:
        for i in range(num_runs):
            sess.run(fetch, feed_dict, options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = json.loads(tl.generate_chrome_trace_format())
            if trace_start < i < trace_end:
                traces["traceEvents"] += ctf["traceEvents"]
                print("{} th step trace added.".format(i))
    with open(os.path.join(profile_result_dir, "temp.json"), "w") as f:
        json.dump(traces, f, indent=4)
    print("Ran to the end.")