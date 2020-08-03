import tensorflow as tf
from google.protobuf.json_format import Parse
import byteps.tensorflow as bps
import numpy as np
import json

with open("./graph.json", "r") as f:
    graph_def = Parse(f.read(), tf.GraphDef())

save_nodes = []
for node in graph_def.node:
    if "save" in node.name:
        save_nodes.append(node.name)

cleaned_graph_def = tf.GraphDef()
cleaned_graph_def.versions.CopyFrom(graph_def.versions)
cleaned_graph_def.library.CopyFrom(graph_def.library)
for node in graph_def.node:
    if "save" in node.name:
        continue
    cleaned_graph_def.node.append(node)

sess = tf.Session()
tf.graph_util.import_graph_def(cleaned_graph_def)

def gen_np_tensor(shape, dtype):
    res = np.random.rand(*shape)
    if isinstance(res, float):
        if "int" in dtype:
            return int(res)
        else:
            return res
    else:
        return res.astype(dtype)


# Restore the variable values
def restore_var_values(sess, var_shapes):
    # Find the variable initialization operations
    assign_ops = []
    feed_dict = {}
    for v, meta in var_shapes.items():
        try:
            assign_op = sess.graph.get_operation_by_name(v + '/Assign')
        except:
            assign_op = sess.graph.get_operation_by_name("import/" + v + '/Assign')
        assign_ops.append(assign_op)
        shape = meta["shape"]
        dtype = meta["dtype"]
        feed_dict[assign_op.inputs[1]] = gen_np_tensor(shape, dtype)
    # Run the initialization operations with the given variable values
    sess.run(assign_ops, feed_dict=feed_dict)

with open("./variables_meta.json", "r") as f:
    var_shapes = json.load(f)

with open("./run_meta.json", "r") as f:
    run_meta = json.load(f)

feed_meta = run_meta["feed_dict"]
fetches = run_meta["fetches"]

print("Restoring var values")
restore_var_values(sess, var_shapes)
print("Populating feed dict...")
input_ops = []
def get_output_tensors_from_graph(graph):
    output_tensors = []
    for op in graph.get_operations():
        if len(op.outputs):
            output_tensors.append(op.outputs[0])
    return output_tensors

run_fetches = []
for tn in fetches:
    try:
        try:
            fetch = sess.graph.get_tensor_by_name(tn)
        except:
            fetch = sess.graph.get_tensor_by_name("import/" + tn)
    except:
        try:
            fetch = sess.graph.get_operation_by_name(tn)
        except:
            fetch = sess.graph.get_operation_by_name("import/" + tn)
    run_fetches.append(fetch)

feed_dict = {}
for op_name, meta in feed_meta.items():
    try:
        op = sess.graph.get_operation_by_name(op_name)
    except:
        op = sess.graph.get_operation_by_name("import/" + op_name)
    input_ops.append(op)
    feed_dict[op.outputs[0]] = gen_np_tensor(meta["shape"], meta["dtype"])

sess.run(run_fetches, feed_dict=feed_dict)

print("Execution finished.")