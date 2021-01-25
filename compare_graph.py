import networkx as nx
from google.protobuf.json_format import MessageToJson
from google.protobuf.text_format import Parse
import tensorflow as tf
import json

try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef

def tf_relabel_func(_name, update_nodes_in_dag):
    for prefix in ["Comm.", "Comp.", "BW.", "FW.", "UPDATE_."]:
        if _name.startswith(prefix):
            return _name
    if _name.startswith("^"):
        _name = _name[1:]
    last_slash_pos = _name.rfind("/")
    if last_slash_pos != -1 and last_slash_pos < len(_name)-1 and _name[last_slash_pos+1] == "_":
        _name = _name[:last_slash_pos]
    if "BytePSPushPull" in _name and "tensor" not in _name:
        _name = "Comm." + _name
    elif "allreduce" in _name.lower():
        if "." in _name:
            _, tensor_name = _name.split(".")
            if "_" in tensor_name:
                tensor_name = tensor_name.split("_")[0]
            _name = "Comm." + tensor_name
        else:
            _name = "UPDATE_." + _name
    else:
        if update_nodes_in_dag is not None and _name in update_nodes_in_dag \
            or _name == "GradientDescent":
            _name = "UPDATE_." + _name
        elif _name == "GradientDescent":
            _name = ""
        elif _name.startswith("gradients"):
            _name = "BW." + _name
        else:
            _name = "FW." + _name
    return _name

def wrap_read_graphdef(graphdef_path):
    if graphdef_path.endswith("pbtxt"):
        with open(graphdef_path, "r") as f:
            pb = f.read()
        graph_def = Parse(pb, GraphDef())
        json_string = MessageToJson(graph_def)
        graph_def = json.loads(json_string)
    else:
        with open(graphdef_path, "r") as f:
            graph_def = json.load(f)
    graph = nx.DiGraph()
    for node in graph_def["node"]:
        if "input" in node:
            for input_tensor_name in node["input"]:
                input_node_name = input_tensor_name.split(":")[0]
                graph.add_edge(input_node_name, node["name"])
    update_nodes_in_dag = set()
    def recursive_add_succs(_node):
        for succ_ in graph.successors(_node):
            update_nodes_in_dag.add(succ_)
            recursive_add_succs(succ_)
    for node in graph.nodes:
        if "allreduce" in node.lower() or "bytepspushpull" in node.lower():
            recursive_add_succs(node)
    new_graph = nx.DiGraph()
    for u, v in graph.edges:
        new_graph.add_edge(tf_relabel_func(u, update_nodes_in_dag), tf_relabel_func(v, update_nodes_in_dag))
    return new_graph, update_nodes_in_dag

dag = nx.read_gml("/root/capture_file/run_0_dec8/simple_dag.gml")

graphdef, update_nodes = wrap_read_graphdef("/root/bert/traces/before_mark_for_compilation_5.pbtxt")

dag_nodes = set(dag.nodes)
graphdef_nodes = set(graphdef.nodes)

import code
code.interact(local=locals())



