from google.protobuf.json_format import MessageToJson
from google.protobuf.text_format import Parse
import tensorflow as tf
import sys, os
import json
import networkx as nx

def wrap_read_graphdef(graphdef_path):
    try:
        GraphDef = tf.GraphDef
    except:
        GraphDef = tf.compat.v1.GraphDef
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
    gml_path = os.path.join(os.path.dirname(graphdef_path), "graphdef_dag.gml")
    nx.write_gml(graph, gml_path)

if __name__ == "__main__":
    wrap_read_graphdef(sys.argv[1])