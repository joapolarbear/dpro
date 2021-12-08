from tqdm import tqdm
import networkx as nx
import pickle
import os

from google.protobuf.json_format import MessageToJson
from google.protobuf.text_format import Parse
import tensorflow as tf
import json

try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef

# from collect import Collector
from cost_model._xla.pk_graph import PKGraph, postorder_contract_nx
from trace_utils import parse_op_name, parse_pid_from_name

TRACE_PATH = "/root/capture_file/run_0_dec8"
OUTPUT_PATH = "/root/cluster_spec_test.txt"

name2index = {}
index2name = {}
index2pid = {}
index2newname = {}

# logger = SingleLogger("/root", "trash_logger", "info")

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
        if update_nodes_in_dag is not None and _name in update_nodes_in_dag:
            _name = "UPDATE_." + _name
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

def relabel_dag_node(_dag) -> nx.DiGraph:
    def relabel_func(old_label):
        if ("BW" in old_label or "FW" in old_label or "Comm" in old_label or "UPDATE" in old_label) and "^" not in old_label:
            layer_name = parse_op_name(old_label)
            layer_pid = parse_pid_from_name(old_label)
            # if layer_pid not in self.cost_models or layer_name not in self.cost_models[layer_pid].graph_def_util.operation_names:
            #     return "DEL~"+old_label
            # TODO (huhanpeng): different pids share the same index
            # if "Comm" in old_label and layer_name in name2index and layer_pid in name2index[layer_name]:
            #     layer_index = name2index[layer_name][layer_pid]
            #     new_name = ("[%d]"%layer_index).join(old_label.split(layer_name))
            #     return new_name

            layer_index = len(index2name)
            new_name = ("[%d]"%layer_index).join(old_label.split(layer_name))
            index2name[layer_index] = layer_name
            index2pid[layer_index] = layer_pid
            if layer_name not in name2index:
                name2index[layer_name] = {}
            name2index[layer_name][layer_pid] = layer_index
            new_label = ("[%d]"%layer_index).join(old_label.split(layer_name))
            index2newname[layer_index] = new_label
            return new_label
        else:
            return old_label
    return nx.relabel_nodes(_dag, relabel_func)


# remove dependency from FW to UPDATE
# for (u, v) in list(dag.edges):
#     dag.remove_edge(u, v)
xla_candidates = set()
with open("/root/xla_candidates.txt", "r") as f:
    for line in f:
        xla_candidates.add(line.strip())

dag = wrap_read_graphdef("/root/bert/traces/before_mark_for_compilation_5.pbtxt")

dag = relabel_dag_node(dag)

pkg = PKGraph(dag, dag)

fw_nodes = []
bw_nodes = []
comm_nodes = []
update_nodes = []

for node in dag.nodes:
    if "FW" in node:
        fw_nodes.append(node)
    elif "BW" in node:
        bw_nodes.append(node)
    elif "Comm" in node:
        comm_nodes.append(node)
    elif "UPDATE" in node:
        update_nodes.append(node)

print("Len FW nodes: {}, Len BW nodes: {}, Len COMM nodes: {}, Len UPDATE nodes: {}" \
    .format(len(fw_nodes), len(bw_nodes), len(comm_nodes), len(update_nodes)))

BW_graph = dag.subgraph(bw_nodes)
BW_sequence = list(nx.topological_sort(BW_graph))

num_forbidden = int(len(BW_sequence) / 2)
forbidden_bw = BW_sequence[num_forbidden:]



filtered_nodes = []
for node in dag.nodes:
    index = int(node.split("[")[1].split("]")[0])
    orig_name = index2name[index]
    if orig_name.split(".")[1] not in xla_candidates:
        filtered_nodes.append(node)

if not os.path.exists("/root/alter_cluster_spec.pickle"):
    # Cluster all FW
    source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))

    # Run post order traversal on G
    print("Finding maximal clusters in FW...")
    visited_nodes = set()
    for source in tqdm(source_nodes, total=len(source_nodes)):
        if source not in visited_nodes and source in dag.nodes:
            _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= filtered_nodes + comm_nodes + bw_nodes)

    with open("/root/alter_cluster_spec.pickle", "wb") as f:
        pickle.dump([fw_nodes, bw_nodes, comm_nodes, update_nodes, 
                    filtered_nodes, index2name, index2pid, dag, pkg], f)
else:
    with open("/root/alter_cluster_spec.pickle", "rb") as f:
        ( fw_nodes, bw_nodes, comm_nodes, update_nodes, filtered_nodes,
        index2name, index2pid, dag, pkg )= pickle.load(f)

# new_fw_nodes = [node for node in dag.nodes if "FW" in node]

# # all BW
# print("Finding maximal clusters in all BW...")
# source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))
# visited_nodes = set()
# for source in tqdm(source_nodes, total=len(source_nodes)):
#     if source not in visited_nodes and source in dag.nodes:
#         _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= filtered_nodes + comm_nodes + new_fw_nodes)

# # all BW, size limit 1/2
# print("Finding maximal clusters in all BW...")
# source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))
# visited_nodes = set()
# for source in tqdm(source_nodes, total=len(source_nodes)):
#     if source not in visited_nodes and source in dag.nodes:
#         _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= filtered_nodes + comm_nodes + new_fw_nodes, size_limit=int(len(bw_nodes)/2))

def _get_original_name_pid_from_index(name_):
    try:
        index = int(name_.split("[")[1].split("]")[0])
    except:
        print(name_)
        input()
    return index2name[index], index2pid[index]

def _get_original_name_pid_from_fused_node(u_):
    single_pid = None
    orig_names = []
    for node_name in u_.split("+"):
        orig_name, pid = _get_original_name_pid_from_index(node_name)
        orig_names.append(orig_name)
        if single_pid is None:
            single_pid = pid
        else:
            if single_pid != pid:
                raise RuntimeError("Fused DAG node {} contains ops from different machines.".format(u_))
    return orig_names, single_pid

bw_cluster_sizes = []
bw_cluster_nodes = []
single_pid = -1
for node in dag.nodes:
    if "+" in node and "BW" in node:
        orig_names, pid = _get_original_name_pid_from_fused_node(node)
        if single_pid == -1:
            single_pid = pid
        else:
            if single_pid != pid:
                continue
        bw_cluster_sizes.append(len(node.split("+")))
        bw_cluster_nodes.append(node)

for idx, node_size in enumerate(bw_cluster_sizes):
    if node_size > 10:
        print("idx: {}, size: {}".format(idx, node_size))

clusters_to_ignore = []
while True:
    s = input("Choose a cluster to disgard: ")
    try:
        discard_id = int(s.strip())
        clusters_to_ignore.append(discard_id)
        print("Remaining clusters:")
        for idx, node_size in enumerate(bw_cluster_sizes):
            if node_size > 10 and idx not in clusters_to_ignore:
                print("idx: {}, size: {}".format(idx, node_size))
    except:
        break

nodes_to_ignore = set()
for idx in clusters_to_ignore:
    nodes_to_ignore.add(bw_cluster_nodes[idx])

# dump cluster mapping
cluster_index = 0
with open("/root/partitions_spec.txt", "w") as f:
    for node in dag.nodes():
        if "+" in node:
            orig_names, pid = _get_original_name_pid_from_fused_node(node)
            if pid != single_pid:
                continue
            if node not in nodes_to_ignore:
                for orig_node_name in orig_names:
                    f.write("{} {}\n".format(orig_node_name, cluster_index))
                cluster_index += 1
            else:
                for orig_node_name in orig_names:
                    f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1