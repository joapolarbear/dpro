from cost_model_xla.process_trace import TRACE_SUFFIX
import tqdm
import networkx as nx

from collect import Collector
from cost_model_xla.pk_graph import PKGraph, postorder_contract_nx
from trace_utils import *

TRACE_PATH = "/root/capture_file/run_0_dec8"
OUTPUT_PATH = "/root/cluster_spec_test.txt"

name2index = {}
index2name = {}
index2pid = {}
index2newname = {}

logger = SingleLogger("/root", "trash_logger", "info")

def relabel_dag_node(_dag) -> nx.DiGraph:
    def relabel_func(old_label):
        if ("BW" in old_label or "FW" in old_label or "Comm" in old_label) and "^" not in old_label:
            layer_name = parse_layer_name(old_label)
            layer_pid = parse_pid_from_name(old_label)
            # if layer_pid not in self.cost_models or layer_name not in self.cost_models[layer_pid].graph_def_util.operation_names:
            #     return "DEL~"+old_label
            # TODO (huhanpeng): different pids share the same index
            if "Comm" in old_label and layer_name in name2index and layer_pid in name2index[layer_name]:
                layer_index = name2index[layer_name][layer_pid]
                new_name = ("[%d]"%layer_index).join(old_label.split(layer_name))
                return new_name

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

clct = Collector(TRACE_PATH, comm_backend="BYTEPS")
clct.init(False)

dag = relabel_dag_node(clct.trail_dag)

dag = clct.trail_dag
pkg = PKGraph(dag, dag)

fw_nodes = []
bw_nodes = []

for node in dag.nodes:
    if "FW" in node:
        fw_nodes.append(node)
    elif "BW" in node:
        bw_nodes.append(node)

print("Len FW nodes: {}, Len BW nodes: {}".format(len(fw_nodes), len(bw_nodes)))

# Cluster all FW
source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))

# Run post order traversal on G
print("Finding maximal clusters in FW...")
visited_nodes = set()
for source in tqdm(source_nodes, total=len(source_nodes)):
    if source not in visited_nodes and source in dag.nodes:
        _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= bw_nodes)

new_fw_nodes = [node for node in dag.nodes if "FW" in node]

# all BW
print("Finding maximal clusters in all BW...")
source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))
visited_nodes = set()
for source in tqdm(source_nodes, total=len(source_nodes)):
    if source not in visited_nodes and source in dag.nodes:
        _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= new_fw_nodes)

# all BW, size limit 1/2
print("Finding maximal clusters in all BW...")
source_nodes = sorted(list(dag.nodes), key=lambda x: dag.in_degree(x))
visited_nodes = set()
for source in tqdm(source_nodes, total=len(source_nodes)):
    if source not in visited_nodes and source in dag.nodes:
        _, _, dag = postorder_contract_nx(dag, pkg, source, visited_nodes, forbidden_list= new_fw_nodes, size_limit=int(len(bw_nodes)/2))

def _get_original_name_pid_from_index(name_):
    index = int(name_.split("[")[1].split("]")[0])
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
for node in dag.nodes:
    if "+" in node and "BW" in node:
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
single_pid = -1
with open(args_.output_path, "w") as f:
    for node in dag.nodes():
        if "+" in node:
            orig_names, pid = _get_original_name_pid_from_fused_node(node)
            if single_pid == -1:
                single_pid = pid
            else:
                if single_pid != pid:
                    continue
            if node not in nodes_to_ignore:
                for orig_node_name in orig_names:
                    f.write("{} {}\n".format(orig_node_name, cluster_index))
                cluster_index += 1
            else:
                for orig_node_name in orig_names:
                    f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1