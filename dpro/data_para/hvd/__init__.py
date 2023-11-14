import json
from ...trace_utils import FileName

from .graph import ncclGraph

def iter_collect_nccl_graph(nccl_graph, pm, arg_list, nccl_algo=None):
    nccl_graph.map_host_prefix_id(pm.dirs)
    for tmp_pm, pid, host_id_str in arg_list:
        collect_nccl_graph(tmp_pm=tmp_pm, pid=pid, host_id=host_id_str, pm=pm, nccl_algo=nccl_algo, nccl_graph=nccl_graph)
                    
def collect_nccl_graph(tmp_pm=None, pid=None, host_id=None, pm=None, nccl_algo=None, nccl_graph=None):
    nccl_rank_graph_path = pm.search(FileName.NCCL_RANK_GRAPH) if tmp_pm is None else tmp_pm.search(FileName.NCCL_RANK_GRAPH)
    with open(nccl_rank_graph_path, 'r') as f:
        nccl_rank_graph = json.load(f)
    if nccl_algo is None:
        raise ValueError("--nccl_algo must be given")
    elif nccl_algo.lower() == "tree":
        nccl_graph.parse_tree_topo(nccl_rank_graph["Tree"], map_to=pid)
    elif nccl_algo.lower() == "ring":
        nccl_graph.parse_ring_topo(nccl_rank_graph["RealRing"], map_to=pid)
        # nccl_graph.parse_connect_topo(traces["Ring"], map_to=pid)
        
__all__ = [ncclGraph, iter_collect_nccl_graph]