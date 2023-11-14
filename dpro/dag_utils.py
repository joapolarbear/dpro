from argparse import ArgumentError
import os
import re
import networkx as nx
import matplotlib.pyplot as plt

from .trace_utils import *

try:
    from data_para.hvd.graph import *
except:
    pass
try:
    from data_para.bps_helper.graph import *
except:
    pass

VIRTUAL_SYNC_OP = True

def visualize_gml(graph, layout="circular"):
    if layout == "spectral":
        pos = nx.spectral_layout(graph, dim=2, scale=0.5)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "random":
        pos = nx.random_layout(graph)
    else:
        raise ArgumentError("Layout must be one of [\"spectral\", \"circular\", \"random\"]")
    nx.draw(graph, pos, with_labels=True, font_size=6)
    plt.show()
    # import matplotlib.pyplot as plt; plt.ion()
    # import netgraph
    # netgraph.draw(graph)
    # plot_instance = netgraph.InteractiveGraph(graph, node_positions=pos)
    # node_positions = plot_instance.node_positions

def part_of_dag(dag, node, 
    max_in_depth=2, max_out_depth=2, 
    path=None, simple=False, 
    focus_nodes=None, use_std_name=True,
    name_size_limit=50):
    ''' Return part of the dag, recursively add the predecessors
        and the successors of `node`
    '''
    small_dag = nx.DiGraph()
    edges_to_add = []

    visited = set()
    start_nodes = set()
    end_nodes = set()

    def recur_add(_node, _max_in_depth, _max_out_depth, _visited, simple=False):
        if _max_in_depth > 0:
            for pred in dag.predecessors(_node):
                if dag.in_degree(pred) == 0:
                    start_nodes.add(pred)
                if dag.out_degree(pred) == 0:
                    end_nodes.add(pred)
                    
                if pred in _visited:
                    continue
                edges_to_add.append((pred, _node))
                _visited.add(_node)
                recur_add(pred, 0 if simple else _max_in_depth-1, _max_out_depth, _visited)
                _visited.remove(_node)
                
        if _max_out_depth > 0:
            for succ in dag.successors(_node):
                if dag.in_degree(succ) == 0:
                    start_nodes.add(succ)
                if dag.out_degree(_node) == 0:
                    end_nodes.add(_node)

                if succ in _visited:
                    continue
                edges_to_add.append((_node, succ))
                _visited.add(_node)
                recur_add(succ, _max_in_depth, 0 if simple else _max_out_depth-1, _visited)
                _visited.remove(_node)
                
    recur_add(node, max_in_depth, max_out_depth, visited, simple=simple)
    small_dag.add_edges_from(edges_to_add)

    for _node in start_nodes:
        small_dag.nodes[_node]["color"] = "red"
    for _node in end_nodes:
        small_dag.nodes[_node]["color"] = "green"
    small_dag.nodes[node]["color"] = "yellow"

    if focus_nodes is not None:
        for _node in focus_nodes:
            if use_std_name:
                _node = parse_allinfo_from_name(_node)[1]
            if _node in small_dag.nodes:
                small_dag.nodes[_node]["fontcolor"] = "blue"
    
    def relabel_func(old_label):
        old_label = "+".join([parse_rawname(_label) for _label in old_label.split("+")])
        if len(old_label) > name_size_limit:
            return old_label[:name_size_limit] + "..."
        else:
            return old_label
    nx.relabel_nodes(small_dag, relabel_func, copy=False)

    if path is not None:
        nx.drawing.nx_pydot.write_dot(small_dag, path)
    return small_dag

def cal_edge_cost(G):
    for u, v in G.edges:
        if "weight" in G.edges[u, v]:
            # print(u, v, G.edges[u, v]["weight"])
            G.edges[u, v]["cost"] = G.edges[u, v]["weight"]
        else:
            gap = 0
            prev_cat = parse_cat_from_name(u)
            next_cat = parse_cat_from_name(v)
            for key, value in G.nodes[u].items():
                if "GAP" in key:
                    if key == GAP_STR_INTERNODE:
                        prev_cat_finegrained = parse_cat_fine_grained(u)
                        next_cat_finegrained = parse_cat_fine_grained(v)
                        if prev_cat_finegrained == "Comm.PUSH_RES" and next_cat_finegrained == "Comm.PULL_REQ":
                            gap += value
                    # elif key == GAP_STR_OP2COMM and "Sync" in u:
                    #     ### TODO (huhanpeng): should be GAP_STR_COMM2COMM
                    #     gap += value
                    else:
                        ### e.g. "gap.operator.operator"
                        key_s = key.split("GAP")
                        if prev_cat == key_s[0] and next_cat == key_s[1]:
                            gap += value
            G.edges[u, v]["cost"] = G.nodes[u]["avg"] + gap / 1000.0
        
def dag_longest_path(G, pathM=None, weight='weight', default_weight=0, _debug_level=0):
    critical_path = nx.algorithms.dag.dag_longest_path(G, weight=weight, default_weight=default_weight)
    prefix = "Critical Path of " + (pathM.ret_id_in_trial() if pathM is not None else "none")
    if _debug_level > 1:  
        SingleLogger().info(prefix + " => ")
    path_length = 0
    len_list = []
    for (u, v) in nx.utils.pairwise(critical_path):
        weight_ = G[u][v].get(weight, default_weight)
        path_length += weight_
        if _debug_level > 1:
            SingleLogger().info("%-80s: %.3f/%.3f ms" % (u, weight_, path_length))
        len_list.append(weight_)
    len_list.append(0)
    # SingleLogger().info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
    if _debug_level > 0:
        SingleLogger().info("Length of the " + prefix + "%12.4f ms\n" % path_length)

    return list(zip(critical_path, len_list))

def wrap_read_gml(gml_path, metadata, pretty=True):
    ''' Read raw DFG file
        * The node name in Tensorflow is not standard, transfer it to standard form first
            Tranverse the dag nodes twice
        * For TF 2.4, the raw DFG does not contain Comm OPs, mannually add Comm OPs for 
            a distributed case
    '''
    
    mygraph = metadata.wrap_read_dfg(gml_path)

    if not pretty:
        try:
            SingleLogger().info(list(nx.find_cycle(mygraph, orientation="original")))
        except:
            SingleLogger().info("No cycles found")

    return mygraph

class DAGManager:
    ''' Maintain a dependency graph for one GPU
    Parameters
    ----------
    path: str
        Root path for one GPU
    
    Examples:

    Legend: ──────▶   intra-worker dependency            ══════▶  inter-worker dependency

    e.g. For NCCL ALLREDUCE RING
    Note: Sync used to sync between ranks                          
                                                                                        
    ┌─────────┐                                                                                       
    │Worker 0 │  FW  ─────▶OUTPUT────▶  BW ──────▶ ...  ────▶ BW                           UPDATE_<id>
    └─────────┘                                                                                       
                                        │                                                       ▲     
                                        └─▶ Comm.Sync ──▶ Comm.<>.SEND~>xxx ──────▶  ...   ─────┘     
                                                                                                    
                                                ║                 ▲                                   
                                                ╠═════════════════╣                                   
                                                ║                 ▼                                   
                                                                                                    
                                        ┌─▶ Comm.Sync ──▶ Comm.<>.SEND~>xxx ──────▶  ...   ─────┐     
                                        │                                                       ▼     
    ┌─────────┐                                                                                       
    │Worker 1 │  FW  ─────▶OUTPUT────▶  BW ──────▶ ...  ────▶ BW                           UPDATE_<id>
    └─────────┘                                                                                       

    e.g.: For PS
    ┌─────────┐                                                                                                                               
    │Worker 0 │    FW  ──▶OUTPUT──▶  BW ───────────▶ ...  ────▶ BW                                                                UPDATE_<id> 
    └─────────┘                                                                                                                               
                                     │                                                                                                 ▲      
                                     └──▶ PUSH_REQ  ────────▶ PUSH_RES  ────▶ PULL_REQ  ────────────────────────────────┬─▶ PULL_RES  ─┘      
                                                                                                                        │                     
                                              │                                                                         │                     
                                           ┌──┘                                                                         │                     
                                           ▼                                                                            │                     
    ┌─────────┐                     ┌─────────────┐    ┌───────┐    ┌────────┐   ┌─────────────────┐   ┌──────────────┐ │                     
    │ Server  │                     │ COPY_FIRST  │───▶│ SUM_0 │───▶│  ...   │──▶│SUM_<worker_num-2│──▶│ COPY_MERGED  │─┤                     
    └─────────┘                     └─────────────┘    └───────┘    └────────┘   └─────────────────┘   └──────────────┘ │                     
                                                           ▲                                                            │                     
                                             ┌─────────────┘                                                            │                     
                                             │                                                                          │                     
                                                                                                                        │                     
                                     ┌─▶ PUSH_REQ  ────────▶ PUSH_RES  ────▶ PULL_REQ  ─────────────────────────────────┴─▶ PULL_RES  ──┐     
                                     │                                                                                                  ▼     
    ┌─────────┐                                                                                                                               
    │Worker 1 │    FW  ──▶OUTPUT──▶  BW ───────────▶ ...  ────▶ BW                                                                 UPDATE_<id>
    └─────────┘                                                                                                                               
                                                                                       
    '''
    def __init__(self, path, traceM, 
            nccl_graph=None,
            byteps_graph=None,
            platform="TENSORFLOW",
            single=False,
            update_barrier=False
        ):
        self.pm = PathManager(path)
        self.platform = platform
        ### traceM's DirLevel = TRAIL
        self.traceM = traceM
        self.dag = []
        self.nodes = set()
        self._fw_bw_dag = None

        # TODO: delete
        self._topo_sort = []
        self.topo_sorts = []

        ### is the dag for single rank
        self.single = single

        ### For fine-grained communication dependency
        # one and only one of NCCL_GRAPH or BYTEPS_GRAPH can be set at a time
        assert self.single or ((nccl_graph or byteps_graph) and not (nccl_graph and byteps_graph)), (self.single, nccl_graph, byteps_graph)
        self.nccl_graph = nccl_graph
        self.byteps_graph = byteps_graph
        if nccl_graph is not None:
            self.comm_backend = "NCCL"
        elif byteps_graph is not None:
            self.comm_backend = "BYTEPS"
        else:
            self.comm_backend = "NONE"
        self.wk_prefix, self.local_rank = self.pm.ret_prefix()
        self.prefix = gen_pid_name(self.comm_backend, self.wk_prefix, self.local_rank)

        self.update_barrier = update_barrier

    def wrap_add_dag(self, u, v):
        self.dag.append((u, v))
        self.nodes.add(u)
        self.nodes.add(v)

    def wrap_in_dag(self, node):
        return node in self.nodes
    
    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.prefix, name)
        else:
            return gen_long_name(_prefix, name)

    def add_update_downstream(self, graph, update_node, _prefix=None):
        ''' Add UPDATE operators and its downstream operators to the final graph
        '''
        u = self.add_prefix(update_node, _prefix=_prefix)
        for succ_ in graph.successors(update_node):
            v = self.add_prefix(succ_, _prefix=_prefix)
            self.wrap_add_dag(u, v)
            self.add_update_downstream(graph, succ_, _prefix)

    def _process_edge_nccl(self, graph, queue_type_list, u, v, para_dict=None, pre_nodes=[], post_nodes=[]):
        ''' Handel one edge in the original depedency graph
        Parameters
        ----------
        graph: class nx.Graph, the original depedency graph
        queue_type_list: str list
        '''
        if "Comm" in u:
            if self.single:
                ### add virtual Comm edges for single rank casts
                self.wrap_add_dag(self.add_prefix(u), self.add_prefix(v))
                return
            elif self.nccl_graph is not None and self.nccl_graph.algo == NCCL_ALGO.RING:
                ### Combine chunkId, sliceId and channelId into the graph for RING algorithm
                chunkNum, sliceNum, channelNum, loopNum = self.nccl_graph.get_IDnum(u)
                for loopId in range(loopNum):
                    for chunkId in range(chunkNum):
                        for sliceId in range(sliceNum):
                            for channelId in range(channelNum):
                                if self.nccl_graph.is_first_step(chunkId):
                                    ### The first step
                                    ### Connect BW nodes to Sync, if this is a fused tensor, there should be multiple BW nodes
                                    # next_rawname = gen_long_name(None, "%s.%s"%(u, queue_type_list[0]), suffix=None)
                                    # for pre_node in pre_nodes:
                                    #     self.wrap_add_dag(self.add_prefix(pre_node), self.add_prefix(next_rawname))

                                    ### pre_nodes: ['Comm.xxx+xxx+...+xxx.Sync']
                                    next_rawname = pre_nodes[0]

                                    ### Connect all ranks' Sync to the first Send
                                    prev_rawname = next_rawname
                                    prev_nodes_prefix = self.nccl_graph.bw_to_first_send(channelId)
                                    next_rawname = "%s.%s"%(u, queue_type_list[1])
                                    for _prefix in prev_nodes_prefix:
                                        prev_name = self.add_prefix(prev_rawname, _prefix=_prefix)
                                        self.wrap_add_dag(prev_name, self.add_prefix(next_rawname))

                                    ### Queue --> MEMCPY_IN_FUSION_BUFFER
                                    prev_rawname = next_rawname 
                                    comm_in_name = "%s.%s"%(u, queue_type_list[2])
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(comm_in_name))

                                    ### MEMCPY_IN_FUSION_BUFFER to the first Send
                                    next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                    ### TODO (huhanpeng) MEMCPY_IN_FUSION_BUFFER to the first RECV
                                    next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                else:
                                    ### normal steps
                                    ### Connect Memory copy in to Send and Recv
                                    comm_in_name = gen_long_name(None, "%s.%s"%(u, queue_type_list[2]), suffix=None)                                    
                                    _, last_chunkId = self.nccl_graph.send_to_last_recv(self.prefix, chunkId)
                                    prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, last_chunkId, sliceId))
                                    next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(prev_rawname))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                    ### Connect from Recv to Send
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))

                                ### Connect from Send to Recv
                                next_rank_prefix, next_chunkId = self.nccl_graph.send_to_recv(self.prefix, chunkId, channelId)
                                prev_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, next_chunkId, sliceId))
                                self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname, _prefix=next_rank_prefix))

                                if self.nccl_graph.is_last_step(chunkId):
                                    ### last RECV --> MEMCPY_OUT_FUSION_BUFFER
                                    prev_name = self.add_prefix(next_rawname, _prefix=next_rank_prefix)
                                    next_name = gen_long_name(next_rank_prefix, "%s.%s"%(u, queue_type_list[-1]), suffix=None)
                                    self.wrap_add_dag(prev_name, next_name)

                                    ### MEMCPY_OUT_FUSION_BUFFER --> UPDATE_CAL
                                    prev_name = next_name
                                    if self.platform == "MXNET":
                                        update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                        self.wrap_add_dag(prev_name, update_name)
                                    elif self.platform == "TENSORFLOW":
                                        if self.update_barrier:
                                            update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                            self.wrap_add_dag(prev_name, update_name)
                                            prev_name = update_name
                                        for post_node in post_nodes:
                                            update_name = self.add_prefix(post_node, _prefix=next_rank_prefix)
                                            self.wrap_add_dag(prev_name, update_name)
                                            self.add_update_downstream(graph, post_node, _prefix=next_rank_prefix)
                                    else:
                                        raise NotImplementedError()
                ### end for loop
            elif self.nccl_graph is not None and self.nccl_graph.algo == NCCL_ALGO.TREE:
                ### Combine chunkId, sliceId and channelId into the graph for Tree algorithm
                ### TODO(huhanpeng): can we reduce the number of %d in the suffix
                raise NotImplementedError("Remove following todo first")
                ### TODO (huhanpeng): What if we consider Sync, Queue, Memcopy operators
                chunkNum, sliceNum, channelNum, loopNum = self.nccl_graph.get_IDnum(u)
                for loopId in range(loopNum):
                    for chunkId in range(chunkNum):
                        for sliceId in range(sliceNum):
                            for channelId in range(channelNum):
                                parent = self.nccl_graph.ret_parent(self.prefix, channelId)
                                childs = self.nccl_graph.ret_childs(self.prefix, channelId)
                                rank = self.nccl_graph.ret_rank_from_prefix(self.prefix)
                                if parent != -1:
                                    ### Not a root node

                                    ### 1. first handel UP process
                                    if len(childs) > 0:
                                        ### 1). Add edges from Recv to Aggerate Nodes first
                                        ### Use 0 to denote UP and 1 to denote Down
                                        next_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 0))
                                        for cld_rank in childs:
                                            prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, cld_rank, rank)) 
                                            ### TODO (huhanpeng) make sure name2sta should contain keys with suffix="%d_%d_%d_%d_%d"
                                            self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))
                                        ### 2). Add edges from Aggregate node to Send
                                        ### Use 0 to denote UP and 1 to denote Down
                                        ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                        prev_rawname = next_rawname
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))
                                    else:
                                        ### 1).2). The second case - The first step, connect BW node to NEGOTIATE_ALLREDUCE
                                        ### Then connect all NEGOTIATE_ALLREDUCE nodes from all ranks to the op first
                                        ### Add edges from all BW nodes to Aggerate Nodes first
                                        next_rawname = gen_long_name(None, "%s.%s"%(u, queue_type_list[0]), suffix=None)
                                        if self.wrap_in_dag(self.add_prefix(next_rawname)):
                                                ### has been processed, no edges shoud be added
                                                return
                                        # for pre_node in pre_nodes:
                                        #     self.wrap_add_dag(self.add_prefix(pre_node), self.add_prefix(next_rawname))

                                        next_rawname = pre_nodes[0]

                                        prev_name_base = next_rawname
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                        prev_nodes_prefix = self.nccl_graph.bw_to_first_send(channelId)
                                        for _prefix in prev_nodes_prefix:
                                            prev_name = self.add_prefix(prev_name_base, _prefix=_prefix)
                                            self.wrap_add_dag(prev_name, self.add_prefix(next_rawname))

                                    ### 3). Add edges from Send to Recv
                                    prev_rawname = next_rawname
                                    next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                    next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(parent)
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname, _prefix=next_rank_prefix))

                                    ### 2. Handel Down Process

                                    ### 1). Add edges from Recv to broadcast node, use 1 to denote Down 
                                    prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, parent, rank))
                                    next_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))
                                    
                                    ### -1): Add Recv to Step nodes, for Down process
                                    prev_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                    if self.platform == "MXNET":
                                        update_name = self.add_prefix("UPDATE_CAL")
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), update_name)
                                    elif self.platform == "TENSORFLOW":
                                        if self.update_barrier:
                                            update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                            self.wrap_add_dag(prev_name, update_name)
                                            prev_name = update_name
                                        for post_node in post_nodes:
                                            update_name = self.add_prefix(post_node)
                                            self.wrap_add_dag(self.add_prefix(prev_rawname), update_name)
                                            self.add_update_downstream(graph, post_node)
                                    else:
                                        raise NotImplementedError()
                                    # ### Connect all UPDATE nodes to an END node
                                    # self.wrap_add_dag(update_name, "END")
                                    for cld_rank in childs:
                                        ### 2). Add edges from broadcast node to Send node
                                        ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                        prev_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))

                                        ### 3). Add edges from Send to Recv
                                        prev_rawname = next_rawname
                                        next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                        next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(cld_rank)
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname, _prefix=next_rank_prefix))
                                        
                                else:
                                    ### Root Nodes
                                    for cld_rank in childs:
                                        ### 1). Add edges from Recv to Aggerate Nodes first
                                        ### Use 0 to denote UP and 1 to denote Down
                                        prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, cld_rank, rank)) 
                                        next_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 0))
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))

                                        ### 2). Add edges from broadcast node to Send node
                                        ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                        prev_rawname = next_rawname
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))

                                        ### 3). Add edges from Send to Recv
                                        prev_rawname = next_rawname
                                        next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                        next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(cld_rank)
                                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname, _prefix=next_rank_prefix))
            else:
                ### Normal Horovod, corse-grained (Including NEGOTIATE_..., ALL_REDUCE, etc )
                pre_nodes_ = pre_nodes
                for suffix in queue_type_list:
                    cur_node = gen_long_name(None, u, suffix=suffix)
                    if self.wrap_in_dag(self.add_prefix(cur_node)):
                        ### has been processed, no edges shoud be added
                        return
                    if self.traceM.lookup_stat(self.comm_backend, self.wk_prefix, self.local_rank, cur_node) == 0:
                        continue
                    for pre_node in pre_nodes_:
                        self.wrap_add_dag(self.add_prefix(pre_node), self.add_prefix(cur_node))
                    pre_nodes_ = [cur_node]
                if self.platform == "MXNET":
                    for update_id in post_nodes:
                        self.wrap_add_dag(self.add_prefix(pre_nodes_[0]), self.add_prefix("UPDATE_%d"%update_id))
                else:
                    raise NotImplementedError()
        elif "BW" in u and "Comm" in v:
            if self.single:
                self.wrap_add_dag(
                    self.add_prefix(u), self.add_prefix(v))
            else:
                ### delete edges from BW to Comm main task.
                return
        elif "UPDATE" in u and "FW" in v:
            ### ignore nodes from UPDATE to FW, avoid cycles
            return
        else:
            self.wrap_add_dag(self.add_prefix(u), self.add_prefix(v))

    def _process_edge_byteps(self, graph, queue_type_list, u, v):
        if "Comm." in u:
            ### BytePS traces
            tensor_name = u.split("Comm.")[1]
            if self.byteps_graph is not None:
                wk_rank = int(self.wk_prefix.split("_")[-1])
                # add push request dependency
                try:
                    push_req_nodes = self.byteps_graph.get_push_req_node(wk_rank, tensor_name)
                except:
                    SingleLogger().warn("{} is not in comm dag. Ignoring.".format(tensor_name))
                    return
                prev_bw_nodes = []
                if "Comm.{}".format(tensor_name) in graph.nodes:
                    prev_bw_nodes += [_u for _u in graph.predecessors("Comm.{}".format(tensor_name))]
                else:
                    for each_tensor_name in tensor_name.split("+"):
                        prev_bw_nodes += [_u for _u in graph.predecessors("Comm.{}".format(each_tensor_name))]
                prev_bw_nodes = set(prev_bw_nodes)
                for prev_bw_node in prev_bw_nodes:
                    prev_name = self.add_prefix(prev_bw_node)
                    for push_req_node in push_req_nodes:
                        self.wrap_add_dag(prev_name, push_req_node)
                # add dependencies to v
                update_nodes = []
                if "Comm.{}".format(tensor_name) in graph.nodes:
                    update_nodes += [_n for _n in graph.successors("Comm.{}".format(tensor_name))]
                else:
                    for each_tensor_name in tensor_name.split("+"):
                        update_nodes += [_n for _n in graph.successors("Comm.{}".format(each_tensor_name))]
                update_nodes = set(update_nodes)
                pull_res_nodes = self.byteps_graph.get_pull_res_node(wk_rank, tensor_name)
                for update_node in update_nodes:
                    next_name = self.add_prefix(update_node)
                    for pull_res_node in pull_res_nodes:
                        self.wrap_add_dag(pull_res_node, next_name)
            else:
                raise NotImplementedError("Tensorflow + NCCL not yet implemented.")
        elif "Comm." in v:
            ### delete edges from BW to Comm main task.
            pass
        else:
            self.wrap_add_dag(
                self.add_prefix(u), 
                self.add_prefix(v))

    def gen_dag_with_prefix_weight(self, old_graph, para_dict=None):
        ''' Gen a dag from the original graph with weighted edges.
        Return: A dag, which
            * is **weighted**;
            * contains FW, BW, OUTPUT, Comm, I/O and UPDATE nodes;
            * node names start with 'host{x}.rank{x}.';
            * partition Comm nodes into sub-task nodes if needed.
        '''
        ### Read the original dag for this gpu first
        mygraph = old_graph
        queue_type_list = QueueType("NCCL").ret_list()

        def _nccl_tensor_name2tensor_id(tensor_name):
            if self.platform == "MXNET":
                return para_dict.tensor_name_to_tensor_id(tensor_name)
            elif self.platform == "TENSORFLOW":
                return int(tensor_name)
            else:
                raise ArgumentError("Unsupported platform {}.".format(self.platform))
        
        def _nccl_tensor_id2tensor_name(tensor_id):
            if self.platform == "MXNET":
                return "Comm." + para_dict.tensor_id_to_tensor_name(int(tensor_id))   # e.g., Comm.bertmodel0_word_embed_embedding0_weight
            elif self.platform == "TENSORFLOW":
                return "Comm.{}".format(tensor_id)
            else:
                raise ArgumentError("Unsupported platform {}.".format(self.platform))

        done_comm = []
        done_sync = []

        for u, v in mygraph.edges:
            pre_nodes, post_nodes = [], []
            if "Comm." in u and self.nccl_graph is not None:
                ### Consider Tensor fusion, only those ready tensors are fused and are used to build a graph together
                tensor_name = u.split("Comm.")[1]
                tensor_id = _nccl_tensor_name2tensor_id(tensor_name)
   
                try:
                    nccl_grp_name = self.nccl_graph.tensor2group_name(tensor_id)
                except TypeError:
                    ### some tensors do not have grads, for MXNet
                    continue
                except:
                    if self.prefix == "host0.rank0":
                        print(u, v)
                        print(tensor_id)
                        print(self.nccl_graph.nccl_fusion["grp_names"])
                    continue
                
                if VIRTUAL_SYNC_OP:
                    nccl_grp_name_sync = nccl_grp_name
                else:
                    raise NotImplementedError("Need to 1. align the profiling range of Horovod;"
                                              "2) check whether some tensor ids have no grads")
                    nccl_grp_name_sync = self.nccl_graph.tensor2group_name_sync(tensor_id)

                ### handle the edges from BW to Comm.xxx.Sync
                sync_op = "Comm." + nccl_grp_name_sync + ".Sync"
                if sync_op not in done_sync:
                    for _id in nccl_grp_name_sync.split("+"):
                        co_comm_op = _nccl_tensor_id2tensor_name(_id)
                        prev_bw_nodes = list(mygraph.predecessors(co_comm_op))
                        # assert len(prev_bw_nodes) == 1, (co_comm_op, prev_bw_nodes)
                        prev_rawname = prev_bw_nodes[0]         # no prefix, start with BW.
                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(sync_op))
                    done_sync.append(sync_op)
                
                ### take the fused name as the node name, e.g., Comm.1+2+3
                u = "Comm." + nccl_grp_name
                if u in done_comm:
                    continue

                ### Set the `pre_nodes` and `post_nodes`
                # nodes in `pre_nodes` --> Comm.xxx.SEND --> ... --> nodes in `post_nodes`
                pre_nodes =[sync_op]
                for _id in nccl_grp_name.split("+"):
                    if self.platform == "MXNET":
                        # e.g., from tensor 256 to update 140
                        update_id = para_dict.tensor_id2update_id(int(_id))
                        post_nodes.append(update_id)
                    elif self.platform == "TENSORFLOW":
                        ### TODO (hphu): for a fused tensor, it corresponds to multiple updates which run cocurrently
                        # Current, only select one as the update operator
                        co_comm_op = "Comm.{}".format(_id)
                        post_update_nodes = list(mygraph.successors(co_comm_op))
                        post_nodes += post_update_nodes
                  
                done_comm.append(u)
            elif "Comm." in u and self.byteps_graph is not None:
                tensor_name = u.split("Comm.")[1]
                u = "Comm." + (tensor_name if "+" \
                    in tensor_name else self.byteps_graph.parse_tensor_grp_name(tensor_name))
                if u in done_comm:
                    continue
                done_comm.append(u)

            if self.byteps_graph is not None:
                self._process_edge_byteps(mygraph, queue_type_list, u, v)
            elif self.nccl_graph is not None:
                self._process_edge_nccl(mygraph, queue_type_list, u, v, para_dict=para_dict, pre_nodes=pre_nodes, post_nodes=post_nodes)
            elif self.single:
                self._process_edge_nccl(mygraph, queue_type_list, u, v, para_dict=para_dict, pre_nodes=pre_nodes, post_nodes=post_nodes)
                
        if self.byteps_graph is not None and self.platform == "MXNET":
            for update_id in range(para_dict.tensor_id2update_id("max") + 1):
                update_name = self.add_prefix("UPDATE_%d"%update_id)
                if self.update_barrier:
                    self.wrap_add_dag(self.add_prefix("UPDATE_CAL"), update_name)
                ### Connect all UPDATE nodes to an END node
                self.wrap_add_dag(update_name, "END")
        elif self.nccl_graph is not None and self.platform == "MXNET":
            # TODO (huhanpeng): need further to unify the name rule, for NCCL case
            # 1) What if there is no barrier ??? 
            # 2) connect the UPDATE_CAL to the following update nodes
            for update_id in range(para_dict.tensor_id2update_id("max") + 1):
                update_name = self.add_prefix("UPDATE_%d"%update_id)
                self.wrap_add_dag(self.add_prefix("UPDATE_CAL"), update_name)
                ### Connect all UPDATE nodes to an END node
                self.wrap_add_dag(update_name, "END")

        # visualize_gml(self.dag, layout="circular")

    def _add_new_edges_via_order(self):
        ''' Add new edges between FW+BW ops, according to their processing order
            such that we can keep the order when replaying.
            TODO (huhanpeng), do we need this
            Parameters
            ----------
            Returns
            ----------
            max_para_degree: int
                Maximum parallelism degree in computation nodes
        '''
        ### Used to store events currently in processing
        in_process_events = []
        ### maximum paralillism degree
        max_para_degree = 1
        ### record the start time of the first op
        first = True
        start_time = None

        def relative_time(time):
            return (time - start_time) / 1000.0

        def in_process_events2str():
            s = ''
            for _idx in in_process_events:
                _event = self.traceM.traces[_idx]
                _n, _ts, _te = _event["name"], _event["ts"], _event["ts"] + _event["dur"]
                s += "\n\t\t\t\t%-60s: %s~%s (%-13.4f ~ %-13.4f)" % (_n, str(_ts), str(_te), relative_time(_ts), relative_time(_te))
            return s

        ### For FW and BW nodes, go through one step of traces
        for idx, event in enumerate(self.traceM.traces):
            if self.traceM._is_ignore_for_sta(event):
                continue
            if event["args"]["step"] > (self.traceM.opt_step + 1):
                ### only go through one step of traces, even if there exists overlapping,
                # no possible overlapping between three steps
                break
            elif event["args"]["step"] != self.traceM.opt_step or event["pid"] != self.prefix:
                continue
            node_name = gen_long_name(event["pid"], event["name"])
            if first:
                SingleLogger().info("The first event - name: %s, ts: %s us, dur: %s us" %
                    (node_name, str(event["ts"]), str(event["dur"])))
                start_time = event["ts"]
                first = False

            #! only consider FW and BW nodes 
            if not self.is_fw_bw_node(node_name):
                continue

            i = 0
            while i < len(in_process_events):
                prev_event = self.traceM.traces[in_process_events[i]]
                assert event["ts"] >= prev_event["ts"]
                assert event["args"]["step"] == prev_event["args"]["step"]
                if event["ts"] >= prev_event["ts"] + prev_event["dur"]:
                    ### prev event has ended, should be deleted from in_process_events
                    del in_process_events[i]

                    ### TODO (huhanpeng) do not follow the dependency graph, may introduce cycle to the dependency graph
                    if "BW.bertencoder0_embedding0" in prev_event["name"] or "BW.bertencoder0_embedding0" in event["name"]:
                        continue

                    #! TODO: only add once, to verify
                    if prev_event["name"] != event["name"]:
                        u = self.add_prefix(prev_event["name"])
                        v = self.add_prefix(event["name"])
                        # TODO (huhanpeng): if not check this, may introduce cycle for tensorflow bert traces
                        # if not self.wrap_in_dag_edges((v, u)):
                        self.wrap_add_dag(u, v)
                        if "FW.bert/encoder/layer_0/attention/self/Softmax" in u and "FW.add_2" in v:
                            print(prev_event, event)
                            raise RuntimeError
                else:
                    ### if prev event has not ended, current node should share 
                    ### the parent ops of the prev event
                    ### TODO (huhanpeng): ignore this first, since we only consider one computation stream
                    ### need to test the correctness in multi-stream cases
                    # parent_list_of_prev = [u for u, _ in self.dag.in_edges(self.add_prefix(prev_event["name"]))]
                    # for u in parent_list_of_prev:
                    #     ### TODO (huhanpeng) do not follow the dependency graph, ignore now
                    #     if "BW.bertencoder0_embedding0" in u or "BW.bertencoder0_embedding0" in self.add_prefix(prev_event["name"]):
                    #         continue
                    #     self.wrap_add_dag(u, self.add_prefix(event["name"]))
                    i += 1

            if len(in_process_events) + 1 > max_para_degree:
                max_para_degree = len(in_process_events) + 1

            if len(in_process_events) > 0:
                SingleLogger().debug("%s (%-13.4f): D=%d => %-60s%s" %
                    (event["ts"], relative_time(event["ts"]),
                        len(in_process_events)+1,
                        event["name"], 
                        in_process_events2str()))
            in_process_events.append(idx)

        SingleLogger().info("Maximum parallelism degree: %d" % (max_para_degree))
        return max_para_degree

    def is_fw_bw_node(self, name):
        return parse_cat_fine_grained(name) in ["operator.FW", "operator.BW"]
    
    def gen_gpu_dag(self, old_graph, _pretty=False, para_dict=None):
        ''' Add edges according to the processing order of FW+BW ops 
            and construct a new graph running on GPU, which we call self.dag.
        Parameter
        __________
        para_dict: dict
            A dict which contains the meta info of gradients/parameters
            and maps from each gradients to its UPDATE operation id
        '''
        self.gen_dag_with_prefix_weight(old_graph, para_dict)

        critical_path = None
        ### generate execution graph according to the execution order,
        # to make sure replayer acts in the same order
        # max_para_degree = self._add_new_edges_via_order()

        #！til now, all the edges for one GPU have been added.
        if not _pretty:
            SingleLogger().info("Calculate critical path and check cycles. This process is time consuming, set --pretty to disable")
            composed_dag = nx.DiGraph()
            composed_dag.add_edges_from(self.dag)
            try:
                SingleLogger().info(list(nx.find_cycle(composed_dag, orientation="original")))
            except:
                SingleLogger().info("No cycle is found")
            # visualize_gml(composed_dag)
            critical_path = dag_longest_path(composed_dag, self.pm, weight="weight", default_weight=0)

        SingleLogger().debug("Generate a local dag with {} edges".format(len(self.dag)))
        # return max_para_degree, critical_path
        return 1, critical_path

    def all_topo_sorts(self):
        ''' generate all possible topological sorts '''
        flag = False

        for n in self._fw_bw_dag.nodes:
            if self._fw_bw_dag.node[n]["in_degree"] == 0 and not self._fw_bw_dag.node[n]["visited"]:
                #! All its successors 
                for next_n in self._fw_bw_dag.successors(n):
                    self._fw_bw_dag.node[next_n]["in_degree"] -= 1

                self._topo_sort.append(n)
                self._fw_bw_dag.node[n]["visited"] = True
                self.all_topo_sorts()

                self._fw_bw_dag.node[n]["visited"] = False
                self._topo_sort.pop()

                #! retrive dependency
                for next_n in self._fw_bw_dag.successors(n):
                    self._fw_bw_dag.node[next_n]["in_degree"] += 1

                flag = True

        if flag == False:
            # SingleLogger().info(str(self._topo_sort))
            self.topo_sorts.append(self._topo_sort)
            SingleLogger().info(self._topo_sort)

    def gen_fw_bw_dag(self):
        raise NotImplementedError("Adapt to list self.dag")
        self._fw_bw_dag = nx.DiGraph()
        self.gen_dag_with_prefix_weight()
        for u, v, _dict in self.dag.edges.data():
            if self.is_fw_bw_node(u) and self.is_fw_bw_node(v): 
                self._fw_bw_dag.add_edge(u, v, **_dict)
        for n, _dict in self._fw_bw_dag.nodes.data():
            _dict["in_degree"] = self._fw_bw_dag.in_degree(n)
            _dict["visited"] = False

        SingleLogger().info(list(self._fw_bw_dag.nodes))
        # self.all_topo_sorts()
        SingleLogger().info(len(self.topo_sorts))


class SmallDAGManager(DAGManager):
    def __init__(self, nrank, rank, traceM, nccl_graph=None, byteps_graph=None, platform="TENSORFLOW", single=False):
        self.platform = platform
        ### traceM's DirLevel = TRAIL
        self.traceM = traceM
        self.dag = []
        self.nodes = set()
        self._fw_bw_dag = None

        # TODO: delete
        self._topo_sort = []
        self.topo_sorts = []

        ### For fine-grained communication dependency
        # one and only one of NCCL_GRAPH or BYTEPS_GRAPH can be set at a time
        assert (nccl_graph or byteps_graph) and not (nccl_graph and byteps_graph)
        self.nccl_graph = nccl_graph
        self.byteps_graph = byteps_graph

        ### is the dag for single rank
        self.single = single

        self.nrank, self.rank = nrank, rank
        self.wk_prefix, self.rank_prefix = "host{}".format(self.rank), "rank0"
        self.prefix = "%s.%s" % (self.wk_prefix, self.rank_prefix)
        self.all_prefix = ["host{}.rank0".format(_id) for _id in range(self.nrank)]
    
    def _process_edge_nccl(self, graph, queue_type_list, u, v, para_dict=None, pre_nodes=[], post_nodes=[]):
        ''' Handel one edge in the original depedency graph
        Parameters
        ----------
        graph: class nx.Graph, the original depedency graph
        queue_type_list: str list
        '''
        if "Comm" in u:
            if self.single:
                ### add virtual Comm edges for single rank casts
                self.wrap_add_dag(self.add_prefix(u), self.add_prefix(v))
                return
            elif self.nccl_graph is not None and self.nccl_graph.algo == NCCL_ALGO.RING:
                ### Combine chunkId, sliceId and channelId into the graph for RING algorithm
                _, sliceNum, channelNum, loopNum = self.nccl_graph.get_IDnum(u)
                chunkNum = 2 * (self.nrank - 1)
                for loopId in range(loopNum):
                    for chunkId in range(chunkNum):
                        for sliceId in range(sliceNum):
                            for channelId in range(channelNum):
                                if chunkId == 0:
                                    ### The first step
                                    ### Connect BW nodes to Sync, if this is a fused tensor, there should be multiple BW nodes
                                    # next_rawname = gen_long_name(None, "%s.%s"%(u, queue_type_list[0]), suffix=None)
                                    # for pre_node in pre_nodes:
                                    #     self.wrap_add_dag(self.add_prefix(pre_node), self.add_prefix(next_rawname))

                                    next_rawname = pre_nodes[0]
                                    
                                    ### Connect all ranks' Sync to the first Send
                                    prev_rawname = next_rawname
                                    next_rawname = "%s.%s"%(u, queue_type_list[1])
                                    for _prefix in self.all_prefix:
                                        prev_name = self.add_prefix(prev_rawname, _prefix=_prefix)
                                        self.wrap_add_dag(prev_name, self.add_prefix(next_rawname))

                                    ### Queue --> MEMCPY_IN_FUSION_BUFFER
                                    prev_rawname = next_rawname 
                                    comm_in_name = "%s.%s"%(u, queue_type_list[2])
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(comm_in_name))

                                    ### MEMCPY_IN_FUSION_BUFFER to the first Send
                                    next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                    ### TODO (huhanpeng) MEMCPY_IN_FUSION_BUFFER to the first RECV
                                    next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                else:
                                    ### normal steps
                                    ### Connect Memory copy in to Send and Recv
                                    comm_in_name = gen_long_name(None, "%s.%s"%(u, queue_type_list[2]), suffix=None)                                    
                                    last_chunkId = chunkId - 1
                                    prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, last_chunkId, sliceId))
                                    next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(prev_rawname))
                                    self.wrap_add_dag(self.add_prefix(comm_in_name), self.add_prefix(next_rawname))
                                    ### Connect from Recv to Send
                                    self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname))

                                ### Connect from Send to Recv
                                next_rank_prefix = "host{}.rank0".format((self.rank + 1) % self.nrank)
                                next_chunkId = chunkId
                                prev_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, next_chunkId, sliceId))
                                self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(next_rawname, _prefix=next_rank_prefix))

                                if chunkId >= (2 * (self.nrank - 1) - 1):
                                    ### last RECV --> MEMCPY_OUT_FUSION_BUFFER
                                    prev_name = self.add_prefix(next_rawname, _prefix=next_rank_prefix)
                                    next_name = gen_long_name(next_rank_prefix, "%s.%s"%(u, queue_type_list[-1]), suffix=None)
                                    self.wrap_add_dag(prev_name, next_name)

                                    ### MEMCPY_OUT_FUSION_BUFFER --> UPDATE_CAL
                                    prev_name = next_name
                                    if self.platform == "MXNET":
                                        update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                        self.wrap_add_dag(prev_name, update_name)
                                    elif self.platform == "TENSORFLOW":
                                        if self.update_barrier:
                                            update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                            self.wrap_add_dag(prev_name, update_name)
                                            prev_name = update_name
                                        for post_node in post_nodes:
                                            update_name = self.add_prefix(post_node, _prefix=next_rank_prefix)
                                            self.wrap_add_dag(prev_name, update_name)
                                            self.add_update_downstream(graph, post_node, _prefix=next_rank_prefix)
                                    else:
                                        raise NotImplementedError()
                ### end for loop
            else:
                ### Normal Horovod, corse-grained (Including NEGOTIATE_..., ALL_REDUCE, etc )
                raise NotImplementedError("Remove following todo first")
        elif "BW" in u and "Comm" in v:
            if self.single:
                self.wrap_add_dag(
                    self.add_prefix(u), self.add_prefix(v))
            else:
                ### delete edges from BW to Comm main task.
                return
        elif "UPDATE" in u or "UPDATE" in v:
            ### ignore nodes from UPDATE to FW, avoid cycles
            return
        else:
            self.wrap_add_dag(self.add_prefix(u), self.add_prefix(v))
    

