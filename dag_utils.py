from argparse import ArgumentError
import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import logger_utils
import arg_utils
from trace_utils import *

args_ = arg_utils.SingleArg().args
if args_.comm_backend == "NCCL":
    from hvd.graph import *
elif args_.comm_backend == "BYTEPS":
    from bps_helper.graph import *

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

def cal_edge_cost(G):
    for u, v in G.edges:
        gap = 0
        prev_cat = parse_cat_from_name(u)
        next_cat = parse_cat_from_name(v)
        for key, value in G.nodes[u].items():
            if "GAP" in key:
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
            SingleLogger().info("%-80s: %f ms" % (u, weight_))
        len_list.append(weight_)
    len_list.append(0)
    # SingleLogger().info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
    if _debug_level > 0:
        SingleLogger().info("Length of the " + prefix + "%12.4f ms\n" % path_length)

    return list(zip(critical_path, len_list))

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
            if "Switch" in _name:
                _name = "BW." + tensor_name + "_Switch"
            else:
                _name = "Comm." + tensor_name
        elif "cond_" in _name and "Switch" in _name:
            _name = "BW." + _name
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

def wrap_read_gml(gml_path, platform="MXNET"):
    ''' The node name in Tensorflow is not standard, transfer it to standard form first
        Tranverse the dag nodes twice
    '''
    mygraph = nx.read_gml(gml_path)
    if not args_.pretty:
        try:
            SingleLogger().info(list(nx.find_cycle(mygraph, orientation="original")))
        except:
            SingleLogger().info("No cycles found")
    if platform == "TENSORFLOW":
        update_nodes_in_dag = set()
        def recursive_add_succs(_node):
            for succ_ in mygraph.successors(_node):
                update_nodes_in_dag.add(succ_)
                recursive_add_succs(succ_)
                if "^"+succ_ in mygraph.nodes:
                    recursive_add_succs("^"+succ_)
        # mygraph.add_edges_from([(n[1:], n) for n in mygraph.nodes if n.startswith("^")])
        for node in mygraph.nodes:
            if "allreduce" in node.lower() or "bytepspushpull" in node.lower() \
                                            and "switch" not in node.lower():
                ### node is the Comm node, add its downstream nodes as update operators
                recursive_add_succs(node)
            # if "apply" in node.lower() or ("gradientdescent" in node.lower() and "learning_rate" not in node.lower()):
            #     update_nodes_in_dag.add(node)
        new_graph = nx.DiGraph()
        for u, v in mygraph.edges:
            nu, nv = tf_relabel_func(u, update_nodes_in_dag), tf_relabel_func(v, update_nodes_in_dag)
            assert not (nu.startswith("Comm") and nv.startswith("Comm")), (u, v)
            new_graph.add_edge(nu, nv)
        mygraph = new_graph
    else:
        update_nodes_in_dag = None
    return mygraph, update_nodes_in_dag

def wrap_read_graphdef(graphdef_path):
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

def standard_name(_name, platform="TENSORFLOW", update_nodes_in_dag=None):
    '''Fetch and handle the trace name'''
    ### TODO combine this function with wrap_read_gml, test MXNET
    if platform == "MXNET":
        #! add for mxnet-gluon case
        if "name=" in _name:
            _name = _name.split("name=")[1].split(";")[0]
        #! backward nodes or forward nodes
        _name = "BW." + _name.split("_backward")[0] if "_backward" in _name else "FW." + _name
        _name = _name.split("_fwd")[0] if "_fwd" in _name else _name
    elif platform == "TENSORFLOW":
        _name = tf_relabel_func(_name, update_nodes_in_dag)
    return _name

class DAGManager:
    ''' Maintain a dependency graph for one GPU
    Parameters
    ----------
    path: str
        Root path for one GPU
    
    e.g. For NCCL ALLREDUCE RING
    Note: Sync used to sync between ranks

    FW ---> OUTPUT ---> BW ------------------- .................. --------------------> UPDATE_CAL ---> UPDATE_<id> ---> END
                         \\                                                         ^  (barrier)
                          \\                                                       //
                            -> Comm.<>.Sync --------> Comm.<>.SEND~>x_x_x_x ...
                                                  \\   ^
                                                   \\ //
                                                     x
                                                   // \\
                                                  //   V
                            -> Comm.<>.Sync --------> Comm.<>.SEND~>x_x_x_x ...
                          //                                                       \\
                         //                                                         V  (barrier)
    FW ---> OUTPUT ---> BW ------------------- .................. --------------------> UPDATE_CAL ---> UPDATE_<id> ---> END
    '''
    def __init__(self, path, traceM, nccl_graph=None, byteps_graph=None, platform="TENSORFLOW", single=False):
        self.pm = PathManager(path)
        self.platform = platform
        ### traceM's DirLevel = TRAIL
        self.traceM = traceM
        self.dag = []
        self.nodes = set()
        self._fw_bw_dag = None

        self.wk_prefix, self.rank_prefix = self.pm.ret_prefix()
        self.prefix = "%s.%s" % (self.wk_prefix, self.rank_prefix)

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
                                        if args_.update_barrier:
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
                                        if args_.update_barrier:
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
                    if self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, cur_node) == 0:
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
        elif "UPDATE" in u or "UPDATE" in v:
            ### ignore nodes from UPDATE to FW, avoid cycles
            return
        else:
            self.wrap_add_dag(self.add_prefix(u), self.add_prefix(v))

    def _process_edge_byteps(self, graph, queue_type_list, u, v):
        if "BytePSPushPull" in u and "tensor" not in u:
            ### BytePS traces
            gra_name = u.split("Comm.")[1]
            if self.byteps_graph is not None:
                wk_rank = int(self.wk_prefix.split("_")[-1])
                # add push request dependency
                try:
                    push_req_nodes = self.byteps_graph.get_push_req_node(wk_rank, gra_name)
                except:
                    SingleLogger().warn("{} is not in comm dag. Ignoring.".format(gra_name))
                    return
                prev_bw_nodes = [_u for _u, _ in graph.in_edges(u)]
                for prev_bw_node in prev_bw_nodes:
                    prev_name = self.add_prefix(standard_name(prev_bw_node, platform=self.platform))
                    for push_req_node in push_req_nodes:
                        self.wrap_add_dag(prev_name, push_req_node)
                # add dependencies to v
                pull_res_nodes = self.byteps_graph.get_pull_res_node(wk_rank, gra_name)
                for pull_res_node in pull_res_nodes:
                    self.wrap_add_dag(pull_res_node, self.add_prefix(standard_name(v, platform=self.platform)))
                    # print("adding edge {} -> {}".format(pull_res_node, self.add_prefix(standard_name(v, platform=self.platform))))
            else:
                raise NotImplementedError("Tensorflow + NCCL not yet implemented.")
        elif "BytePSPushPull" in v and "tensor" not in v:
            ### delete edges from BW to Comm main task.
            pass
        # elif "" in v:
            # avoid circle
            # pass
        else:
            self.wrap_add_dag(
                self.add_prefix(standard_name(u, platform=self.platform)), 
                self.add_prefix(standard_name(v, platform=self.platform)))

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

        done_comm = []
        done_sync = []
        for u, v in mygraph.edges:
            pre_nodes, post_nodes = [], []
            if "Comm." in u and self.nccl_graph is not None:
                ### Consider Tensor fusion, only those ready tensors are fused and are used to build a graph together
                tensor_name = u.split("Comm.")[1]
                if self.platform == "MXNET":
                    tensor_id = para_dict.name_to_tensor_id(tensor_name)
                elif self.platform == "TENSORFLOW":
                    tensor_id = int(tensor_name)
                else:
                    raise ArgumentError("Unsupported platform {}.".format(self.platform))
                try:
                    nccl_grp_name = self.nccl_graph.tensor2group_name(tensor_id)
                    nccl_grp_name_sync = self.nccl_graph.tensor2group_name_sync(tensor_id)
                except TypeError:
                    ### some tensors do not have grads
                    continue

                ### handle the edges from BW to Comm.xxx.Sync
                sync_op = "Comm." + nccl_grp_name_sync + ".Sync"
                if sync_op not in done_sync:
                    for _id in nccl_grp_name_sync.split("+"):
                        if self.platform == "MXNET":
                            co_comm_op = "Comm." + para_dict.tensor_id_to_name(int(_id))   # e.g., Comm.bertmodel0_word_embed_embedding0_weight
                        elif self.platform == "TENSORFLOW":
                            co_comm_op = "Comm.{}".format(_id)
                        else:
                            raise ArgumentError("Unsupported platform {}.".format(self.platform))
                        prev_bw_nodes = [_u for _u, _ in mygraph.in_edges(co_comm_op)]
                        # assert len(prev_bw_nodes) == 1, (co_comm_op, prev_bw_nodes)
                        prev_rawname = prev_bw_nodes[0]         # no prefix, start with BW.
                        self.wrap_add_dag(self.add_prefix(prev_rawname), self.add_prefix(sync_op))
                    done_sync.append(sync_op)
                
                ### take the fused name as the node name, e.g., Comm.1+2+3
                u = "Comm." + nccl_grp_name
                if u in done_comm:
                    continue
                pre_nodes =[sync_op]
                _first = True
                for _id in nccl_grp_name.split("+"):
                    if self.platform == "MXNET":
                        # e.g., from tensor 256 to update 140
                        update_id = para_dict.tensor_id2update_id(int(_id))
                        post_nodes.append(update_id)
                    elif self.platform == "TENSORFLOW":
                        ### TODO (hphu): for a fused tensor, it corresponds to multiple updates which run cocurrently
                        # Current, only select one as the update operator
                        if _first:
                            post_update_nodes = list(mygraph.successors(co_comm_op))
                            post_nodes += post_update_nodes
                            _first = False
                done_comm.append(u)
                
            
            if self.byteps_graph is not None:
                self._process_edge_byteps(mygraph, queue_type_list, u, v)
            elif self.nccl_graph is not None:
                self._process_edge_nccl(mygraph, queue_type_list, u, v, para_dict=para_dict, pre_nodes=pre_nodes, post_nodes=post_nodes)

        if self.byteps_graph is not None and self.platform == "MXNET":
            for update_id in range(para_dict.tensor_id2update_id("max") + 1):
                update_name = self.add_prefix("UPDATE_%d"%update_id)
                if args_.update_barrier:
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





