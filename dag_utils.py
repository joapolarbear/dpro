import os
import networkx as nx
import matplotlib.pyplot as plt
import logger_utils
from trace_utils import *
from horovod.graph import *

def visualize_gml(graph, layout="circular"):
    if layout == "spectral":
        pos = nx.spectral_layout(graph, dim=2, scale=0.5)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "random":
        pos = nx.random_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_size=6)
    plt.show()
    # import matplotlib.pyplot as plt; plt.ion()
    # import netgraph
    # netgraph.draw(graph)
    # plot_instance = netgraph.InteractiveGraph(graph, node_positions=pos)
    # node_positions = plot_instance.node_positions

def dag_longest_path(G, pathM, weight='weight', default_weight=0, _debug_level=0):
    critical_path = nx.algorithms.dag.dag_longest_path(G, weight=weight, default_weight=default_weight)
    prefix = "Critical Path of " + pathM.ret_id_in_trial()
    logger = SingleLogger()
    if _debug_level > 1:  
        logger.info(prefix + " => ")
    path_length = 0
    len_list = []
    for (u, v) in nx.utils.pairwise(critical_path):
        weight_ = G[u][v].get(weight, default_weight)
        path_length += weight_
        if _debug_level > 1:
            logger.info("%-80s: %f ms" % (u, weight_))
        len_list.append(weight_)
    len_list.append(0)
    # logger.info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
    if _debug_level > 0:
        logger.info("Length of the " + prefix + "%12.4f ms\n" % path_length)

    return list(zip(critical_path, len_list))

class DAGManager:
    '''
    Parameters
    ----------
    path: str
        Root path for one GPU
    '''
    def __init__(self, path, traceM, nccl_graph=None):
        self.pm = PathManager(path)
        ### traceM's DirLevel = TRAIL
        self.traceM = traceM
        self.logger = logger_utils.SingleLogger()
        self.dag = self.gpu_dag = self._fw_bw_dag = None

        self.wk_prefix, self.rank_prefix = self.pm.ret_prefix()
        self.prefix = "%s.%s" % (self.wk_prefix, self.rank_prefix)
            
        self._topo_sort = []
        self.topo_sorts = []

        ### For fine-grained communication dependency
        self.nccl_graph = nccl_graph

    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.prefix, name)
        else:
            return gen_long_name(_prefix, name)

    def gen_dag_with_prefix_weight(self, update_dict=None):
        ''' Gen a dag from the original graph with weighted edges.
        Args:
            gml_path: stores the dag output by byteprofile
                TODO, all Comm OPs of one single gradients are considered as one node.
        Return: A dag, which
            * is **weighted**;
            * containing FW, BW, OUTPUT, Comm, I/O and UPDATE nodes;
            * node names start with 'rank{id}.';
            * partition Comm nodes into sub-task nodes if needed.
        '''
        ### Read the original dag for this gpu first
        mygraph = nx.read_gml(self.pm.search(FileName.DAG))
        self.dag = nx.DiGraph()
        queue_type_list = QueueType().ret_list()

        for u, v in mygraph.edges:
            if "Comm" in u:
                gra_name = u.split("Comm.")[1]
                # update_id = 0 if update_dict is None else update_dict[gra_name]
                if self.nccl_graph is not None and self.nccl_graph.algo == NCCL_ALGO.RING:
                    ### Combine chunkId, sliceId and channelId into the graph for RING algorithm
                    chunkNum, sliceNum, channelNum, loopNum = self.nccl_graph.get_IDnum(u)
                    for loopId in range(loopNum):
                        for chunkId in range(chunkNum):
                            for sliceId in range(sliceNum):
                                for channelId in range(channelNum):
                                    if self.nccl_graph.is_first_step(chunkId):
                                        ### The first step

                                        ### Connect BW nodes to NEGOTIATE_ALLREDUCE 
                                        prev_fw_nodes = [_u for _u, _ in mygraph.in_edges(u)]
                                        assert len(prev_fw_nodes) == 1
                                        prev_rawname = prev_fw_nodes[0]
                                        next_rawname = gen_long_name(None, "%s.%s"%(u, queue_type_list[0]), suffix=None)
                                        self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname), 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))

                                        ### Connect all ranks' NEGOTIATE_ALLREDUCE to the first
                                        prev_rawname = next_rawname
                                        prev_nodes_prefix = self.nccl_graph.bw_to_first_send(channelId)
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                        for _prefix in prev_nodes_prefix:
                                            prev_name = self.add_prefix(prev_rawname, _prefix=_prefix)
                                            self.dag.add_edge(
                                                prev_name, 
                                                self.add_prefix(next_rawname), 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_name))
                  
                                        ### Connect from Send to Recv
                                        next_rank_prefix, next_chunkId = self.nccl_graph.send_to_recv(self.prefix, chunkId, channelId)
                                        prev_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                        next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, next_chunkId, sliceId))
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            self.add_prefix(next_rawname, _prefix=next_rank_prefix), 
                                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))
                                        
                                    else:
                                        ### normal steps
                                        ### Connect from Recv to Send
                                        _, last_chunkId = self.nccl_graph.send_to_last_recv(self.prefix, chunkId)
                                        prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, last_chunkId, sliceId))
                                        next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            self.add_prefix(next_rawname), 
                                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))   

                                        ### Connect from Send to Recv
                                        next_rank_prefix, next_chunkId = self.nccl_graph.send_to_recv(self.prefix, chunkId, channelId)
                                        prev_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId))
                                        next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d"%(loopId, channelId, next_chunkId, sliceId))
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            self.add_prefix(next_rawname, _prefix=next_rank_prefix), 
                                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))   
                                        
                                        if self.nccl_graph.is_last_step(chunkId):
                                            prev_name = self.add_prefix(next_rawname, _prefix=next_rank_prefix)
                                            update_name = self.add_prefix("UPDATE_CAL", _prefix=next_rank_prefix)
                                            self.dag.add_edge(
                                                prev_name, 
                                                update_name, 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_name))
                                            # ### Connect all UPDATE nodes to an END node
                                            # self.dag.add_edge(update_name, "END",
                                            #     weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, update_name))
                    ### end for loop         
                elif self.nccl_graph is not None and self.nccl_graph.algo == NCCL_ALGO.TREE:
                    ### Combine chunkId, sliceId and channelId into the graph for Tree algorithm
                    ### TODO(huhanpeng): can we reduce the number of %d in the suffix
                    raise NotImplementedError("Remove following todo first")
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
                                                self.dag.add_edge(
                                                    self.add_prefix(prev_rawname), 
                                                    self.add_prefix(next_rawname), 
                                                    weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname)
                                                )
                                            ### 2). Add edges from Aggregate node to Send
                                            ### Use 0 to denote UP and 1 to denote Down
                                            ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                            prev_rawname = next_rawname
                                            next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname), 
                                                weight=0)
                                        else:
                                            ### 1).2). The second case - The first step, connect BW node to NEGOTIATE_ALLREDUCE
                                            ### Then connect all NEGOTIATE_ALLREDUCE nodes from all ranks to the op first
                                            ### Add edges from all BW nodes to Aggerate Nodes first
                                            prev_fw_nodes = [_u for _u, _ in mygraph.in_edges(u)]
                                            assert len(prev_fw_nodes) == 1
                                            prev_rawname = prev_fw_nodes[0]
                                            next_rawname = gen_long_name(None, "%s.%s"%(u, queue_type_list[0]), suffix=None)
                                            self.dag.add_edge(
                                                    self.add_prefix(prev_rawname), 
                                                    self.add_prefix(next_rawname), 
                                                    weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))

                                            prev_name_base = next_rawname
                                            next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                            prev_nodes_prefix = self.nccl_graph.bw_to_first_send(channelId)
                                            for _prefix in prev_nodes_prefix:
                                                prev_name = self.add_prefix(prev_name_base, _prefix=_prefix)
                                                self.dag.add_edge(
                                                    prev_name, 
                                                    self.add_prefix(next_rawname), 
                                                    weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_name))

                                        ### 3). Add edges from Send to Recv
                                        prev_rawname = next_rawname
                                        next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, parent))
                                        next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(parent)
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            self.add_prefix(next_rawname, _prefix=next_rank_prefix), 
                                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname)) 

                                        ### 2. Handel Down Process

                                        ### 1). Add edges from Recv to broadcast node, use 1 to denote Down 
                                        prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, parent, rank))
                                        next_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            self.add_prefix(next_rawname), 
                                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))
                                        
                                        ### -1): Add Recv to Step nodes, for Down process
                                        prev_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                        update_name = self.add_prefix("UPDATE_CAL")
                                        self.dag.add_edge(
                                            self.add_prefix(prev_rawname), 
                                            update_name, 
                                            weight=0)
                                        # ### Connect all UPDATE nodes to an END node
                                        # self.dag.add_edge(update_name, "END",
                                        #     weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, update_name))

                                        for cld_rank in childs:
                                            ### 2). Add edges from broadcast node to Send node
                                            ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                            prev_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 1))
                                            next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname), 
                                                weight=0)

                                            ### 3). Add edges from Send to Recv
                                            prev_rawname = next_rawname
                                            next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                            next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(cld_rank)
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname, _prefix=next_rank_prefix), 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname)) 
                                            
                                    else:
                                        ### Root Nodes
                                        for cld_rank in childs:
                                            ### 1). Add edges from Recv to Aggerate Nodes first
                                            ### Use 0 to denote UP and 1 to denote Down
                                            prev_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, cld_rank, rank)) 
                                            next_rawname = gen_long_name(None, "%s.AGGR"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, 0))
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname), 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname)
                                            )

                                            ### 2). Add edges from broadcast node to Send node
                                            ### TODO (huhanpeng): If we need to consider the aggregation time, consider following weight
                                            prev_rawname = next_rawname
                                            next_rawname = gen_long_name(None, "%s.SEND"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname), 
                                                weight=0)

                                            ### 3). Add edges from Send to Recv
                                            prev_rawname = next_rawname
                                            next_rawname = gen_long_name(None, "%s.RECV"%u, suffix="%d_%d_%d_%d_%d_%d"%(loopId, channelId, chunkId, sliceId, rank, cld_rank))
                                            next_rank_prefix = self.nccl_graph.ret_prefix_from_rank(cld_rank)
                                            self.dag.add_edge(
                                                self.add_prefix(prev_rawname), 
                                                self.add_prefix(next_rawname, _prefix=next_rank_prefix), 
                                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_rawname))             
                else:
                    ### Normal Horovod, corse-grained (Including NEGOTIATE_..., ALL_REDUCE, etc )
                    prev_fw_nodes = [_u for _u, _ in mygraph.in_edges(u)]
                    assert len(prev_fw_nodes) == 1
                    prev_node = prev_fw_nodes[0]
                    for suffix in queue_type_list:
                        cur_node = gen_long_name(None, u, suffix=suffix)
                        if self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, cur_node) == 0:
                            continue
                        self.dag.add_edge(
                                self.add_prefix(prev_node), 
                                self.add_prefix(cur_node), 
                                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_node)
                            )
                        prev_node = cur_node
                    self.dag.add_edge(
                            self.add_prefix(prev_node), 
                            self.add_prefix("UPDATE_%d"%update_id), 
                            weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, prev_node)
                        )
            elif "BW" in u and "Comm" in v:
                ### delete edges from BW to Comm main task.
                pass
            elif "UPDATE" in u and "FW" in v:
                ### ignore nodes from UPDATE to FW, avoid a circle
                pass
            else:
                self.dag.add_edge(
                    self.add_prefix(u), 
                    self.add_prefix(v), 
                    weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, u)
                )

        for update_id in range(update_dict["max"] + 1):
            update_name = self.add_prefix("UPDATE_%d"%update_id)
            self.dag.add_edge(
                self.add_prefix("UPDATE_CAL"), 
                update_name, 
                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, self.add_prefix("UPDATE_CAL"))
                )
            ### Connect all UPDATE nodes to an END node
            self.dag.add_edge(update_name, "END",
                weight=self.traceM.lookup_stat(self.wk_prefix, self.rank_prefix, update_name))
        self.dag.remove_node(self.add_prefix("UPDATE"))

        for e in self.dag.edges.data("weight"):
            self.logger.debug(e)
        # visualize_gml(self.dag, layout="circular")
        # raise

    def _add_new_edges_via_order(self, _pretty):
        ''' Add new edges between FW+BW ops, according to their processing order
            such that we can keep the order when replaying.
            TODO (huhanpeng)
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
            for _event in in_process_events:
                _n, _ts, _te = _event["name"], _event["ts"], _event["ts"] + _event["dur"]
                s += "\n\t\t\t\t%-60s: %s~%s (%-13.4f ~ %-13.4f)" % (_n, str(_ts), str(_te), relative_time(_ts), relative_time(_te))
            return s

        ### Used to mark the arrival of each FW+BW op
        fw_bw_arrive = set()
        comm_cnt = 0
        for node in self.dag.nodes:
            if self.prefix not in node:
                continue
            if "FW" in node or "BW" in node:
                fw_bw_arrive.add(node)
            elif "Comm" in node:
                comm_cnt += 1
        self.logger.info("Total number of operators: %d" % len(fw_bw_arrive))
        self.logger.info("Total number of Comm OPs: %d" % comm_cnt)

        ### For FW and BW nodes, go through one step of traces
        for event in self.traceM.traces:
            if event["pid"] != self.prefix:
                continue
            node_name = gen_long_name(event["pid"], event["name"])
            if first:
                self.logger.info("The first event - name: %s, ts: %s us, dur: %s us" %
                    (node_name, str(event["ts"]), str(event["dur"])))
                start_time = event["ts"]
                first = False

            #! only consider FW and BW nodes
            if event["cat"] != "operator" or "UPDATE_" in node_name:
                continue

            #! TODO(huhanpeng): will never break, since some BW nodes do not exist.
            #! but can still avoid repeated processing
            if node_name in fw_bw_arrive:
                fw_bw_arrive.remove(node_name)
                if len(fw_bw_arrive) == 0:
                    break
            else:
                #! ignore some trival nodes or the nodes which appears for the second time.
                continue

            i = 0
            while True:
                if i >= len(in_process_events):
                    break
                prev_event = in_process_events[i]
                assert event["ts"] >= prev_event["ts"]
                assert event["args"]["cnt"] == prev_event["args"]["cnt"]
                if event["ts"] >= prev_event["ts"] + prev_event["dur"]:
                    ### prev event has ended, should be deleted from in_process_events
                    del in_process_events[i]

                    ### TODO (huhanpeng) do not follow the dependency graph, ignore now
                    if "BW.bertencoder0_embedding0" in prev_event["name"] or "BW.bertencoder0_embedding0" in event["name"]:
                        continue
                    #! TODO: only add once, to verify
                    self.gpu_dag.add_edge(
                        self.add_prefix(prev_event["name"]), 
                        self.add_prefix(event["name"]), 
                        weight=0)
                else:
                    ### if prev event has not ended, current node should share 
                    ### the parent ops of the prev event
                    parent_list_of_prev = [(u, self.gpu_dag.edges[(u, v)]["weight"]) for u, v in self.gpu_dag.in_edges(self.add_prefix(prev_event["name"]))]
                    for u, w in parent_list_of_prev:
                        ### TODO (huhanpeng) do not follow the dependency graph, ignore now
                        if "BW.bertencoder0_embedding0" in u or "BW.bertencoder0_embedding0" in v:
                            continue
                        self.gpu_dag.add_edge(u, self.add_prefix(event["name"]), weight=0)
                    i += 1

            if len(in_process_events) + 1 > max_para_degree:
                max_para_degree = len(in_process_events) + 1

            if not _pretty and len(in_process_events) > 0:
                self.logger.info("%s (%-13.4f): D=%d => %-60s%s" %
                    (event["ts"], relative_time(event["ts"]),
                        len(in_process_events)+1,
                        event["name"], 
                        in_process_events2str()))
            in_process_events.append(event)

        self.logger.info("Maximum parallelism degree: %d, remain %d FW+BW node(s)" % (max_para_degree, len(fw_bw_arrive)))
        return max_para_degree

    def is_fw_bw_node(self, name):
        return "BW" in name or "FW" in name
    
    def gen_gpu_dag(self, _pretty=False, update_dict=None):
        ''' Add edges according to the processing order of FW+BW ops 
            and construct a new graph running on GPU, which we call gpu_dag.
        Parameter
        __________
        update_dict: dict
            A dict which maps from each gradients to its UPDATE operation id
        '''
        self.gen_dag_with_prefix_weight(update_dict)
        self.gpu_dag = self.dag

        critical_path = None
        max_para_degree = self._add_new_edges_via_order(_pretty)
        max_para_degree = None

        #！til now, all the edges for one GPU have been added.
        if not _pretty:
            critical_path = dag_longest_path(self.gpu_dag, self.pm, weight="weight", default_weight=0)

        return max_para_degree, critical_path

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
            # self.logger.info(str(self._topo_sort))
            self.topo_sorts.append(self._topo_sort)
            self.logger.info(self._topo_sort)

    def gen_fw_bw_dag(self):
        self._fw_bw_dag = nx.DiGraph()
        self.gen_dag_with_prefix_weight()
        for u, v, _dict in self.dag.edges.data():
            if self.is_fw_bw_node(u) and self.is_fw_bw_node(v): 
                self._fw_bw_dag.add_edge(u, v, **_dict)
        for n, _dict in self._fw_bw_dag.nodes.data():
            _dict["in_degree"] = self._fw_bw_dag.in_degree(n)
            _dict["visited"] = False

        self.logger.info(list(self._fw_bw_dag.nodes))
        # self.all_topo_sorts()
        self.logger.info(len(self.topo_sorts))





