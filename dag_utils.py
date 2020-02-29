import os
import networkx as nx
import matplotlib.pyplot as plt
import logger_utils
from trace_utils import read_traces, return_stat, return_path_dict, QueueType, lookup_stat

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

def dag_longest_path(G, local_rank, logger, weight='weight', default_weight=0):
    critical_path = nx.algorithms.dag.dag_longest_path(G, weight=weight, default_weight=default_weight)
    prefix = "Critical Path of " + ("the Entire Graph: " if local_rank == -1 else "GPU-%d: " % local_rank)
    logger.info(prefix + " => ")
    path_length = 0
    for (u, v) in nx.utils.pairwise(critical_path):
        path_length += G[u][v].get(weight, default_weight)
        logger.info("%-80s: %f ms" % (u, G[u][v].get(weight, default_weight)))
    # logger.info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
    logger.info("Length of the " + prefix + "%12.4f ms\n" % path_length)
    return critical_path

class DAGManager:
    '''
    Parameters
    ----------
    path: str
        Root path for one GPU
    '''
    def __init__(self, path, local_rank, del_queue):
        self.path_dict = return_path_dict(path)
        self.traces = read_traces(self.path_dict["trace_path"])
        self.logger = logger_utils.SingleLogger()
        self.name2sta, cat2sta = return_stat(self.traces)
        self.dag = self.gpu_dag = self._fw_bw_dag = None
        self.local_rank = local_rank
        self.del_queue = del_queue

        # TODO: delete
        tmp = []
        for trace in self.traces:
            if trace["ts"] is not None:
                tmp.append(trace)
        self.traces = tmp
            
        self.traces = sorted(self.traces, key=lambda x: (x["ts"], x["name"]), reverse=False)

        self._topo_sort = []
        self.topo_sorts = []

    def add_prefix(self, name):
        if "Comm" in name:
            return name
        else:
            return "rank%d."%self.local_rank + name

    def gen_dag_with_rank_weight(self):
        ''' Gen a dag from the original graph with weighted edges.
        Args:
            gml_path: stores the dag output by byteprofile
                TODO, all Comm OPs of one single gradients are considered as one node.
            del_queue: if set True, `BW -> Comm_main_task -> FW` edges will 
                be substituded with `BW -> Comm_sub_task1 -> Comm_sub_task2 ... -> FW` edges
        Return: A dag, which
            * is **weighted**;
            * containing FW, BW, OUTPUT, Comm, I/O and STEP nodes;
            * node names start with 'rank{id}.';
            * partition Comm nodes into sub-task nodes if needed.
        '''
        mygraph = nx.read_gml(self.path_dict["gml_path"])
        self.dag = nx.DiGraph()
        queue_type_list = QueueType().ret_list()

        for u, v in mygraph.edges:
            if "Comm" in u:
                if self.del_queue == True:
                    prev_fw_nodes = [_u for _u, _ in mygraph.in_edges(u)]
                    assert len(prev_fw_nodes) == 1
                    prev_node = prev_fw_nodes[0]
                    for suffix in queue_type_list:
                        cur_node = u + '.' + suffix
                        if lookup_stat(self.name2sta, cur_node) == 0:
                            continue
                        self.dag.add_edge(self.add_prefix(prev_node), self.add_prefix(cur_node), weight=lookup_stat(self.name2sta, prev_node))
                        prev_node = cur_node
                    self.dag.add_edge(self.add_prefix(prev_node), self.add_prefix("STEP"), weight=lookup_stat(self.name2sta, prev_node))
                else:
                    self.dag.add_edge(self.add_prefix(u), self.add_prefix("STEP"), weight=lookup_stat(self.name2sta, u))
            elif "BW" in u and "Comm" in v and self.del_queue == True:
                #! if del_queue is set True, delete edges from BW to Comm main task.
                pass
            elif "STEP" in u and "FW" in v:
                #! ignore nodes from STEP to FW, avoid cycle
                pass
            else:
                self.dag.add_edge(self.add_prefix(u), self.add_prefix(v), weight=lookup_stat(self.name2sta, u))

        for e in self.dag.edges.data("weight"):
            self.logger.debug(e)
        # visualize_gml(self.dag, layout="circular")

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
            if "FW" in node or "BW" in node:
                fw_bw_arrive.add(".".join(node.split(".")[1:]))
            elif "Comm" in node:
                comm_cnt += 1
        self.logger.info("Total number of operators: %d" % len(fw_bw_arrive))
        self.logger.info("Total number of Comm OPs: %d" % comm_cnt)
      
        ### For FW and BW nodes, go through one step of traces
        for event in self.traces:
            if first:
                self.logger.info("The first event - name: %s, ts: %s us, dur: %s us" %
                    (event["name"], str(event["ts"]), str(event["dur"])))
                start_time = event["ts"]
                first = False

            #! only consider FW and BW nodes
            if event["cat"] != "operator" or "STEP" in event["name"]:
                continue

            #! TODO(huhanpeng): will never break, since some BW nodes do not exist.
            #! but can still avoid repeated processing
            node_name = event["name"]
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
                if event["ts"] >= prev_event["ts"] + prev_event["dur"]:
                    ### prev event has ended, should be deleted from in_process_events
                    del in_process_events[i]
                    #! TODO: only add once, to verify
                    self.gpu_dag.add_edge(self.add_prefix(prev_event["name"]), self.add_prefix(event["name"]), weight=lookup_stat(self.name2sta, prev_event))
                else:
                    ### if prev event has not ended, current node should share 
                    ### the parent ops of the prev event
                    parent_list_of_prev = [(u, self.gpu_dag.edges[(u, v)]["weight"]) for u, v in self.gpu_dag.in_edges(self.add_prefix(prev_event["name"]))]
                    for u, w in parent_list_of_prev:
                        self.gpu_dag.add_edge(u, self.add_prefix(event["name"]), weight=w)
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

        self.logger.info("Maximum parallelism degree: %d" % max_para_degree)
        return max_para_degree

    def is_computation_node(self, _node):
        return "BW" in _node or "FW" in _node
    
    def gen_gpu_dag(self, _pretty=False):
        ''' Add edges according to the processing order of FW+BW ops 
            and construct a new graph running on GPU, which we call gpu_dag.
        '''
        self.gen_dag_with_rank_weight()
        self.gpu_dag = nx.DiGraph()

        critical_path = None
        max_para_degree = self._add_new_edges_via_order(_pretty)

        #! Then, read IO, Comm, OUTPUT, and STEP nodes
        for u, v in self.dag.edges:
            if self.is_computation_node(u) and self.is_computation_node(v):
                #! ignore edges whose u v are both computation nodes
                continue
            self.gpu_dag.add_edge(u, v, weight=self.dag.edges[(u, v)]["weight"])

        #！til now, all the edges for one GPU have been added.
        if not _pretty:
            critical_path = dag_longest_path(self.gpu_dag, self.local_rank, self.logger, weight="weight", default_weight=0)

        return max_para_degree, critical_path

    def all_topo_sorts(self):
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
        self.gen_dag_with_rank_weight()
        for u, v, _dict in self.dag.edges.data():
            if self.is_computation_node(u) and self.is_computation_node(v): 
                self._fw_bw_dag.add_edge(u, v, **_dict)
        for n, _dict in self._fw_bw_dag.nodes.data():
            _dict["in_degree"] = self._fw_bw_dag.in_degree(n)
            _dict["visited"] = False

        self.logger.info(list(self._fw_bw_dag.nodes))
        # self.all_topo_sorts()
        self.logger.info(len(self.topo_sorts))





