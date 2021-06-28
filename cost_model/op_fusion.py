import os, sys
import networkx as nx
from scipy.stats.mstats import gmean
import numpy as np
import pickle

from tqdm import tqdm, trange

import arg_utils
from trace_utils import parse_cat_from_name, parse_pid_from_name, \
    CatName, parse_cat_fine_grained, _parse_tf_layer_names, \
    SingleLogger, GAP_STR_OP2OP, GAP_STR_OP2COMM, FileName
from base import bcolors

from cost_model.base import _BaseGraphPass, OptQueryCostModelError
from cost_model._xla.utils import parse_xla_candidate_ops, IGNORE_OP_TYPES
from cost_model._xla.pk_graph import PKGraph, contract_nodes_nx, \
    defuse_nodes_inplace_nx, postorder_contract_nx, \
    subgraph_partition_connected_nx_using_topo, get_concated_names
from cost_model._xla.xla_module_cost_model import XLAModuleCostModel

args_ = arg_utils.SingleArg().args
XLA_INIT_NO_BW = True
FUSION_TIME_ESTIMATE_RATIO = 0.8
FORCE_ESTIMATE_FUSION_TIME = False
ENABLE_PRUNING = False

class AttrCache():
    def __init__(self):
        self._cache = {}

    def _parse_key(self, node_name):
        if "+" in node_name:
            key = tuple(sorted(node_name.split("+")))
        else:
            key = node_name
        return key

    def __contains__(self, node_name):
        key = self._parse_key(node_name)
        return key in self._cache

    def __getitem__(self, node_name):
        key = self._parse_key(node_name)
        return self._cache[key]

    def __setitem__(self, node_name, value):
        key = self._parse_key(node_name)
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)


class XLAGraphPass(_BaseGraphPass):
    def __init__(self, opt, root_path):
        super().__init__(opt)
        self.root_path = root_path

        if not args_.simulate:
            self.cost_models = self._load_cm()
        self.forbidden_list = set()
        self.initial_forbidden_list = set()
        # self._init_forbidden_list()
        self.token = ["+", "-"]

        ### Need to cache
        self.ckpt_path = os.path.join(self.root_path, "xla_ckpt.pickle")
        ### Used to cache the node attribtue
        self.node_attr_cache = AttrCache()

        self.expore_fusion = True
        self.enable_partition = True

    def flush(self, is_accept):
        pass

    def load_init_ckpt(self, G_prime=None):
        ''' Other cost model may initialize the DFG, init DFG based on that
        '''
        init_ckpt_path = os.path.join(self.root_path, "xla_init_ckpt.pickle")
        trajectory = []
        if os.path.isfile(init_ckpt_path):
            with open(init_ckpt_path, "rb") as f:
                G, PKG, node_attr_cache, initial_partitions_formed, self.forbidden_list = pickle.load(
                    f)
                self.node_attr_cache = node_attr_cache
            SingleLogger().info("Reading init graph from cache.")
        else:
            G = self.dag.copy() if G_prime is None else G_prime.copy()
            PKG = PKGraph(G)
            self._init_forbidden_list(G_prime=G)
            # # randomly contract edges if possible
            # k = int(len(G.edges()) * init_edges_to_contract)
            for node in G.nodes():
                if node not in self.node_attr_cache:
                    self._cache_node_attr(node, G.nodes[node])

            if args_.layer_num_limit:
                self._sample_strategies(
                    G, PKG, layer_num_limit=args_.layer_num_limit)
                exit(0)

            G, initial_partitions_formed = self._init_partition(G, PKG)
            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, self.node_attr_cache, initial_partitions_formed,
                             self.forbidden_list], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))

        if "BPF_DUMP_INIT_CLUSTER_TO" in os.environ:
            self._dump_cluster_mapping(G, os.environ["BPF_DUMP_INIT_CLUSTER_TO"], partition=True)
        SingleLogger().info("Successfully initialized {} partitions.".format(initial_partitions_formed))

        # self._check_dag_avg(G)

        return G, PKG, trajectory

    def load_ckpt(self):
        if os.path.isfile(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                node_attr_cache = pickle.load(f)
                self.node_attr_cache = node_attr_cache

    def checkpoint(self):
        with open(self.ckpt_path, "wb") as f:
            pickle.dump(self.node_attr_cache, f)

    def _load_cm(self):
        cost_models = {}
        models_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "_xla/.cost_model")

        cost_model_tmp_dir = os.path.join(self.root_path, "cost_model_tmp")
        if not os.path.exists(cost_model_tmp_dir):
            os.makedirs(cost_model_tmp_dir)
        SingleLogger().info("Searching for XLA Cost Model dumps in {}".format(models_dir))
        cost_models["default"] = XLAModuleCostModel(models_dir, tmp_dir=os.path.join(cost_model_tmp_dir))
        return cost_models

    def _reduce_nx_size(self, G):
        ret_G = nx.DiGraph()
        edges_to_add = []
        for u, v in G.edges:
            if u == v:
                continue
            if (u.startswith("host0.rank0") and v.startswith("host0.rank0")) \
                or (u.startswith("traces_0.rank0") and v.startswith("traces_0.rank0")):
                # we also remove FW -> UPDATE egdes here since now we have 
                # removed communication nodes, postorder_contract will try to
                # fuse UPDATE with FW
                if not (("FW" in u or "BW" in u) and "UPDATE" in v):
                    edges_to_add.append((u, v))
        ret_G.add_edges_from(edges_to_add)
        return ret_G

    def _sample_strategies(self, G, PKG, layer_num_limit):
        ''' Sample some operator fusion strategies by fusing operators layer by layer
            and generate cluster mapping files
        '''
        SingleLogger().info("Start to sample strategies, ... ")
        if layer_num_limit.isdigit():
            layer_num_limit = int(layer_num_limit)
            layer_num_list = layer_num_limit if layer_num_limit >= 0 else [1, 3, 5, 10, 20]
        else:
            layer_num_list = [int(n) for n in layer_num_limit.split(",")]

        first = True
        for _layer_num_limit in layer_num_list:
            partition_G = self._reduce_nx_size(G)
            partition_PKG = PKGraph(partition_G)
            source_nodes = sorted(
                [node for node in partition_G.nodes if node not in self.forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))
            # print(source_nodes)

            if first:
                layer_set = set()
                for n in partition_G.nodes():
                    if "BW" not in n:
                        continue
                    layer_name = _parse_tf_layer_names(n)
                    layer_set.add(layer_name[0])
                SingleLogger().info(bcolors.CYELLOWBG + "There are {} BW layers in total".format(len(layer_set)) + bcolors.ENDC)
                first = False

            # Run post order traversal on partition_G
            visited_nodes = set()
            for source in tqdm(source_nodes, total=len(source_nodes)):
                if source not in visited_nodes and source in partition_G.nodes:
                    _, _, partition_G = postorder_contract_nx(
                        partition_G, partition_PKG, source, visited_nodes,
                        forbidden_list = self.forbidden_list,
                        layer_num_limit = _layer_num_limit
                    )
            self._dump_cluster_mapping(partition_G, os.path.join(
                self.root_path, "cluster_mapping_layer_num_limit_{}.txt".format(_layer_num_limit)))
            
            G_copy = G.copy()
            PKG_copy = PKG.copy()
            G_copy, _ = self._create_clusters(G_copy, PKG_copy, partition_G)
            cost, _, _ = self.opt.evaluate(G_copy)
            SingleLogger().info(bcolors.CYELLOWBG +
                                "_layer_num_limit: {} ==> cost: {}".format(_layer_num_limit, cost) + bcolors.ENDC)

        SingleLogger().info("Done. Strategies are stored at {}".format(self.root_path))

    def _init_partition(self, G, PKG):
        ''' Initialize the graph with a default operator fusion strategy
            By default, this function tries to fuse all operators and avoids cycles
        '''
        # partition_G = G.copy()

        partition_G = self._reduce_nx_size(G)
        partition_PKG = PKGraph(partition_G)

        if args_.layer_by_layer:
            source_nodes = sorted(
                [node for node in partition_G.nodes if node not in self.forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))
        else:
            source_nodes = sorted(
                [node for node in partition_G.nodes if node not in self.initial_forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))

        # Run post order traversal on partition_G
        visited_nodes = set()
        SingleLogger().info("Start to postorder_contract_nx ... ")
        for source in tqdm(source_nodes, total=len(source_nodes)):
            if source not in visited_nodes and source in partition_G.nodes:
                if args_.layer_by_layer:
                    _, _, partition_G = postorder_contract_nx(
                        partition_G, partition_PKG, source, visited_nodes,
                        forbidden_list=self.forbidden_list,
                        layer_num_limit=1)
                else:
                    _, _, partition_G = postorder_contract_nx(
                        partition_G, partition_PKG, source, visited_nodes,
                        forbidden_list=self.initial_forbidden_list)

        self._dump_cluster_mapping(partition_G, os.path.join(
                self.root_path, "cluster_mapping_after_initialization.txt"))

        SingleLogger().info("Start to init partition graph ... ")
        return self._create_clusters(G, PKG, partition_G)

    def _create_clusters(self, G, PKG, partition_G):
        ''' Init partition on `G` based on the the pre-partitioned DFG `partition_G`
            **NOTE**: this modification is in place
        '''
        initial_partitions_formed = 0
        for node_name in tqdm(partition_G.nodes()):
            if node_name in self.initial_forbidden_list or "Comm" in node_name:
                continue
            
            if "+" in node_name:
                # fused node, test if compilable
                # print("hhp: {}".format(node_name))
                try:
                    avg = self._get_node_avg(node_name, verbose=False)
                    self._parse_node_attr(partition_G, node_name, avg)
                    compilable=True
                except (OptQueryCostModelError, ValueError):
                    compilable=False
                if compilable:
                    for _node_name in [node_name] + self.opt._debug_convert_to_other_machines(node_name):
                        ns = _node_name.split("+")
                        new_node_name = contract_nodes_nx(G, ns)
                        PKG.contract_nodes_unsafe(ns)
                        ### TODO (huhanpeng): since we assume data parallel
                        ### we directly use the fused time of the first GPU for all GPUs' fused nodes
                        self._parse_node_attr(G, new_node_name, avg) # type: ignore
                        initial_partitions_formed += 1
        return G, initial_partitions_formed

    def _init_forbidden_list(self, G_prime=None):
        ''' Operators in the forbidden list will not be contacted with other operators
        `self.forbidden_list` is used through the search process
        `self.initial_forbidden_list` is only used when initalizing the fusion pattern.
        '''
        xla_candidates, _ = parse_xla_candidate_ops(args_.xla_candidate_path)
        # limit the range of nodes during search
        filtered_xla_candidates = set()
        for op in xla_candidates:
            should_ignore = False
            for ignore_type in IGNORE_OP_TYPES:
                if ignore_type in op:
                    should_ignore = True
                    break
            if not should_ignore:
                filtered_xla_candidates.add(op)
        

        dag = self.dag if G_prime is None else G_prime
        for node in dag.nodes:
            # ignore BW nodes and communication nodes
            if XLA_INIT_NO_BW and "BW" in node:
                self.initial_forbidden_list.add(node)

            try:
                op_name, pid = self.opt._get_original_name_pid_from_index(node)
            except:
                # not standard nodes, ignore
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                continue
            cat = parse_cat_from_name(node)
            if (not args_.simulate and op_name not in self._wrap_xla_operation_names(pid)) \
                    or "Assign" in op_name or cat == CatName.COMM.value \
                    or op_name not in filtered_xla_candidates:
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                
    def _get_node_attr(self, n, attr_):
        if attr_ in self.node_attr_cache[n]:
            return self.node_attr_cache[n][attr_]
        else:
            return 0

    def _cache_node_attr(self, n, attrs):
        ### TODO (huhanpeng): need .copy() ???
        self.node_attr_cache[n] = attrs

    def _dump_cluster_mapping(self, dag, output_path, partition=False):
        ''' Dump cluster mapping in `dag` at `output_path`
            If `partition` is set True, the dag will be partitioned
        '''
        if partition:
            dag = self._reduce_nx_size(dag)
        cluster_index = 0
        with open(output_path, "w") as f:
            for node in dag.nodes():
                if "+" in node and "Comm" not in node:
                    op_names, _ = self._get_original_name_pid_from_fused_node(node)
                    for orig_node_name in op_names:
                        f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1

    def _get_original_name_pid_from_fused_node(self, u_):
        single_pid = None
        op_names = []
        for node_name in self._get_defused_node_names(u_):
            op_name, pid = self.opt._get_original_name_pid_from_index(node_name)
            op_names.append(op_name)
            if single_pid is None:
                single_pid = pid
            else:
                if single_pid != pid:
                    raise RuntimeError(
                        "Fused DAG node {} contains ops from different machines.".format(u_))
        return op_names, single_pid

    def _get_defused_node_names(self, fused_node_):
        return fused_node_.split("+")

    def _wrap_xla_predict(self, pid, nodes_to_fuse, fused_u_, simulate=False):
        ''' 
        nodes_to_fuse: a list of layer names to fuse
        fused_u_: a str of fused names with layer index
        '''
        if FORCE_ESTIMATE_FUSION_TIME or simulate:
            _sum = 0
            origin_nodes = self._get_defused_node_names(fused_u_)
            for idx in range(len(origin_nodes) - 1):
                _sum += self.node_attr_cache[origin_nodes[idx]]["avg"]
            return _sum * FUSION_TIME_ESTIMATE_RATIO + self.node_attr_cache[origin_nodes[-1]]["avg"]
        else:
            # return self.cost_models[pid].predict(nodes_to_fuse)
            predicted_time, brkdn_dict = self.cost_models["default"].predict(nodes_to_fuse)
            return predicted_time / 1000

    def _wrap_xla_need_fuse(self, long_name):
        try:
            orig_name, pid = self.opt._get_original_name_pid_from_index(long_name)
        except (IndexError, KeyError):
            return False
        
        ### TODO (huhanpeng)
        # if orig_name.endswith("_Switch"):
        #     return True

        if args_.simulate:
            return long_name not in self.forbidden_list
        else:
            # if "_Switch" in orig_name and orig_name not in self._wrap_xla_operation_names(pid):
            #     print("{} not in xla_operation_names".format(orig_name))
            # if "_Switch" in orig_name and long_name in self.forbidden_list:
            #     print("{} in the forbidden list".format(orig_name))
            return (orig_name in self._wrap_xla_operation_names(pid)) and long_name not in self.forbidden_list

    def _wrap_xla_operation_names(self, pid):
        return self.cost_models["default"].graph_def_util.operation_names

    def _query_cost_model(self, fused_u_, verbose=True):
        assert "+" in fused_u_
        # query cost model for exec time of a fused node u
        nodes_in_u, u_pid = self._get_original_name_pid_from_fused_node(fused_u_)
        nodes_to_fuse = set(nodes_in_u)
        if verbose:
            if len(nodes_to_fuse) < 10:
                SingleLogger().info("[COST MODEL QUERY] {} Nodes to fuse: {}".format(
                    len(nodes_to_fuse), nodes_to_fuse))
            else:
                SingleLogger().info(
                    "[COST MODEL QUERY] {} Nodes to fuse ...".format(len(nodes_to_fuse)))

        predicted_time = self._wrap_xla_predict(u_pid, nodes_to_fuse, fused_u_, simulate=args_.simulate)
        if predicted_time < 0:
            if args_.disable_estimate:
                raise OptQueryCostModelError("Failed to query cost model.")
            else:
                predicted_time = self._wrap_xla_predict(
                    u_pid, nodes_to_fuse, fused_u_, simulate=True)
                if verbose:
                    SingleLogger().warn(
                        "[COST MODEL QUERY] Exec time {}ESTIMATED{}: {}".format(bcolors.CYELLOWBG, bcolors.ENDC, predicted_time))
        else:
            if verbose:
                SingleLogger().info("[COST MODEL QUERY] Exec time predicted: {} (Avg. sum of origin: {}".format(
                    predicted_time, sum([self._get_node_avg(n) for n in fused_u_.split("+")])))
            # self.cost_model_error.append(np.abs(predicted_time - executed_time) / executed_time)
            # SingleLogger().info("[COST MODEL QUERY] Average prediction accuracy: {}".format(np.average(self.cost_model_error)))
            # if len(self.cost_model_error) > 20:
            # self.cost_model_error = []
            pass
        return predicted_time
        
    def _wrap_can_fuse_to_b(self, _pkg: PKGraph, a, b):
        ''' Return whether a can be fused with b
        '''
        if "Comm" in b or not _pkg.can_contract_edge(a, b):
            return False

        if "+" not in b and not self._wrap_xla_need_fuse(b):
            # if 'BW' in succ_:
            #     print("Ignore succ {}".format(succ_))
            return False
        
        if parse_pid_from_name(a) != parse_pid_from_name(b) or parse_cat_from_name(a) != parse_cat_from_name(b):
            return False

        return True

    def init_search_space_by_nodes(self, preds, node, _pkg):
        search_space = []
        if not self._wrap_xla_need_fuse(node):
            return []
        
        for pred in preds:
            if self._wrap_xla_need_fuse(pred) and self._wrap_can_fuse_to_b(_pkg, pred, node):
                search_space.append(("+", pred, node))

        return search_space

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        ### TODO (huhanpeng): currently only consider fusion
        ###             Need to add quantization
        search_space = []
        weights = []
        prun_cnt = 0
        fusion_cnt = 0
        defusion_cnt = 0

        # TODO (huhanpeng): below code forces to search without the limitation of critical paths.
        if args_.no_crit:
            candidates = [(n, _dag.nodes[n]["avg"]) for n in _dag.nodes() if "BW" in n and "host0.rank0" in n]

        for n, l in candidates:
            # node heat
            heat = self.opt._get_heat_from_history(n)
            if "Comm" in n:
                continue

            if "+" in n and self.enable_partition:
                ### This a fused node
                if args_.layer_by_layer and "FW" in n and "BW" not in n:
                    continue
                ns = n.split("+")
                cat = parse_cat_fine_grained(ns[0])
                pid = parse_pid_from_name(ns[0])
                ns = set(ns)
                subgraph = self.dag.subgraph(ns)

                # randomly split edges using spanning tree
                ### a list of tupe(nodes_in_a, nodes_in_b)
                valid_split_plans = subgraph_partition_connected_nx_using_topo(
                    subgraph, layer_by_layer=args_.layer_by_layer)
                split_weights = []
                for splits in valid_split_plans:
                    split_weights.append(gmean([len(nodes) for nodes in splits]))
                split_weights = np.exp(5e-4*(len(ns) - 80)) * (np.array(split_weights) / np.sum(split_weights))
                for split_index, splits in enumerate(valid_split_plans):
                    search_space.append(("-", n, splits))
                    defusion_cnt += 1
                    weights.append(self.opt._combine_weight(l, 1 / (heat + 1) - 1) * split_weights[split_index])
            else:
                ### Nodes that have never been fused
                cat = parse_cat_fine_grained(n)
                pid = parse_pid_from_name(n)
            
            ### TODO (huhanpeng): !!! NOTE: should modify below code back
            if args_.fusion_once and not self._wrap_xla_need_fuse(n):
                continue
            elif not args_.fusion_once and "+" not in n and not self._wrap_xla_need_fuse(n):
                # if 'BW' in n:
                #     print("Ignore {}".format(n))
                continue

            for succ_ in _dag.successors(n):
                if not self._wrap_can_fuse_to_b(_pkg, n, succ_):
                    continue

                ### Assumption: for edge bw_u->bw_v, if comm_bw_u > bw_v, it can not bring any speedup if fusing u and v.
                def ret_comm_time(_node):
                    __ret = _dag.nodes[_node]["avg"]
                    for __succ in _dag.successors(_node):
                        _pid = parse_pid_from_name(__succ)
                        if "Comm" in __succ and pid == _pid:
                            __ret += ret_comm_time(__succ)
                    return __ret
                # if comm_t >= _dag.nodes[bw_v]["avg"]:
                if ENABLE_PRUNING:
                    comm_t = 0
                    effective_succ_bw = set()
                    # for bw_u_succ in _dag.successors(bw_u):
                    for bw_u_succ in _dag.successors(n):
                        if "Comm" in bw_u_succ:
                            if self.opt.comm_backend == "NCCL":
                                # check if the comm node's predecessor includes succ_
                                if succ_ in _dag.predecessors(bw_u_succ):
                                    # OK to fuse
                                    continue
                                else:
                                    succ_bws = [node for node in _dag.predecessors(bw_u_succ) if "BW" in node]
                                    effective_succ_bw.union(set(succ_bws))
                                    comm_t += ret_comm_time(bw_u_succ)
                            else:
                                ### TODO (huhanpeng): is there only one comm sub operator ???
                                comm_t += _dag.nodes[bw_u_succ]["avg"]
                                effective_succ_bw.add(succ_)
                    effective_bw_size = 0
                    for node in effective_succ_bw:
                        effective_bw_size += _dag.nodes[node]["avg"]

                    if comm_t > effective_bw_size:
                        prun_cnt += 1
                        SingleLogger().debug("Prune fusing {} and {} with comm time {}".format(n, succ_, comm_t))
                        continue

                # calculate heat using max(heat(n), heat(succ_))
                heat_succ = self.opt._get_heat_from_history(succ_)
                heat_combined = (heat + heat_succ) / 2

                search_space.append(("+", n, succ_))
                fusion_cnt += 1
                weights.append(self.opt._combine_weight(l, heat_combined))
                # weights.append(1)
        
        SingleLogger().info("Init search space from {} candidates, prune {}: {} fusion strategies, {} defusion strategies".format(
            len(candidates), prun_cnt if ENABLE_PRUNING else "(disabled)", fusion_cnt, defusion_cnt))

        if len(search_space) > 0:
            weights /= np.linalg.norm(weights)
            weights = list(weights)

        ### TODO (huhanpeng) delete
        bw_num_in_critical_path = len([n for n, _ in candidates if "BW" in n])
        bw_num_in_g = len([n for n in _dag.nodes() if "BW" in n and "host0.rank0" in n])
        SingleLogger().info(bcolors.CSELECTED +
                            "{}/{} BW nodes in the critical path".format(bw_num_in_critical_path, bw_num_in_g) + bcolors.ENDC)

        return search_space, weights

    def _concat_name(self, u_, v_):
        return "%s+%s" % (u_, v_)
        
    def _combine_gap(self, ug, vg):
        ### TODO (huhanpeng): key component
        ### Use max to avoid one input is zero,
        ### some how for the new gap x, ug < x < ug + vg, vg < x < ug + vg
        # return max(max((ug + vg) / 0.8, ug), vg)
        return max(ug, vg)

    def _combine_nodes_attr(self, _dag, target, u_, v_, avg):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
        _dag.nodes[target]["avg"] = avg
        _dag.nodes[target][GAP_STR_OP2OP] = self._combine_gap(self._get_node_attr(u_, GAP_STR_OP2OP), self._get_node_attr(v_, GAP_STR_OP2OP))
        _dag.nodes[target][GAP_STR_OP2COMM] = self._combine_gap(self._get_node_attr(u_, GAP_STR_OP2COMM), self._get_node_attr(v_, GAP_STR_OP2COMM))

    def _combine_attr_except_avg(self, target, attr1, attr2):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target

        if GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(attr1[GAP_STR_OP2OP], attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP not in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(0, attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP not in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(attr1[GAP_STR_OP2OP], 0)

        if GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(attr1[GAP_STR_OP2COMM], attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM not in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(0, attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM not in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(attr1[GAP_STR_OP2COMM], 0)

    def _get_node_avg(self, new_name, verbose=True):
        if new_name in self.node_attr_cache:
            return self.node_attr_cache[new_name]["avg"]
        else:
            return self._query_cost_model(new_name, verbose=verbose)

    def _parse_node_attr(self, _dag, new_name, avg):
        ''' Parse the fused node attribute corresponding to `new_name` and set _dag
        * If new_name has been cached, directly set _dag with the cached attributes
        * Otherwise, combine the attribution of all original nodes
            * If avg is not given, query the cost model
            * Otherwize, use the given avg (TODO Disabled, since we have cached attr in self.node_attr_cache)
        
        Return
        ------
        avg: average time
        '''
        if new_name in self.node_attr_cache:
            nx.set_node_attributes(_dag, {new_name: self.node_attr_cache[new_name]})
            # _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            ns = new_name.split("+")
            attrs = self.node_attr_cache[ns[0]].copy()
            for idx in range(1, len(ns)):
                self._combine_attr_except_avg(attrs, attrs, self.node_attr_cache[ns[idx]])
            # combine attr avg
            attrs["avg"] = avg
            ### set and cache the attribute
            nx.set_node_attributes(_dag, {new_name: attrs})
            self._cache_node_attr(new_name, _dag.nodes[new_name])

        ### TODO (huhanpeng): apply to other GPUs, cache the same attribute for corresponding operators on other GPUs
        for other_name in self.opt._debug_convert_to_other_machines(new_name):
            if other_name in self.node_attr_cache:
                continue
            self._cache_node_attr(other_name, _dag.nodes[new_name])

        return self.node_attr_cache[new_name]["avg"]

    def _op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # test if two nodes can be fused
        if _pkg.can_contract_edge(u_, v_):
            nodes_to_add = []
            nodes_to_remove = []
            MAX_FUSION_EXLPORE_DEPTH = 20
            for explore_idx in range(MAX_FUSION_EXLPORE_DEPTH):
                # pkg must contract after calling _fuse_pair since it can
                # throw errors
                SingleLogger().info(bcolors.CBLUE + "The {} th exploration...".format(explore_idx) + bcolors.ENDC)
                avg = self._fuse_pair(_dag, u_, v_)
                _pkg.contract_edge(u_, v_)

                nodes_to_add.append(u_+"+"+v_)
                nodes_to_remove += [u_, v_]

                ### apply the same strategy to other GPUs
                ul = self.opt._debug_convert_to_other_machines(u_)
                vl = self.opt._debug_convert_to_other_machines(v_)
                for u__, v__ in zip(ul, vl):
                    assert _pkg.can_contract_edge(u__, v__)
                    ### TODO (huhanpeng): since we assume data parallel
                    ### use the same avg for the fused operators
                    self._fuse_pair(_dag, u__, v__, avg=avg)
                    _pkg.contract_edge(u__, v__)
                    nodes_to_add.append(u__+"+"+v__)
                    nodes_to_remove += [u__, v__]
                
                if not self.expore_fusion:
                    break

                is_end = True
                if avg > self._get_node_avg(u_) + self._get_node_avg(v_):
                    succs = [s for s in _dag.successors(
                        u_+"+"+v_) if (self._wrap_can_fuse_to_b(_pkg, u_+"+"+v_, s))] 
                    if len(succs) > 0:
                        heats = [self.opt._combine_weight(None, self.opt._get_heat_from_history(s)) for s in succs]
                        u_ = u_+"+"+v_
                        st_idx = self.opt.select_one_stategy(heats, range(len(succs)))
                        v_ = succs[st_idx]
                        is_end = False
                if is_end:
                    break
            nodes_to_add = set(nodes_to_add)
            nodes_to_remove = set(nodes_to_remove)
            return True, nodes_to_add.difference(nodes_to_remove), nodes_to_remove.difference(nodes_to_add)
        else:
            return False, None, None

    def _fuse_pair(self, _dag, u_, v_, avg=None):
        # print("fuse {} {}".format(u_, v_))
        ### Cache the node attributes in case they will be used when de-fuse
        if u_ not in self.node_attr_cache:
            self._cache_node_attr(u_, _dag.nodes[u_])
        if v_ not in self.node_attr_cache:
            self._cache_node_attr(v_, _dag.nodes[v_])

        SingleLogger().debug("Fuse {} ({}) and {} ({})".format(u_, self.node_attr_cache[u_]["avg"],
            v_, self.node_attr_cache[v_]["avg"]))

        new_name = self._concat_name(u_, v_)
        ### Add new nodes and get the attibute
        if new_name in self.node_attr_cache:
            _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            ### Calculate the new attribute
            if avg is None:
                # an error is thrown here if cannot combine
                # we must put all modification of dag after this line
                avg = self._get_node_avg(new_name, verbose=False)
            _dag.add_node(new_name)
            self._combine_nodes_attr(_dag, new_name, u_, v_, avg)
            ### cache the attribute
            self._cache_node_attr(new_name, _dag.nodes[new_name])

        ### Update edges
        for in_, _ in _dag.in_edges(u_):
            if in_ != v_:
                _dag.add_edge(in_, new_name)
        for in_, _ in _dag.in_edges(v_):
            if in_ != u_:
                _dag.add_edge(in_, new_name)

        for out_ in _dag.successors(u_):
            if out_ != v_:
                _dag.add_edge(new_name, out_)
        for out_ in _dag.successors(v_):
            if out_ != u_:
                _dag.add_edge(new_name, out_)

        ### Remove current nodes
        _dag.remove_node(u_)
        _dag.remove_node(v_)

        assert u_ not in _dag.nodes
        assert v_ not in _dag.nodes
        assert u_ in self.node_attr_cache and "avg" in self.node_attr_cache[u_]
        assert v_ in self.node_attr_cache and "avg" in self.node_attr_cache[v_]
        return self.node_attr_cache[new_name]["avg"]

    def _op_defusion(self, _dag, _pkg: PKGraph, target, components):
        nodes2add = []
        nodes2rm = []

        _, new_node_names = self._defuse_node(_dag, _pkg, target, components)
        nodes2add += new_node_names
        nodes2rm.append(target)

        ### apply the same strategy to other GPUs
        target_l = self.opt._debug_convert_to_other_machines(target)
        components_l = [tuple([self.opt._debug_convert_to_other_machines(node) for node in comp]) for comp in components]
        for idx, target_ in enumerate(target_l):
            components_ = [tuple([node_l[idx] for node_l in comp]) for comp in components_l]
            _, new_node_names_ = self._defuse_node(_dag, _pkg, target_, components_)
            nodes2add += new_node_names_
            nodes2rm.append(target_)

        return True, set(nodes2add), set(nodes2rm)

    def _defuse_node(self, _dag, _pkg, target, components):
        avgs = []
        component_names = get_concated_names(components)
        for new_node_name in component_names:
            avg = self._get_node_avg(new_node_name, verbose=False)
            avgs.append(avg)
        _pkg.split_node(target, components)

        # override successors for BW nodes if searching along with tensor fusion
        if "++" in self.opt.cst_md_mng.strategy2model:
            def succ_overide_func(_node):
                if "BW" not in _node:
                    return None
                else:
                    assert "+" not in _node
                    return self.opt.cst_md_mng.strategy2model["++"].get_current_comm_from_unfused_bw(_node)
            def pred_override_func(_node):
                if "UPDATE" not in _node:
                    return None
                else:
                    assert "+" not in _node
                    return self.opt.cst_md_mng.strategy2model["++"].get_current_comm_from_unfused_update(_node)
        else:
            succ_overide_func = None
            pred_override_func = None

        defuse_nodes_inplace_nx(_dag, _pkg, target, components, 
                                succ_override=succ_overide_func,
                                pred_override=pred_override_func)
        for idx, new_node_name in enumerate(component_names):
            self._parse_node_attr(_dag, new_node_name, avgs[idx])
        return True, component_names

    def apply(self, s, __dag, __pkg):
        op, target, next_ = s
        ### TODO (huhanpeng): need further add other optimization techiniques
        if op == "+":
            ### Fuse two nodes
            return self._op_fusion(__dag, __pkg, target, next_)
        elif op == "-":
            return self._op_defusion(__dag, __pkg, target, next_)

    def _check_dag_avg(self, G: nx.DiGraph):
        for n in G.nodes():
            if "Comm" in n or "host0.rank0" not in n:
                continue
            if "+" not in n:
                continue
            fused_avg = self.node_attr_cache[n]["avg"]
            avg_sum = 0
            all_fuse_nodes = n.split("+")
            for _n in all_fuse_nodes:
                avg_sum += self.node_attr_cache[_n]["avg"]
            print("Fuse {} nodes, predicted avg: {}, fused nodes avg sum: {}".format(
                len(all_fuse_nodes), fused_avg, avg_sum))
        raise RuntimeError()
