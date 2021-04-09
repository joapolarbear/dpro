import os, sys
import networkx as nx
from scipy.stats.mstats import gmean
import numpy as np
import pickle

from tqdm import tqdm, trange

import arg_utils
from trace_utils import parse_cat_from_name, parse_pid_from_name, \
    CatName, parse_cat_fine_grained, SingleLogger

from cost_model.base import _BaseGraphPass
from cost_model._xla.gen_dataset_utils import parse_xla_candidate_ops
from cost_model._xla.pk_graph import PKGraph, contract_nodes_nx, \
    defuse_nodes_inplace_nx, postorder_contract_nx, \
    subgraph_partition_connected_nx_using_topo, get_concated_names
from cost_model._xla.xla_module_cost_model import XLAModuleCostModel

args_ = arg_utils.SingleArg().args
XLA_INIT_NO_BW = True
ROOT_PATH = os.path.join(
    args_.workspace if args_.workspace else args_.path, ".opt_ws")

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
    def __init__(self, opt):
        super().__init__(opt)
        if not args_.simulate:
            self.cost_models = self._load_cm()
        self.forbidden_list = set()
        self.initial_forbidden_list = set()
        # self._init_forbidden_list()
        self.token = ["+", "-"]

        ### Need to cache
        self.ckpt_path = os.path.join(ROOT_PATH, "xla_ckpt.pickle")
        ### Used to cache the node attribtue
        self.node_attr_cache = AttrCache()

    def flush(self, is_accept):
        pass

    def load_init_ckpt(self, G_prime=None):
        ''' Other cost model may initialize the DFG, init DFG based on that
        '''
        init_ckpt_path = os.path.join(ROOT_PATH, "xla_init_ckpt.pickle")
        trajectory = []
        if os.path.isfile(init_ckpt_path):
            with open(init_ckpt_path, "rb") as f:
                G, PKG, node_attr_cache, initial_partitions_formed = pickle.load(f)
                self.node_attr_cache = node_attr_cache
            SingleLogger().info("Reading init graph from cache.")
        else:
            G = self.dag.copy() if G_prime is None else G_prime.copy()
            PKG = PKGraph(G)
            self._init_forbidden_list(G_prime=G)
            # # randomly contract edges if possible
            # k = int(len(G.edges()) * init_edges_to_contract)
            initial_partitions_formed = 0
            for node in G.nodes():
                if node not in self.node_attr_cache:
                    self._cache_node_attr(node, G.nodes[node])

            if args_.layer_num_limit:
                self._sample_strategies(
                    G, layer_num_limit=args_.layer_num_limit)
                exit(0)

            G, initial_partitions_formed = self._init_partition(G, PKG, initial_partitions_formed)
            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, self.node_attr_cache, initial_partitions_formed], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))

        if "BPF_DUMP_INIT_CLUSTER_TO" in os.environ:
            self._dump_cluster_mapping(G, os.environ["BPF_DUMP_INIT_CLUSTER_TO"])
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
            __file__)), "cost_model/_xla/.cost_model")

        cost_model_tmp_dir = os.path.join(ROOT_PATH, "cost_model_tmp")
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

    def _sample_strategies(self, G, layer_num_limit):
        ''' Sample some operator fusion strategies by fusing operators layer by layer
            and generate cluster mapping files
        '''
        SingleLogger().info("Start to sample strategies, ... ")
        layer_num_list = layer_num_limit if layer_num_limit >= 0 else [1, 3, 5, 10, 20]
        for _layer_num_limit in layer_num_list:
            partition_G = self._reduce_nx_size(G)
            partition_PKG = PKGraph(partition_G)
            source_nodes = sorted(
                [node for node in partition_G.nodes if node not in self.forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))
            # print(source_nodes)

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
                ROOT_PATH, "cluster_mapping_layer_num_limit_{}.txt".format(_layer_num_limit)))
        SingleLogger().info("Done. Strategies are stored at {}".format(ROOT_PATH))

    def _init_partition(self, G, PKG, initial_partitions_formed):
        ''' Initialize the graph with a default operator fusion strategy
            By default, this function tries to fuse all operators and avoids cycles
        '''
        # partition_G = G.copy()
        partition_G = self._reduce_nx_size(G)
        partition_PKG = PKGraph(partition_G)

        source_nodes = sorted([node for node in partition_G.nodes if node not in self.initial_forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))

        # Run post order traversal on partition_G
        visited_nodes = set()
        SingleLogger().info("Start to postorder_contract_nx ... ")
        for source in tqdm(source_nodes, total=len(source_nodes)):
            if source not in visited_nodes and source in partition_G.nodes:
                _, _, partition_G = postorder_contract_nx(partition_G, partition_PKG, source, visited_nodes, forbidden_list=self.initial_forbidden_list)

        self._dump_cluster_mapping(partition_G, os.path.join(
                ROOT_PATH, "cluster_mapping_after_initialization.txt"))

        SingleLogger().info("Start to init partition graph ... ")
        for node_name in tqdm(partition_G.nodes()):
            if node_name in self.initial_forbidden_list or "Comm" in node_name:
                continue
            if "+" in node_name:
                # fused node, test if compilable
                try:
                    avg = self._get_node_avg(node_name)
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
        xla_candidates = parse_xla_candidate_ops(xla_candidate_path=args_.xla_candidate_path)
        # limit the range of nodes during search
        IGNORE_OP_TYPES = ["ShapeN", "_Arg", "_Send", "_Recv", "VarIsInitializedOp", "ReadVariableOp", "VarHandleOp",
                    "IsVariableInitialized", "ResourceApplyGradientDescent",
                    "IteratorToStringHandle", "IteratorGetNext", "MakeIterator", "IteratorV2"]
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
                orig_name, pid = self.opt._get_original_name_pid_from_index(node)
            except:
                # not standard nodes, ignore
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                continue
            cat = parse_cat_from_name(node)
            if (not args_.simulate and orig_name not in self._wrap_xla_operation_names(pid)) \
                    or "Assign" in orig_name or cat == CatName.COMM.value \
                    or orig_name not in filtered_xla_candidates:
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

    def _dump_cluster_mapping(self, dag, output_path):
        cluster_index = 0
        with open(output_path, "w") as f:
            for node in dag.nodes():
                if "+" in node and "Comm" not in node:
                    orig_names, _ = self._get_original_name_pid_from_fused_node(node)
                    for orig_node_name in orig_names:
                        f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1

    def _get_original_name_pid_from_fused_node(self, u_):
        single_pid = None
        orig_names = []
        for node_name in self._get_defused_node_names(u_):
            orig_name, pid = self.opt._get_original_name_pid_from_index(node_name)
            orig_names.append(orig_name)
            if single_pid is None:
                single_pid = pid
            else:
                if single_pid != pid:
                    raise RuntimeError(
                        "Fused DAG node {} contains ops from different machines.".format(u_))
        return orig_names, single_pid

    def _get_defused_node_names(self, fused_node_):
        return fused_node_.split("+")

    def _wrap_xla_predict(self, pid, nodes_to_fuse, fused_u_):
        '''
        nodes_to_fuse: a list of layer names to fuse
        fused_u_: a str of fused names with layer index
        '''
        if args_.simulate:
            _sum = 0
            for name_ in self._get_defused_node_names(fused_u_):
                _sum += self.node_attr_cache[name_]["avg"]
            return _sum * 0.8, None
        else:
            # return self.cost_models[pid].predict(nodes_to_fuse)
            predicted_time, brkdn_dict = self.cost_models["default"].predict(nodes_to_fuse)
            return predicted_time / 1000, brkdn_dict

    def _wrap_xla_need_fuse(self, pid, orig_name, long_name):
        if args_.simulate:
            return long_name not in self.forbidden_list
        else:
            return (orig_name in self._wrap_xla_operation_names(pid)) and long_name not in self.forbidden_list

    def _wrap_xla_operation_names(self, pid):
        return self.cost_models["default"].graph_def_util.operation_names

    def _query_cost_model(self, fused_u_):
        # query cost model for exec time of a fused node u
        nodes_in_u, u_pid = self._get_original_name_pid_from_fused_node(fused_u_)
        nodes_to_fuse = set(nodes_in_u)
        if len(nodes_to_fuse) < 10:
            SingleLogger().info("[COST MODEL QUERY] {} Nodes to fuse: {}".format(
                len(nodes_to_fuse), nodes_to_fuse))
        else:
            SingleLogger().info(
                "[COST MODEL QUERY] {} Nodes to fuse ...".format(len(nodes_to_fuse)))

        predicted_time, _ = self._wrap_xla_predict(u_pid, nodes_to_fuse, fused_u_)
        SingleLogger().info(
            "[COST MODEL QUERY] Exec time predicted: {}".format(predicted_time))
        if predicted_time < 0:
            raise OptQueryCostModelError("Failed to query cost model.")
        else:
            # self.cost_model_error.append(np.abs(predicted_time - executed_time) / executed_time)
            # SingleLogger().info("[COST MODEL QUERY] Average prediction accuracy: {}".format(np.average(self.cost_model_error)))
            # if len(self.cost_model_error) > 20:
            # self.cost_model_error = []
            pass
        return predicted_time

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        ### TODO (huhanpeng): currently only consider fusion
        ###             Need to add quantization
        search_space = []
        weights = []
        prun_cnt = 0

        for n, l in candidates:
            # node heat
            heat = self.opt._get_heat_from_history(n)
            if "Comm" in n:
                continue

            if "+" in n:
                ### This a fused node
                ns = n.split("+")
                cat = parse_cat_fine_grained(ns[0])
                pid = parse_pid_from_name(ns[0])
                ns = set(ns)
                subgraph = self.dag.subgraph(ns)

                # randomly split edges using spanning tree
                valid_split_plans = subgraph_partition_connected_nx_using_topo(subgraph)
                split_weights = []
                for splits in valid_split_plans:
                    split_weights.append(gmean([len(nodes) for nodes in splits]))
                split_weights = np.exp(5e-4*(len(ns) - 80)) * (np.array(split_weights) / np.sum(split_weights))
                for split_index, splits in enumerate(valid_split_plans):
                    search_space.append(("-", n, splits))
                    weights.append(self.opt._combine_weight(l, heat) * split_weights[split_index])
            else:
                ### Nodes that have never been fused
                cat = parse_cat_fine_grained(n)
                pid = parse_pid_from_name(n)
            try:
                n_orig_name, n_pid = self.opt._get_original_name_pid_from_index(n)
            except (IndexError, KeyError):
                continue

            if not self._wrap_xla_need_fuse(n_pid, n_orig_name, n):
                continue
            
            candidate_names = [c[0] for c in candidates]

            for succ_ in _dag.successors(n):
                if succ_ not in candidate_names or "Comm" in succ_:
                    continue
                # some filters
                if not _pkg.can_contract_edge(n, succ_):
                    continue
                if "+" not in succ_:
                    try:
                        succ_orig_name, succ_pid = self.opt._get_original_name_pid_from_index(succ_)
                    except (IndexError, KeyError):
                        continue

                    if not self._wrap_xla_need_fuse(succ_pid, succ_orig_name, succ_):
                        continue

                _pid = parse_pid_from_name(succ_)
                _cat = parse_cat_fine_grained(succ_)
                if pid != _pid or cat != _cat:
                    continue

                ### Assumption: for edge bw_u->bw_v, if comm_bw_u > bw_v, it can not bring any speedup if fusing u and v.
                def ret_comm_time(_node):
                    __ret = _dag.nodes[_node]["avg"]
                    for __succ in _dag.successors(_node):
                        _pid = parse_pid_from_name(__succ)
                        if "Comm" in __succ and pid == _pid:
                            __ret += ret_comm_time(__succ)
                    return __ret

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

                # if comm_t >= _dag.nodes[bw_v]["avg"]:
                if comm_t > effective_bw_size:
                    prun_cnt += 1
                    SingleLogger().debug("Prune fusing {} and {} with comm time {}".format(n, succ_, comm_t))
                    continue

                # calculate heat using max(heat(n), heat(succ_))
                heat_succ = self.opt._get_heat_from_history(succ_)

                heat_combined = (heat + heat_succ) / 2

                search_space.append(("+", n, succ_))
                weights.append(self.opt._combine_weight(l, heat_combined))
                # weights.append(1)
        SingleLogger().info("Init search space len={} from {} candidates, prune {}".format(
            len(search_space), len(candidates), prun_cnt))
        # SingleLogger().info("Time spent for spanning tree: {}".format(sum(time_spanning_trees)/ len(time_spanning_trees)))
        # SingleLogger().info("Time spent for source/sink: {}".format(sum(time_st)/ len(time_st)))
        # normalize weights
        min_weight = min(weights)
        max_weight = max(weights)
        for idx in range(len(weights)):
            weights[idx] -= min_weight
            weights[idx] /= max_weight - min_weight
        return search_space, weights

    def _concat_name(self, u_, v_):
        return "%s+%s" % (u_, v_)

    def _combine_avg(self, u, v):
        ### call cost model to obtain the combined time
        fused_name = self._concat_name(u, v)
        return self._get_node_avg(fused_name)

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
        # target["avg"] = self._combine_avg(attr1["avg"], attr2["avg"])

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

    def _get_node_avg(self, new_name):
        if new_name in self.node_attr_cache:
            return self.node_attr_cache[new_name]["avg"]
        else:
            return self._query_cost_model(new_name)

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

            # pkg must contract after calling _fuse_pair since it can
            # throw errors
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

            return True, nodes_to_add, nodes_to_remove
        else:
            return False, None, None

    def _fuse_pair(self, _dag, u_, v_, avg=None):
        # print("fuse {} {}".format(u_, v_))
        ### Cache the node attributes in case they will be used when de-fuse
        # SingleLogger().info("\033[94m Fusing pair: {}, {}\033[0m".format(u_, v_))
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
                avg = self._combine_avg(u_, v_)
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
            avg = self._get_node_avg(new_node_name)
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
