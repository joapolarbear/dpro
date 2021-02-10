from collections import deque
import enum
import networkx as nx
import random
import math
import time
from collections import deque
from scipy.stats.mstats import gmean
import numpy as np
import code
import traceback
import pickle
import ujson as json
from pathlib import Path

from tqdm import tqdm, trange

from replay import Replayer
from trace_utils import *
from dag_utils import *
from cost_model_xla.pk_graph import PKGraph, PKGraphCycleError, contract_nodes_nx, \
                    defuse_nodes_inplace_nx, postorder_contract_nx, \
                    subgraph_partition_connected_nx_using_topo, get_concated_names
from cost_model_xla.gen_dataset_utils import parse_xla_candidate_ops
from memory import MemoryEstimator
from memory.cost_model import MemoryCostModel

from cost_model.base import _BaseCostModel
# from cost_model.mixed_precision import _AMPCostModel
from cost_model.tensor_fusion import _TensorFusionCM

class GraphExpand(Enum):
    NOT = 0
    PARTIAL = 1
    FULLY = 2

args_ = arg_utils.SingleArg().args
if args_.option == "optimize" and args_.sub_option not in ["amp", "tensor_fusion"]:
    from cost_model_xla.xla_module_cost_model import XLAModuleCostModel
    import horovod.tensorflow as hvd

MAX_TREE_DEPTH = 1000
MAX_LOOP = 1000
UCB_GAMMA = args_.ucb_gamma
MCMC_BETA = args_.mcmc_beta
ROOT_PATH = os.path.join(args_.workspace, ".opt_ws")
XLA_INIT_NO_BW = True

class OptApplyStrategyError(Exception):
    pass


class OptNoValidStrategyError(Exception):
    pass


class OptQueryCostModelError(Exception):
    pass


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


class GraphState:
    def __init__(self, depth):
        self.visit_cnt = 1
        self.quality = -1

        self.space = None
        self.childs = None
        self.parent = None
        self.depth = depth

        # Whether the actions have been tranversed, not, partial or fully
        self.state = GraphExpand.NOT

        self.strategy = None
        self.iter_time = None

    def update_expand_state(self):
        if self.childs is None:
            self.state = GraphExpand.NOT
            return
        assert not self.space is None
        if len(self.childs) == len(self.space):
            self.state = GraphExpand.FULLY
        else:
            self.state = GraphExpand.PARTIAL

class _XLACostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.cost_models = self._load_cm()
        self.forbidden_list = set()
        self.initial_forbidden_list = set()
        self._init_forbidden_list()
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
            G = self.dag.copy() if G_prime is None else G_prime
            PKG = PKGraph(G)
            # # randomly contract edges if possible
            # k = int(len(G.edges()) * init_edges_to_contract)
            initial_partitions_formed = 0
            for node in G.nodes():
                if node not in self.node_attr_cache:
                    self._cache_node_attr(node, G.nodes[node])

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
            __file__)), "cost_model_xla/.cost_model")

        cost_model_tmp_dir = os.path.join(ROOT_PATH, "cost_model_tmp")
        if not os.path.exists(cost_model_tmp_dir):
            os.makedirs(cost_model_tmp_dir)
        SingleLogger().info("Searching for XLA Cost Model dumps in {}".format(models_dir))
        cost_models["default"] = XLAModuleCostModel(models_dir, tmp_dir=os.path.join(cost_model_tmp_dir))
        # for model_dump_dir in os.listdir(models_dir):
        #     model_path = os.path.join(models_dir, model_dump_dir)
        #     p = Path(model_path)
        #     if p.is_dir():
        #         node_name = p.name
        #         cm = XLAModuleCostModel(model_path, tmp_dir=os.path.join(cost_model_tmp_dir, node_name))
        #         cost_models[node_name] = cm
        #         SingleLogger().info(" - Added cost model for {}".format(node_name))
        #     else:
        #         SingleLogger().warn(" - {} not a directory.".format(model_path))
        return cost_models

    def _reduce_nx_size(self, G):
        ret_G = nx.DiGraph()
        edges_to_add = []
        for u, v in G.edges:
            if (u.startswith("host0.rank0") and v.startswith("host0.rank0")) \
                or (u.startswith("traces_0.rank0") and v.startswith("traces_0.rank0")):
                # we also remove FW -> UPDATE egdes here since now we have 
                # removed communication nodes, postorder_contract will try to
                # fuse UPDATE with FW
                if not (("FW" in u or "BW" in u) and "UPDATE" in v):
                    edges_to_add.append((u, v))
        ret_G.add_edges_from(edges_to_add)
        return ret_G

    def _init_partition(self, G, PKG, initial_partitions_formed):
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
            if node_name in self.initial_forbidden_list:
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

    def _init_forbidden_list(self):
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

        for node in self.dag.nodes:
            # ignore BW nodes and communication nodes
            if XLA_INIT_NO_BW and "BW" in node:
                self.initial_forbidden_list.add(node)
                print(node)

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
        # time_spanning_trees = []
        # time_st = []
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

                # st = time.time()
                # randomly split edges using spanning tree
                valid_split_plans = subgraph_partition_connected_nx_using_topo(subgraph)
                split_weights = []
                for splits in valid_split_plans:
                    split_weights.append(gmean([len(nodes) for nodes in splits]))
                split_weights = np.exp(5e-4*(len(ns) - 80)) * (np.array(split_weights) / np.sum(split_weights))
                for split_index, splits in enumerate(valid_split_plans):
                    search_space.append(("-", n, splits))
                    weights.append(self.opt._combine_weight(l, heat) * split_weights[split_index])
                    # weights.append(1)
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
                if succ_ not in candidate_names:
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
        defuse_nodes_inplace_nx(_dag, _pkg, target, components)
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


class CostModelManager:
    def __init__(self, opt):
        self.cost_model_list = []
        # if "amp" in args_.sub_option:
        #     self.cost_model_list.append(_AMPCostModel(opt))
        # if "tensor_fusion" in args_.sub_option:
        #     self.cost_model_list.append(_TensorFusionCM(opt))
        if "xla" in args_.sub_option:
            self.cost_model_list.append(_XLACostModel(opt))
        if len(self.cost_model_list) == 0:
            self.cost_model_list = [
                # _XLACostModel(opt),
                # _AMPCostModel(opt),
            ]
        self.cost_model_list = []
        self.mem_model_list = [MemoryCostModel(opt)]
        self.strategy2model = {}

        ### Register Thoughput-oriented Cost Models
        for cm in self.cost_model_list:
            for _tok in cm.token:
                assert _tok not in self.strategy2model
                self.strategy2model[_tok] = cm

        ### Register Memory-oriented Cost Models
        for cm in self.mem_model_list:
            for _tok in cm.token:
                assert _tok not in self.strategy2model
                self.strategy2model[_tok] = cm


class Optimizer:
    def __init__(self, collector):
        self.clct = collector
        self.platform = self.clct.platform
        self.comm_backend = self.clct.comm_backend
        self.memory_estimator = MemoryEstimator(self.platform)

        self.step = 0
        if args_.relabel:
            ### To simplify the DAG representation, relabel dag with indexes
            ### index2name: index to the layer name, i.e., which is substituted by the index
            self.index2name = {}
            ### index2pid: index to the pid which this index belongs to,
            # i.e., the same op in different pid use different index
            self.index2pid = {}
            ### name2index: given the layer name, store the index in each pid
            self.name2index = {}
            ### index2newname: index to new label in DAG
            self.index2newname = {}

            ## Get the dependency graph
            self.dag = self.relabel_dag_node(self.clct.trail_dag)
            with open(os.path.join(ROOT_PATH, "index2name.txt"), "w") as f:
                for index, name in self.index2name.items():
                    f.write(str(index))
                    f.write(" : ")
                    f.write(str(name))
                    f.write("\n")
        else:
            self.dag = self.clct.trail_dag

        self.forbidden_list = []

        if not os.path.exists(ROOT_PATH):
            os.mkdir(ROOT_PATH)
        if not os.path.exists(os.path.join(ROOT_PATH, "searched_graph")):
            os.mkdir(os.path.join(ROOT_PATH, "searched_graph"))

        # if "BPF_DUMP_INIT_GRAPH_TO" in os.environ:
        #     bpf_dump_init_graph_to = os.environ["BPF_DUMP_INIT_GRAPH_TO"]
        # else:
        #     bpf_dump_init_graph_to = None
        self.base_cost, self.exct_dag, self.base_mem_usage = self.evaluate(
            self.dag, _filename=os.path.join(ROOT_PATH, "searched_graph/base.json"))

        ### Budget, in GB
        self.memory_budget = args_.memory_budget

        ### Some hyper-parameter
        self.enable_defusion = False

        ### DEBUG ONLY
        self.cost_model_error = []

        self.cst_md_mng = CostModelManager(self)

        

    def relabel_dag_node(self, _dag) -> nx.DiGraph:
        def relabel_func(old_label):
            if ("BW" in old_label or "FW" in old_label or "Comm" in old_label or "UPDATE" in old_label) and "^" not in old_label:
                layer_name = parse_layer_name(old_label)
                layer_pid = parse_pid_from_name(old_label)
                # if layer_pid not in self.cost_models or layer_name not in self._wrap_xla_operation_names(layer_pid):
                #     return "DEL~"+old_label
                # TODO (huhanpeng): different pids share the same index
                if "Comm" in old_label and layer_name in self.name2index and layer_pid in self.name2index[layer_name]:
                    layer_index = self.name2index[layer_name][layer_pid]
                    new_label = ("[%d]" % layer_index).join(old_label.split(layer_name))
                    return new_label

                layer_index = len(self.index2name)
                self.index2name[layer_index] = layer_name
                self.index2pid[layer_index] = layer_pid
                if layer_name not in self.name2index:
                    self.name2index[layer_name] = {}
                self.name2index[layer_name][layer_pid] = layer_index
                new_label = ("[%d]" % layer_index).join(old_label.split(layer_name))
                self.index2newname[layer_index] = new_label
                return new_label
            else:
                return old_label
        # new_dag = nx.relabel_nodes(_dag, relabel_func)
        # nodes_to_remove = []
        # for node, attrs in new_dag.nodes.items():
        #     if "DEL~" in node:
        #         nodes_to_remove.append(node)
        # for node in nodes_to_remove:
        #     new_dag.remove_node(node)
        return nx.relabel_nodes(_dag, relabel_func)

    def _parse_index_from_name(self, name_):
        return int(name_.split("[")[1].split("]")[0])
    
    def _debug_convert_to_other_machines(self, name_):
        if not "+" in name_:
            ret = []
            if args_.relabel:
                name, pid = self._get_original_name_pid_from_index(name_)
                for other_pid in self.name2index[name]:
                    if other_pid == pid:
                        continue
                    other_index = self.name2index[name][other_pid]
                    ret.append(self.index2newname[other_index])
            else:
                pid, rawname, cat, suffix = parse_allinfo_from_name(name_)
                for other_pid in self.clct.all_pid():
                    if other_pid == pid:
                        continue
                    ret.append(gen_long_name(other_pid, rawname, suffix))
            return ret
        else:
            new_names = []
            for sub_name in name_.split("+"):
                new_names.append(self._debug_convert_to_other_machines(sub_name))
            new_names = list(np.array(new_names).T)
            return ["+".join(ns) for ns in new_names]

    def _get_original_name_pid_from_index(self, name_):
        if args_.relabel:
            index = self._parse_index_from_name(name_)
            return self.index2name[index], self.index2pid[index]
        else:
            return parse_layer_name(name_), parse_pid_from_name(name_)

    def evaluate(self, _dag, _filename=None, _crit_filename = None):
        # t = time.time()
        ### input _dag is a dependency graph, using the replayer to get the simulated traces and execution graph
        ### Return the iteration time and the execution graph
        _output = False if _filename is None else True
        replayer = Replayer(dag=_dag, _step_num=1,
                            leaf_dirs=self.clct.all_prefix_list(),
                            dump_path=self.clct.pm.path,
                            comm_backend=self.clct.comm_backend,
                            byteps_graph=self.clct.byteps_graph,
                            infi_para_update=args_.update_infi_para)
        step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(
            None, _output=_output, _filename=_filename).values()]
        # print("Evaluate time {}".format(time.time() - t))
        if _crit_filename is not None:
            prefix, crit_file_name = os.path.split(_crit_filename)
            critical_path = list(zip(*self.wrap_critical_path(replayer.exct_dag)))[0]
            replayer.dump_critical_path(crit_file_name, critical_path, prefix=prefix)
        
        estimated_memory_usage = self.memory_estimator.estimate(_dag, self.clct.para_dict)
        # print("output critical path")
        # critical_path = self.wrap_critical_path(replayer.exct_dag)
        # replayer.dump_critical_path("critical_path_{}.json".format(self.tmp_id), [n for (n, e) in critical_path])
        # self.tmp_id += 1

        return max(step_end_time_ms), replayer.exct_dag, estimated_memory_usage

    def candidate_selection(self, GS, topk=None, critical_path=None):
        ''' Select nodes on the critical path of the execution graph as the candidates
            Return the candidates and the revised dependency graph
        '''
        if critical_path is None:
            raise ValueError("critical_path must be given to select candidates")
            if isinstance(GS, GraphState):
                new_dag = self.apply_strategies(self.dag, GS.strategy)
            elif isinstance(GS, nx.DiGraph):
                new_dag = GS
            else:
                raise ValueError("Invalid type for input (type: {}), only GraphState and nx.DiGraph are allowed".format(type(GS)))

            iter_time, exct_dag, mem_usage = self.evaluate(new_dag)
            if isinstance(GS, GraphState) and GS.iter_time is None:
                GS.iter_time = iter_time
                if self.opt_GS is None or iter_time < self.opt_GS.iter_time:
                    self.opt_GS = GS

            ### Need to pick some candidates
            ### TODO (huhanpeng): ??? how to decide which nodes to select as candiates
            ### Currently, pick all nodes on the critical path of the execution graph as the candidates
            critical_path = self.wrap_critical_path(exct_dag)
        else:
            new_dag = GS

        if topk is None:
            return critical_path, new_dag
        else:
            critical_path = sorted(critical_path, key=lambda x: x[1], reverse=True)
            return critical_path[:topk], new_dag

    def wrap_critical_path(self, _dag, verbose=False):
        # t = time.time()
        cal_edge_cost(_dag)
        ret = dag_longest_path(_dag, None, weight="cost", default_weight=0, _debug_level=(1 if verbose else 0))
        # print("critical path time {}".format(time.time() - t))
        return ret

    def _combine_weight(self, l: float, heat: float) -> float:
        # return l * (0.05 + heat)
        return heat + 0.01
        # return 1

    def _get_heat_from_history(self, node):
        heat = 0
        for (h, t) in self.heat_history[node]:
            if h is not None:
                heat += h * np.exp(-0.5*(self.step - t - 1))
        return heat

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        search_space = []
        weights = []
        ### OOM
        # if OOM, only search memory cost model
        if self.mem_usage > self.memory_budget:
            SingleLogger().warn("Estimated memory usage exceeds memory budget: {:.2f}GB > {:.2f}GB".format(
                self.mem_usage, self.memory_budget))
            for _cost_model in self.cst_md_mng.mem_model_list:
                ss_, wei_ = _cost_model.init_search_space(candidates, _dag, _pkg)
                search_space += ss_
                weights += wei_
            if len(search_space) == 0:
                SingleLogger().warn("No optimization strategy to reduce memory usage: {:.2f}GB > {:.2f}GB".format(
                    self.mem_usage, self.memory_budget))
        else:
            SingleLogger().info("Estimated memory usage does not exceed memory budget: {:.2f}GB < {:.2f}GB".format(
                self.mem_usage, self.memory_budget))

            for _cost_model in self.cst_md_mng.cost_model_list:
                ss_, wei_ = _cost_model.init_search_space(candidates, _dag, _pkg)
                search_space += ss_
                weights += wei_
        # SingleLogger().info("Init search space len={} from {} candidates, prune {}".format(len(search_space), len(candidates), prun_cnt))
        # SingleLogger().info("Time spent for spanning tree: {}".format(sum(time_spanning_trees)/ len(time_spanning_trees)))
        # SingleLogger().info("Time spent for source/sink: {}".format(sum(time_st)/ len(time_st)))
        return search_space, weights

    def apply_strategies(self, _dag, _pkg: PKGraph, strategy):
        # print(strategy)
        if isinstance(strategy, list):
            nodes_introduced = set()
            nodes_removed = set()
            for s in strategy:
                op, target, next_ = s
                success, n_introduced, n_removed = self.cst_md_mng.strategy2model[op].apply(s, _dag, _pkg)
                if not success:
                    raise OptApplyStrategyError
                nodes_introduced.update(n_introduced)
                nodes_removed.update(n_removed)
        else:
            op, target, next_ = strategy
            success, nodes_introduced, nodes_removed = self.cst_md_mng.strategy2model[op].apply(strategy, _dag, _pkg)
            if not success:
                raise OptApplyStrategyError
        return nodes_introduced, nodes_removed

    def cost_model_flush(self, is_accept):
        for cm in self.cst_md_mng.cost_model_list:
            cm.flush(is_accept)

    def pick_strategy(self, search_space, weights=None, invalid_strategies=None):
        ### TODO (huhanpeng): need some priority/heuristic
        valid_search_space_idx = []
        valid_weights = []
        if invalid_strategies:
            for st_idx, st in enumerate(search_space):
                if st not in invalid_strategies:
                    valid_search_space_idx.append(st_idx)
                    if weights is not None:
                        valid_weights.append(weights[st_idx])
        else:
            valid_search_space_idx = range(len(search_space))
            if weights is not None:
                valid_weights = weights.copy()
        if not valid_search_space_idx:
            raise OptNoValidStrategyError

        if not valid_weights:
            st_idx = random.choice(valid_search_space_idx)
        else:
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights - np.min(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            try:
                st_idx = random.choices(valid_search_space_idx, weights=valid_weights, k=1)[0]
            except:
                ### Adapt to python <3.6
                st_idx = self.weighted_choice(valid_search_space_idx, valid_weights)
        st = search_space[st_idx]
        search_space.pop(st_idx)
        if weights:
            weights.pop(st_idx)
        return st

    def search(self):
        raise NotImplementedError()

    def weighted_choice(self, choices, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i in range(len(weights)):
            if upto + weights[i] >= r:
                return choices[i]
            upto += weights[i]
        # assert False, "Shouldn't get here"
        return choices[0]


class MCMCOptimizer(Optimizer):
    ''' Markov Chain Monte Carlo algorithm'''

    def __init__(self, *args, **kwargs):
        super(MCMCOptimizer, self).__init__(*args, **kwargs)
        self.heat_history = {}
        if args_.heat_window_size:
            self.heat_window_size = args_.heat_window_size
        else:
            self.heat_window_size = 5

    def search(self, graph_cache=os.path.join(ROOT_PATH, "graph_cache.pickle"), step_size=1):
        G = self.dag.copy()
        PKG = PKGraph(G)

        self.trajectory = []
        ### load init checkpoint
        # G = None
        # for _cost_model in self.cst_md_mng.cost_model_list:
        #     _G, _PKG, _trajectory = _cost_model.load_init_ckpt(G_prime=G)
        #     if _G is not None:
        #         G = _G
        #     if _PKG is not None:
        #         PKG = _PKG
        #     self.trajectory += _trajectory
            
        ### load checkpoint
        if args_.ckpt and graph_cache is not None and os.path.isfile(graph_cache):
            ### TODO (hhp): need to guarantee the consistence of checkpoints of both cost models and DFG states
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.load_ckpt()
            with open(graph_cache, "rb") as f:
                G, PKG, self.heat_window_size, self.heat_history, self.best_cost, self.best_strategy, self.best_step, self.step, self.trajectory = pickle.load(f)
            SingleLogger().info("Loading checkpoint of step {}".format(self.step))
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _filename=os.path.join(ROOT_PATH, "searched_graph/init.json"))
            self.cost_star = self.exct_dag_star = self.mem_usage_star = None
        else:
            for node in G.nodes:
                self.heat_history[node] = [(0, 0)] * self.heat_window_size
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _filename=os.path.join(ROOT_PATH, "searched_graph/init.json"))
            self.cost_star = self.exct_dag_star = self.mem_usage_star = None
            self.best_cost = self.cur_cost
            self.best_strategy = self.trajectory
            self.best_step = 0
            self.step = 0
            self.trajectory = []
            SingleLogger().info("No checkpoint found, search from scratch")

        SingleLogger().info("="*20 + " Search Starts " + "="*20)
        SingleLogger().info("\033[92m" + "Start to search, the original iteration time is %f, init cost is %f" %
                            (self.base_cost, self.cur_cost) + "\033[0m")
        candidates, _ = self.candidate_selection(
            G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
        search_space, weights = self.init_search_space(candidates, G, PKG)
        SingleLogger().info("\033[94m # of candidates: {}, space: {}\033[0m".format(
            len(candidates), len(search_space)))

        def display_and_ckpt():
            SingleLogger().info("\033[94m" + "Current speedup to the origin: %6.4f %%" % (
                100 * (self.base_cost - self.cur_cost) / self.base_cost) + "'\033[0m'")
            SingleLogger().info("\033[94m" + "Best speedup: %d th step, speed up to the origin: %6.4f %%" % (
                self.best_step, 100 * (self.base_cost - self.best_cost) / self.base_cost) + "'\033[0m'\n")

            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {},{},{}".format(
                    self.step,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost,
                    100 * (self.base_cost - self.best_cost) / self.base_cost) + "\n")

            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": self.best_strategy}, f)

            ### checkpints
            if args_.ckpt:
                for _cost_model in self.cst_md_mng.cost_model_list:
                    _cost_model.checkpoint()
                with open(graph_cache, "wb") as f:
                    pickle.dump([G, PKG, self.heat_window_size, self.heat_history,
                                 self.best_cost, self.best_strategy, self.best_step, self.step, self.trajectory], f)

        '''
        ### Test some strategies
        grp_num_to_test = [1, 10, 20, 40, 80]
        search_space = [("++", grp_num, None) for grp_num in grp_num_to_test]
        for st in search_space:
            G_star = G.copy()
            PKG_star = PKG.copy()
            nodes_introduced, nodes_removed = self.apply_strategies(G_star, PKG_star, st)
            self.cost_star, self.exct_dag_star, self.mem_usage_star = self.evaluate(
                G_star, _filename=os.path.join(ROOT_PATH, "searched_graph/grp_num_{}.json".format(st[1])))
            SingleLogger().info("\033[94m Group Num: {}: default fusion: {} ms, Cur cost: {} ms \033[0m".format(
                st[1], self.cur_cost, self.cost_star))
            self.cost_model_flush(False)
        raise
        '''
        
        while len(search_space) > 0:
            invalid_strategies = set()
            while True and len(search_space) > 0:
                G_star = G.copy()
                PKG_star = PKG.copy()
                successful_strategies = 0
                strategy_history_in_step = []
                strategy_introduced_nodes = set()
                strategy_removed_nodes = set()
                while successful_strategies < step_size:
                    try:
                        strategy = self.pick_strategy(search_space, weights=weights, invalid_strategies=invalid_strategies)
                        msg = "\033[94m" + "Picked strategy ({}, {}, {}).".format(*strategy)
                        if len(msg) > 200:
                            msg = msg[:200] + "..."
                        SingleLogger().info(msg + "\033[0m")
                    except OptNoValidStrategyError:
                        # no valid strategies available, refresh search space
                        SingleLogger().info("\033[94m Search space exhausted. \033[0m")
                        candidates, _ = self.candidate_selection(G_star, topk=None, critical_path=None)
                        search_space, weights = self.init_search_space(candidates, G_star, PKG_star)
                        invalid_strategies = set()
                        continue
                    try:
                        nodes_introduced, nodes_removed = self.apply_strategies(G_star, PKG_star, strategy)
                    except OptApplyStrategyError:
                        # strategy invalid
                        # traceback.print_exc()
                        SingleLogger().warn("Strategy invalid (will cause a cycle in the DAG).")
                        invalid_strategies.add(strategy)
                        continue
                    except OptQueryCostModelError:
                        SingleLogger().warn("Strategy invalid (failed to query cost model).")
                        invalid_strategies.add(strategy)
                        continue
                    successful_strategies += 1
                    strategy_history_in_step.append(strategy)
                    strategy_introduced_nodes.update(nodes_introduced)
                    strategy_removed_nodes.update(nodes_removed)

                    if self.step % 100 == 0:
                        self.cost_star, self.exct_dag_star, self.mem_usage_star = \
                            self.evaluate(G_star, 
                            _filename=os.path.join(ROOT_PATH, "searched_graph/{}.json".format(self.step)),
                            _crit_filename=os.path.join(ROOT_PATH, "searched_graph/{}_crit.json".format(self.step)))
                        # dump cluster mapping
                        ### TODO (HHP): we should only dump cluster mapping for the best strategy 
                        # if "+" in self.cst_md_mng.strategy2model:
                        #     self.cst_md_mng.strategy2model["+"]._dump_cluster_mapping(G, os.path.join(ROOT_PATH, "searched_graph/cluster_mapping_{}.txt".format(self.step)))
                    else:
                        self.cost_star, self.exct_dag_star, self.mem_usage_star = self.evaluate(G_star)
                    if successful_strategies < step_size:
                        candidates, _ = self.candidate_selection(
                            G_star, topk=None, critical_path=self.wrap_critical_path(self.exct_dag_star))
                        search_space, weights = self.init_search_space(
                            candidates, G_star, PKG_star)
                    invalid_strategies = set()
                    msg = "\033[94m" + "Strategy ({}, {}, {}) successfully applied.".format(*strategy)
                    if len(msg) > 200:
                        msg = msg[:200] + "... successfully applied."
                    SingleLogger().info(msg + "\033[0m")

                # if step < 200:
                #     MCMC_BETA = 1
                # else:
                #     MCMC_BETA = args.mcmc_beta
                SingleLogger().info("\033[94m Step: {} - cost from {} -> {} \033[0m".format(
                    self.step, self.cur_cost, self.cost_star))
                self.step += 1

                ### Update heat history
                combined_history_list = [None] * self.heat_window_size
                for node in strategy_removed_nodes:
                    if node in self.heat_history:
                        node_history = self.heat_history[node]
                        for idx, (h, _) in enumerate(node_history):
                            if combined_history_list[idx] is None:
                                combined_history_list[idx] = [h]
                            elif h is None:
                                continue
                            else:
                                combined_history_list[idx].append(h)
                        node_history.insert(0, (self.cur_cost - self.cost_star, self.step))
                        node_history.pop()
                        # print("node: {}".format(node))
                        # print("node_history: {}".format(node_history))
                        # input()
                combined_history = []
                for h_list in combined_history_list:
                    if h_list is not None:
                        h_avg = np.average(h_list)
                        combined_history.append((h_avg, self.step))
                    else:
                        combined_history.append((None, None))
                
                # calculate the proposal probability q here
                # proposed_candidates, _ = self.candidate_selection(
                        # G_star, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
                # proposed_search_space, proposed_weights = self.init_search_space(
                        # proposed_candidates, G_star, PKG_star)
                # forward_prob = 1 / (len(search_space)+1)
                # backward_prob = 1 / len(proposed_search_space)

                is_accept = self.accept_or_not(self.cur_cost, self.cost_star)
                ### update cost model internal states
                self.cost_model_flush(is_accept)
                if is_accept:
                    invalid_strategies = set()

                    ### generate history for new nodes
                    combined_history.insert(0, (self.cur_cost - self.cost_star, self.step))
                    combined_history.pop()
                    for node in strategy_introduced_nodes:
                        self.heat_history[node] = combined_history.copy()

                    G = G_star
                    PKG = PKG_star
                    self.trajectory += strategy_history_in_step
                    self.cur_cost = self.cost_star
                    self.exct_dag = self.exct_dag_star
                    self.mem_usage = self.mem_usage_star

                    ### clean up heat history for removed nodes
                    for node in strategy_removed_nodes:
                        if node in self.heat_history:
                            self.heat_history.pop(node)

                    ### Cache the best strategy
                    if self.cur_cost < self.best_cost:
                        self.best_cost = self.cur_cost
                        self.best_strategy = self.trajectory.copy()
                        self.best_step = self.step - 1
                        if "+" in self.cst_md_mng.strategy2model:
                            self.cst_md_mng.strategy2model["+"]._dump_cluster_mapping(
                                G, os.path.join(ROOT_PATH, "cluster_mapping.txt"))
                        
                        if "++" in self.cst_md_mng.strategy2model:
                            self.cst_md_mng.strategy2model["++"].dump_tensor_grp_mapping()
                        # DEBUG: log best graph for debugging
                        # self.evaluate(G, 
                        #     _filename=os.path.join(ROOT_PATH, "best.json".format(self.step)),
                        #     _crit_filename=os.path.join(ROOT_PATH, "best_crit.json".format(self.step)))
                    ### Init new search space
                    candidates, _ = self.candidate_selection(
                        G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
                    search_space, weights = self.init_search_space(
                        candidates, G, PKG)
                    # candidates = proposed_candidates
                    # search_space = proposed_search_space
                    # weights = proposed_weights
                    break
            display_and_ckpt()
        display_and_ckpt()

    def accept_or_not(self, cost, new_cost):
        # prob = min(1, (math.exp(beta * (cost - new_cost))))
        try:
            prob = math.exp(MCMC_BETA*math.log(self.step+1) * (cost - new_cost))
        except OverflowError:
            prob = float('inf')
        # if cost > new_cost:
        if prob > 1:
            SingleLogger().info(
                "\033[92m" + "Accept a better action, orig cost: {}, new cost: {}".format(cost, new_cost) + "\033[0m")
            return True
        else:
            # prob = math.exp(MCMC_BETA * (cost - new_cost))
            r = random.random()
            if r < prob:
                SingleLogger().info(
                    "\033[92m" + "Accept a worse action with random value: {} < {} ".format(r, prob) + "\033[0m")
                return True
            else:
                SingleLogger().info(
                    "\033[93m" + "Rejected a worse action with random value: {} >= {} ".format(r, prob) + "\033[0m")
                return False


class MCTSOptimizer(Optimizer):
    ''' Monte Carlo Tree Search '''

    def __init__(self, *args, **kwargs):
        super(MCTSOptimizer, self).__init__(*args, **kwargs)
        self.loop_cnt = 0
        self.GS_root = None
        self.opt_GS = None
        self.ucb_type = args_.ucb_type
        if self.ucb_type != "MAX" and self.ucb_type != "AVG":
            raise ValueError(
                "UCB type should be MAX or AVG, but {} is given.".format(self.ucb_type))
        self.no_mutation = args_.no_mutation

    def search(self):
        ### Initialize the root graph state
        self.GS_root = GraphState(depth=0)
        self.GS_root.strategy = []

        while self.check_loop_time() and self.check_loop_num():
            GS = self.tree_policy(self.GS_root)
            reward = self.default_policy(GS)
            SingleLogger().info("Speedup to the origin %6.4f %%" % (100 * reward))
            self.backpropagation(GS, reward)
            if args_.ucb_visual:
                self.visualize_tree()
            self.show_opt_strategies()
        return

    def visualize_tree(self):
        def iter_print(GS, cnt):
            ### `cnt` is used to decide how many parent branches to print for current nodes
            LENOFNODE = 11
            LENOFARROW = 5
            node_string = "  %5.4f %% " % (
                GS.quality * 100) if GS.quality >= 0 else " -%5.4f %% " % (-GS.quality * 100)
            sys.stdout.write(node_string)
            assert len(node_string) == LENOFNODE
            if GS.childs is None:
                return
            for idx, child in enumerate(GS.childs):
                if idx > 0:
                    sys.stdout.write("\n{}".format(" "*(LENOFNODE + LENOFARROW//2)))
                    sys.stdout.write("{}".format(" "*((LENOFNODE + LENOFARROW) * (GS.depth - cnt))))
                    sys.stdout.write("{}".format(("|" + " " * (LENOFNODE + LENOFARROW - 1))*(cnt)))
                    sys.stdout.write("{}".format("|" if idx < (len(GS.childs) - 1) else "\\"))
                    sys.stdout.write("{}".format("-"*(LENOFARROW - LENOFARROW//2 - 1)))
                else:
                    sys.stdout.write("{}".format('-'*LENOFARROW))
                if idx < (len(GS.childs) - 1):
                    next_cnt = cnt + 1
                else:
                    next_cnt = cnt
                iter_print(child, next_cnt)

        iter_print(self.GS_root, 0)
        sys.stdout.write("\n")

    def show_opt_strategies(self):
        SingleLogger().info("Best speedup: %d th layer, speed up to the origin: %6.4f %%" %
                            (len(self.opt_GS.strategy), 100 * self.opt_GS.quality))

    def check_loop_num(self):
        self.loop_cnt += 1
        if self.loop_cnt > MAX_LOOP:
            return False  # End
        else:
            return True  # continue

    def check_loop_time(self):
        return True  # continue

    def tree_policy(self, GS):
        while self.fully_expanded(GS):
            GS = self.best_UCB(GS)
        return self.expansion(GS)

    def default_policy(self, GS):
        if not self.no_mutation:
            while not self.terminal(GS):
                action = self.pick_strategy(GS.space)[0]
                GS_c = GraphState(depth=(GS.depth+1))
                GS_c.strategy = GS.strategy.copy()
                GS_c.strategy.append(action)
                GS = GS_c
        ### Evaluate the final graph
        if GS.iter_time is None:
            self.check_search_space(GS)
        cost = GS.iter_time
        SingleLogger().debug("Evaluate the strategy %s" % (str(GS.strategy)))
        return (self.base_cost - cost)/self.base_cost

    def backpropagation(self, GS, reward):
        if self.ucb_type == "MAX":
            GS.quality = max(reward, GS.quality)
        elif self.ucb_type == "AVG":
            GS.quality += reward
        GS.visit_cnt += 1
        if GS.depth == 0:
            return
        else:
            self.backpropagation(GS.parent, reward)

    def best_UCB(self, GS):
        GS_opt = c_opt = None
        for GS_c in GS.childs:
            if self.ucb_type == "MAX":
                c = GS_c.quality + UCB_GAMMA * \
                    math.sqrt((2 * math.log(GS.visit_cnt)) / GS_c.visit_cnt)
            elif self.ucb_type == "AVG":
                c = GS_c.quality / GS_c.visit_cnt + UCB_GAMMA * \
                    math.sqrt((2 * math.log(GS.visit_cnt)) / GS_c.visit_cnt)
            else:
                raise RuntimeError("Invalid UCB_type")
            if GS_opt is None or c > c_opt:
                c_opt = c
                GS_opt = GS_c
        return GS_opt

    def fully_expanded(self, GS):
        if self.terminal(GS):
            return False

        if GS.state == GraphExpand.NOT or GS.state == GraphExpand.PARTIAL:
            return False
        else:
            return True

    def expansion(self, GS):
        ### Pick an unvisided child to expand
        assert GS.state == GraphExpand.NOT or GS.state == GraphExpand.PARTIAL
        action = self.pick_unvisited(GS)
        if action is None:
            ### Current state is the terminal state, expansion failed
            return GS

        GS_c = GraphState(depth=(GS.depth+1))
        GS_c.strategy = GS.strategy.copy()
        GS_c.strategy.append(action)
        GS_c.parent = GS
        if GS.childs is None:
            GS.childs = []
        GS.childs.append(GS_c)

        if len(GS.space) == len(GS.childs):
            GS.state = GraphExpand.FULLY
        else:
            GS.state = GraphExpand.PARTIAL

        return GS_c

    def pick_unvisited(self, GS):
        ### TODO (huhanpeng): how to pick with some heuristic
        for idx in range(len(GS.space)):
            if GS.space[idx][1] == 0:
                GS.space[idx][1] += 1
                return GS.space[idx][0]
        return None

    def check_search_space(self, GS):
        ### TODO (huhanpeng): we can do some pruning here
        if GS.space is None:
            candidates, new_dag = self.candidate_selection(GS, topk=None)
            search_space, _ = self.init_search_space(candidates, new_dag)
            # The integer value is used as a counter
            GS.space = [[action, 0] for action in search_space]

    def terminal(self, GS):
        self.check_search_space(GS)
        if GS.depth > MAX_TREE_DEPTH or len(GS.space) == 0:
            return True
        else:
            return False
