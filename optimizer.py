import networkx as nx
import random
import math
import time
import numpy as np
import code
import traceback
import pickle
import ujson as json

from replay import Replayer
from trace_utils import *
from dag_utils import *

from memory import MemoryEstimator
from memory.cost_model import IncreasingBatchSizeCostModel, MemoryGraphPass

from cost_model.base import OptApplyStrategyError, OptNoValidStrategyError, OptQueryCostModelError
from cost_model._xla.pk_graph import PKGraph
from cost_model.mixed_precision import AMPGraphPass
from cost_model.tensor_fusion import TensorFusionGraphPass

class GraphExpand(Enum):
    NOT = 0
    PARTIAL = 1
    FULLY = 2

args_ = arg_utils.SingleArg().args
if args_.option == "optimize" and args_.sub_option not in ["amp", "tensor_fusion"]:
    from cost_model.op_fusion import XLAGraphPass
    # import horovod.tensorflow as hvd

MAX_TREE_DEPTH = 1000
MAX_LOOP = 1000
UCB_GAMMA = args_.ucb_gamma
MCMC_BETA = args_.mcmc_beta
ROOT_PATH = os.path.join(
    args_.workspace if args_.workspace else args_.path, ".opt_ws")

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


class CostModelManager:
    def __init__(self, opt):
        self.cost_model_list = []
        # if "amp" in args_.sub_option:
        #     self.cost_model_list.append(AMPGraphPass(opt))
        if "tensor_fusion" in args_.sub_option:
            self.cost_model_list.append(TensorFusionGraphPass(opt))
        if "xla" in args_.sub_option:
            self.cost_model_list.append(XLAGraphPass(opt))
        if len(self.cost_model_list) == 0:
            SingleLogger().warn("No optimization techniques for computation. ")
            # self.cost_model_list = [
            #    XLAGraphPass(opt),
            #    AMPGraphPass(opt),
            # ]
        if "^memory" in args_.sub_option:
            self.mem_model_list = []
        else:
            self.cost_model_list.append(IncreasingBatchSizeCostModel(opt))
            self.mem_model_list = [MemoryGraphPass(opt)]
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
            self.dag, _path=os.path.join(ROOT_PATH, "searched_graph/base.json"))

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
                layer_name = parse_op_name(old_label)
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
            return parse_op_name(name_), parse_pid_from_name(name_)

    def evaluate(self, _dag, _path=None, _crit_filename=None):
        # t = time.time()
        ### input _dag is a dependency graph, using the replayer to get the simulated traces and execution graph
        ### Return the iteration time and the execution graph
        _output = False if _path is None else True
        replayer = Replayer(dag=_dag, _step_num=1,
                            leaf_dirs=self.clct.all_prefix_list(),
                            dump_path=self.clct.pm.path,
                            comm_backend=self.clct.comm_backend,
                            byteps_graph=self.clct.byteps_graph,
                            infi_para_update=args_.update_infi_para)
        step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(
            None, _output=_output, _path=_path).values()]
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

            cm_types = []
            cm_start_end = []
            cm_weight_dict = {}
            for _cost_model in self.cst_md_mng.cost_model_list:
                if isinstance(_cost_model, XLAGraphPass):
                    model_type = "xla"
                elif isinstance(_cost_model, TensorFusionGraphPass):
                    model_type = "tensor_fusion"
                else:
                    raise NotImplementedError
                cm_types.append(model_type)
                ss_, wei_ = _cost_model.init_search_space(candidates, _dag, _pkg)
                search_space += ss_
                cm_start_end.append((len(weights), len(weights)+len(wei_)))
                cm_weight_dict[model_type] = sum(wei_)
                SingleLogger().info("Weight sum for {}: {}".format(model_type, sum(wei_)))
                weights += wei_
            # assign a specific portion to each strategy, according to step size
            # TEMP: xla : tensor_fusion = 2:1
            if len(cm_weight_dict) >= 2:
                for idx, cm_type in enumerate(cm_types):
                    if cm_type == "xla":
                        scale_factor = cm_weight_dict["tensor_fusion"] / cm_weight_dict["xla"] * 2
                        SingleLogger().info("Scale factor: {}".format(scale_factor))
                        for i in range(*cm_start_end[idx]):
                            weights[i] *= scale_factor
        return search_space, weights

    def apply_strategies(self, _dag, _pkg: PKGraph, strategy):
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

        st_idx = self.select_one_stategy(valid_weights, valid_search_space_idx)
        st = search_space[st_idx]
        search_space.pop(st_idx)
        if weights:
            weights.pop(st_idx)
        return st
    
    def select_one_stategy(self, valid_weights, valid_search_space_idx):
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
        return st_idx

    def pick_strategies(self, search_space, weights=None, invalid_strategies=None):
        valid_search_space_idx = []
        valid_weights = []
        grouped_sts = {}
        for st_idx, st in enumerate(search_space):
            if invalid_strategies and st not in invalid_strategies:
                valid_search_space_idx.append(st_idx)
                if weights is not None:
                    valid_weights.append(weights[st_idx])
                ### TODO (HHP): only consider operator fusion and tensor fusion here
                token = "+" if st[0] == "+" or st[0] == "-" else "++"
                if token not in grouped_sts:
                    grouped_sts[token] = {"space": [], "weights": []}
                grouped_sts[token]["space"].append(st_idx)
                grouped_sts[token]["weights"].append(weights[st_idx])
        if not valid_search_space_idx:
            raise OptNoValidStrategyError
        
        st_list = [None for cm in self.cst_md_mng.cost_model_list]
        while True:
            st_idx = self.select_one_stategy(valid_weights, valid_search_space_idx)
            is_exist_none = False
            for cm_idx, cm in enumerate(self.cst_md_mng.cost_model_list):
                st = search_space[st_idx]
                if st_list[cm_idx] is None:
                    if st[0] in cm.token:
                        st_list[cm_idx] = st
                        search_space.pop(st_idx)
                        if weights:
                            weights.pop(st_idx)
                    is_exist_none = True
            if not is_exist_none:
                break
        return [st for st in st_list if st is not None]

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

    def search(self, graph_cache=os.path.join(ROOT_PATH, "graph_cache.pickle")):
        step_size = args_.step_size

        G = self.dag.copy()
        PKG = PKGraph(G)

        self.trajectory = []
        ## load init checkpoint
        G = None
        for _cost_model in self.cst_md_mng.cost_model_list:
            _G, _PKG, _trajectory = _cost_model.load_init_ckpt(G_prime=G)
            if _G is not None:
                G = _G
            if _PKG is not None:
                PKG = _PKG
            self.trajectory += _trajectory

        ### load checkpoint
        if args_.ckpt and graph_cache is not None and os.path.isfile(graph_cache):
            ### TODO (hhp): need to guarantee the consistence of checkpoints of both cost models and DFG states
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.load_ckpt()
            with open(graph_cache, "rb") as f:
                G, PKG, self.heat_window_size, self.heat_history, self.best_cost, self.best_strategy, self.best_step, self.step, self.trajectory = pickle.load(f)
            SingleLogger().info("Loading checkpoint of step {}".format(self.step))
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"))
            self.cost_star = self.exct_dag_star = self.mem_usage_star = None
        else:
            for node in G.nodes:
                self.heat_history[node] = [(0, 0)] * self.heat_window_size
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"))
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
        
        while True:
            invalid_strategies = set()
            while len(search_space) > 0:
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
                    ### Apply strategies
                    is_succ_apply = True
                    nodes_introduced = []
                    nodes_removed = []
                    ### TODO (HHP): futher modify to allow apply mutliple stragtegies at one step
                    strategies = [strategy]
                    for st in strategies:
                        try:
                            _nodes_introduced, _nodes_removed = self.apply_strategies(G_star, PKG_star, st)
                            nodes_introduced += _nodes_introduced
                            nodes_removed += _nodes_removed
                        except OptApplyStrategyError:
                            # strategy invalid
                            # traceback.print_exc()
                            SingleLogger().warn("Strategy invalid (will cause a cycle in the DAG).")
                            invalid_strategies.add(st)
                            is_succ_apply = False
                            break
                        except OptQueryCostModelError:
                            SingleLogger().warn("Strategy invalid (failed to query cost model).")
                            invalid_strategies.add(st)
                            is_succ_apply = False
                            break
                    if not is_succ_apply:
                        continue
                    successful_strategies += 1
                    strategy_history_in_step += strategies
                    strategy_introduced_nodes.update(nodes_introduced)
                    strategy_removed_nodes.update(nodes_removed)

                    if self.step % 100 == 0:
                        self.cost_star, self.exct_dag_star, self.mem_usage_star = \
                            self.evaluate(G_star, 
                            _path=os.path.join(ROOT_PATH, "searched_graph/{}.json".format(self.step)),
                            _crit_filename=os.path.join(ROOT_PATH, "searched_graph/{}_crit.json".format(self.step)))
                        # dump cluster mapping
                        ### TODO (HHP): we should only dump cluster mapping for the best strategy 
                        # if "+" in self.cst_md_mng.strategy2model:
                        #     self.cst_md_mng.strategy2model["+"]._dump_cluster_mapping(G, 
                        #          os.path.join(ROOT_PATH, "searched_graph/cluster_mapping_{}.txt".format(self.step)),
                        #           partition=True)
                    else:
                        try:
                            self.cost_star, self.exct_dag_star, self.mem_usage_star = self.evaluate(G_star)
                        except:
                            traceback.print_exc()
                            print("~~~~~~~~~~~~~~FAILED TO RUN REPLAY~~~~~~~~~~~~~")
                            import code
                            code.interact(local=locals())
                            exit(-1)
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
                if strategy[0] in ["gradient_accumulation", "recomputation"]:
                    is_accept = True
                else:    
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
                                G, os.path.join(ROOT_PATH, "cluster_mapping.txt"), partition=True)
                        
                        if "++" in self.cst_md_mng.strategy2model:
                            self.cst_md_mng.strategy2model["++"].dump_tensor_grp_mapping()
                        # DEBUG: log best graph for debugging
                        self.evaluate(G, 
                            _path=os.path.join(ROOT_PATH, "best.json".format(self.step)),
                            _crit_filename=os.path.join(ROOT_PATH, "best_crit.json".format(self.step)))
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
            if len(search_space) == 0:
                ### Init new search space
                candidates, _ = self.candidate_selection(
                    G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
                search_space, weights = self.init_search_space(
                    candidates, G, PKG)
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
