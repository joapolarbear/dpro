from collections import deque
import enum
import networkx as nx
import random
import math
import time
from collections import deque
from networkx.algorithms.coloring.greedy_coloring import strategy_connected_sequential
from scipy.stats.mstats import gmean
import numpy as np
import code
import traceback
import pickle
import ujson as json

from tqdm import tqdm, trange

from replay import Replayer
from trace_utils import *
from dag_utils import *
from cost_model_amp.amp_pred import AMPPredictor
from cost_model_xla.pk_graph import PKGraph, PKGraphCycleError, contract_nodes_nx, \
                    defuse_nodes_inplace_nx, postorder_contract_nx, subgraph_partition_connected_nx

class GraphExpand(Enum):
    NOT=0
    PARTIAL=1
    FULLY=2

args_ = arg_utils.SingleArg().args
MAX_TREE_DEPTH = 1000
MAX_LOOP = 1000
UCB_GAMMA = args_.ucb_gamma
MCMC_BETA = args_.mcmc_beta
ROOT_PATH = args_.workspace
if not os.path.exists(ROOT_PATH):
    os.mkdir(ROOT_PATH)
if not os.path.exists(os.path.join(ROOT_PATH, "searched_graph")):
    os.mkdir(os.path.join(ROOT_PATH, "searched_graph"))

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

        self.state = GraphExpand.NOT  ### Whether the actions have been tranversed, not, partial or fully

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

class _BaseCostModel:
    def __init__(self, opt):
        self.opt = opt
        self.dag = self.opt.dag
        self.node_attr_cache = self.opt.node_attr_cache
        ### token is the indendifier of each optimization technique
        self.token = None

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        raise NotImplementedError()

    def apply(self, s, __dag, __pkg):
        raise NotImplementedError()

    def init_partition(self, G, PKG, initial_partitions_formed) -> int:
        return 0

class _XLACostModel(_BaseCostModel):
    def __init__(self, opt, xla_cost_model):
        super().__init__(opt)
        self.cost_models = xla_cost_model
        self.forbidden_list = set()
        self.init_forbidden_list()
        self.token = ["+", "-"]

    def init_forbidden_list(self):
        self.initial_forbidden_list = set()
        # limit the range of nodes during search
        for node in self.dag.nodes:
            # ignore BW nodes and communication nodes
            if "BW" in node:
                self.initial_forbidden_list.add(node)

            try:
                orig_name, pid = self._get_original_name_pid_from_index(node)
            except:
                # not standard nodes, ignore
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                continue
            cat = parse_cat_from_name(orig_name)
            if orig_name not in self.cost_models[pid].graph_def_util.operation_names or "Assign" in orig_name or cat == CatName.COMM.value:
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)

    def init_partition(self, G, PKG, initial_partitions_formed):
        partition_G = G.copy()
        partition_PKG = PKGraph(partition_G)

        source_nodes = sorted([node for node in partition_G.nodes if node not in self.initial_forbidden_list], key=lambda x: partition_G.in_degree(x))

        # Run post order traversal on partition_G
        visited_nodes = set()
        for source in tqdm(source_nodes, total=len(source_nodes)):
            if source not in visited_nodes and source in partition_G.nodes:
                _, _, partition_G = postorder_contract_nx(partition_G, partition_PKG, source, visited_nodes, forbidden_list=self.initial_forbidden_list, size_limit=800)
        for node_name in tqdm(partition_G.nodes()):
            if "+" in node_name:
                # fused node, test if compilable
                try:
                    self.parse_node_attr(partition_G, node_name)
                    compilable=True
                except OptQueryCostModelError:
                    # traceback.print_exc()
                    compilable=False
                if compilable:
                    ns = node_name.split("+")
                    G, new_node_name = contract_nodes_nx(G, ns)
                    PKG.contract_nodes_unsafe(ns)
                    self.parse_node_attr(G, new_node_name)
                    initial_partitions_formed += 1

        return initial_partitions_formed

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
                    raise RuntimeError("Fused DAG node {} contains ops from different machines.".format(u_))
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
            return self.cost_models[pid].predict(nodes_to_fuse)

    def _wrap_xla_need_fuse(self, pid, orig_name):
        return (args_.simulate or orig_name in self.cost_models[pid].graph_def_util.operation_names) and orig_name not in self.forbidden_list

    def _query_cost_model(self, fused_u_):
        # query cost model for exec time of a fused node u
        nodes_in_u, u_pid = self._get_original_name_pid_from_fused_node(fused_u_)
        nodes_to_fuse = set(nodes_in_u)
        if len(nodes_to_fuse) < 10:
            SingleLogger().info("[COST MODEL QUERY] {} Nodes to fuse: {}".format(len(nodes_to_fuse), nodes_to_fuse))
        else:
            SingleLogger().info("[COST MODEL QUERY] {} Nodes to fuse ...".format(len(nodes_to_fuse)))

        predicted_time, _ = self._wrap_xla_predict(u_pid, nodes_to_fuse, fused_u_)

        # executed_time, _ = self.cost_models[u_pid].execute(nodes_to_fuse)
        predicted_time /= 1000
        # executed_time /= 1000
        SingleLogger().info("[COST MODEL QUERY] Exec time predicted: {}".format(predicted_time))
        # SingleLogger().info("[COST MODEL QUERY] Actuall exec time: {}".format(executed_time))
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

            if "+" in n:
                ### This a fused node
                ns = n.split("+")
                cat = parse_cat_fine_grained(ns[0])
                pid = parse_pid_from_name(ns[0])
                ns = set(ns)
                subgraph = self.dag.subgraph(ns)

                # st = time.time()
                # randomly split edges using spanning tree
                valid_split_plans = subgraph_partition_connected_nx(subgraph)
                split_weights = []
                for splits in valid_split_plans:
                    split_weights.append(gmean([len(nodes) for nodes in splits]))
                split_weights = np.exp(5e-4*(len(ns) - 80)) * (np.array(split_weights) / np.sum(split_weights))
                for split_index, splits in enumerate(valid_split_plans):
                    search_space.append(("-", n, splits))
                    weights.append(self.opt._combine_weight(l, heat) * split_weights[split_index] )
            else:
                ### Nodes that have never been fused
                cat = parse_cat_fine_grained(n)
                pid = parse_pid_from_name(n)
            try:
                n_orig_name, n_pid = self.opt._get_original_name_pid_from_index(n)
            except (IndexError, KeyError):
                continue

            if not self._wrap_xla_need_fuse(n_pid, n_orig_name):
                continue

            for succ_ in _dag.successors(n):
                # some filters
                if not _pkg.can_contract_edge(n, succ_):
                    continue
                if "+" not in succ_:
                    try:
                        succ_orig_name, succ_pid = self.opt._get_original_name_pid_from_index(succ_)
                    except (IndexError, KeyError):
                        continue

                    if not self._wrap_xla_need_fuse(succ_pid, succ_orig_name):
                        continue

                _pid = parse_pid_from_name(succ_)
                _cat = parse_cat_fine_grained(succ_)
                if pid != _pid or cat != _cat:
                    continue

                ### Assumption 1: for edge a->b, only if the indegree of b is 1, the node can be fused
                # bw_v = self.convert_fw2bw(n)
                # if len(_dag.in_edges(succ_)) > 1 or len(_dag.in_edges(bw_v)) > 1:
                # if len(_dag.in_edges(succ_)) > 1:
                #     continue

                # bw_u = self.convert_fw2bw(succ_)
                # assert bw_u in _dag.nodes and bw_v in _dag.nodes

                ### Assumption 2: for edge bw_u->bw_v, if comm_bw_u > bw_v, it can not bring any speedup if fusing u and v.
                def ret_comm_time(_node):
                    __ret = _dag.nodes[_node]["avg"]
                    for __succ in _dag.successors(_node):
                        _pid = parse_pid_from_name(__succ)
                        if "Comm" in __succ and pid == _pid:
                            __ret += ret_comm_time(__succ)
                    return __ret

                comm_t = 0
                # for bw_u_succ in _dag.successors(bw_u):
                for bw_u_succ in _dag.successors(n):
                    if "Comm" in bw_u_succ:
                        if self.opt.comm_backend == "NCCL":
                            comm_t += ret_comm_time(bw_u_succ)
                        else:
                            ### TODO (huhanpeng): is there only one comm sub operator ???
                            comm_t += _dag.nodes[bw_u_succ]["avg"]
                
                # if comm_t >= _dag.nodes[bw_v]["avg"]:
                if comm_t >= _dag.nodes[succ_]["avg"]:
                    prun_cnt += 1
                    # G_star = self.apply_strategies(self.dag, ("+", n, succ_))
                    # iter_time, _ = self.evaluate(G_star)
                    SingleLogger().debug("Prune fusing {} and {} with comm time {}".format(n, succ_, comm_t))
                    continue

                # calculate heat using max(heat(n), heat(succ_))
                heat_succ = self.opt._get_heat_from_history(succ_)

                heat_combined = (heat + heat_succ) / 2

                # if n == "traces_0.rank0->BW.[2654]":
                #     print("Heat for (traces_0.rank0->BW.[2654], {}):".format(succ_))
                #     print(heat_combined)
                #     input()
                
                # if heat_combined > 0:
                #     print("Heat for ({}, {}): {}".format(n, succ_, heat_combined))
                #     input()

                # DEBUG_COMPARE
                # heat_combined = 1

                search_space.append(("+", n, succ_))
                weights.append(self.opt._combine_weight(l, heat_combined))
        SingleLogger().info("Init search space len={} from {} candidates, prune {}".format(len(search_space), len(candidates), prun_cnt))
        # SingleLogger().info("Time spent for spanning tree: {}".format(sum(time_spanning_trees)/ len(time_spanning_trees)))
        # SingleLogger().info("Time spent for source/sink: {}".format(sum(time_st)/ len(time_st)))
        return search_space, weights

    def concat_name(self, u_, v_):
        return "%s+%s"%(u_, v_)

    def combine_avg(self, u, v):
        # call cost model to obtain the combined time
        fused_name = self.concat_name(u, v)
        return self._query_cost_model(fused_name)

    def combine_gap(self, ug, vg):
        ### TODO (huhanpeng): key component
        ### Use max to avoid one input is zero, 
        ### some how for the new gap x, ug < x < ug + vg, vg < x < ug + vg
        # return max(max((ug + vg) / 0.8, ug), vg)
        return max(ug, vg)

    def combine_nodes_attr(self, _dag, target, u_, v_):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
        _dag.nodes[target]["avg"] = self.combine_avg(u_, v_)
        _dag.nodes[target][GAP_STR_OP2OP] = self.combine_gap(self.opt.get_node_attr(u_, GAP_STR_OP2OP), self.opt.get_node_attr(v_, GAP_STR_OP2OP))
        _dag.nodes[target][GAP_STR_OP2COMM] = self.combine_gap(self.opt.get_node_attr(u_, GAP_STR_OP2COMM), self.opt.get_node_attr(v_, GAP_STR_OP2COMM))

    def combine_attr_except_avg(self, target, attr1, attr2):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
        # target["avg"] = self.combine_avg(attr1["avg"], attr2["avg"])

        if GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self.combine_gap(attr1[GAP_STR_OP2OP], attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP not in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self.combine_gap(0, attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP not in attr2:
            target[GAP_STR_OP2OP] = self.combine_gap(attr1[GAP_STR_OP2OP], 0)

        if GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self.combine_gap(attr1[GAP_STR_OP2COMM], attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM not in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self.combine_gap(0, attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM not in attr2:
            target[GAP_STR_OP2COMM] = self.combine_gap(attr1[GAP_STR_OP2COMM], 0)

    def parse_node_attr(self, _dag, new_name):
        if new_name in self.node_attr_cache:
            nx.set_node_attributes(_dag, {new_name:self.node_attr_cache[new_name]})
            # _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            ns = new_name.split("+")
            attrs = self.node_attr_cache[ns[0]].copy()
            for idx in range(1, len(ns)):
                self.combine_attr_except_avg(attrs, attrs, self.node_attr_cache[ns[idx]])
            # combine attr avg
            attrs["avg"] = self._query_cost_model(new_name)
            ### set and cache the attribute
            nx.set_node_attributes(_dag, {new_name:attrs})
            self.opt.cache_node_attr(new_name, _dag.nodes[new_name])

    def op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # test if two nodes can be fused
        if _pkg.can_contract_edge(u_, v_):
            nodes_to_add = []
            nodes_to_remove = []
            _pkg.contract_edge(u_, v_)
            self._fuse_pair(_dag, u_, v_)
            nodes_to_add.append(u_+"+"+v_)
            nodes_to_remove += [u_, v_]
            return True, nodes_to_add, nodes_to_remove
        else:
            return False, None, None

    def _fuse_pair(self, _dag, u_, v_):
        # print("fuse {} {}".format(u_, v_))
        ### Cache the node attributes in case they will be used when de-fuse
        # SingleLogger().info("\033[94m Fusing pair: {}, {}\033[0m".format(u_, v_))
        if u_ not in self.node_attr_cache:
            self.opt.cache_node_attr(u_, _dag.nodes[u_])
        if v_ not in self.node_attr_cache:
            self.opt.cache_node_attr(v_, _dag.nodes[v_])

        new_name = self.concat_name(u_, v_)
        ### Add new nodes and get the attibute
        if new_name in self.node_attr_cache:
            _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            _dag.add_node(new_name)
            ### Calculate the new attribute
            self.combine_nodes_attr(_dag, new_name, u_, v_)
            ### cache the attribute
            self.opt.cache_node_attr(new_name, _dag.nodes[new_name])

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

    def op_defusion(self, _dag, _pkg: PKGraph, target, components):
        nodes2add = []
        nodes2rm = []

        _pkg.split_node(target, components)
        _, new_node_names = self._defuse_node(_dag, _pkg, target, components)
        nodes2add += new_node_names
        nodes2rm.append(target)
        return True, set(nodes2add), set(nodes2rm)

    def _defuse_node(self, _dag, _pkg, target, components):
        component_names = defuse_nodes_inplace_nx(_dag, _pkg, target, components)
        for new_node_name in component_names:
            self.parse_node_attr(_dag, new_node_name)
        return True, component_names

    def apply(self, s, __dag, __pkg):
        op, target, next_ = s
        ### TODO (huhanpeng): need further add other optimization techiniques
        if op == "+":
            ### Fuse two nodes
            return self.op_fusion(__dag, __pkg, target, next_)
        elif op == "-":
            return self.op_defusion(__dag, __pkg, target, next_)

class _AMPCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        ### AMP predictor
        self.amp_predictor = AMPPredictor(self.opt.clct.para_dict)
        self.token = [">", "<"]
        self.meta_info = self.opt.clct.para_dict

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []
        for n, l in candidates:
            # node heat
            heat = self.opt._get_heat_from_history(n)
            # ### Nodes that have never been fused
            # cat = parse_cat_fine_grained(n)
            # pid = parse_pid_from_name(n)

            ### check if mixed precision can be used for this node
            if self.amp_predictor.is_need_amp(_dag, n):
                search_space.append((">", n, None))
                weights.append(l)
        
        # return [(">", "host1.rank6->FW.resnet50/conv1_conv/Conv2D", None)], [1]
        SingleLogger().info("MP Cost Model init {} strategies.".format(len(search_space)))
        return search_space, weights

    def init_partition(self, G, PKG, initial_partitions_formed) -> int:
        return initial_partitions_formed

    def apply(self, s, __dag, __pkg):
        op, target, _ = s
        nodes_introduced = self.amp_predictor.quantize(__dag, target)
        ### apply this strategy to other GPUs' corresponding operators
        ### we assume data parallel, use the same model
        on_other_ranks = self.opt._debug_convert_to_the_other_machine(target)
        for target in on_other_ranks:
            nodes_introduced += self.amp_predictor.quantize(__dag, target)
        return True, nodes_introduced, []

class _TensorFusionCM(_BaseCostModel):
    ''' This is a cost model for HOROVOD tensor fusion
    '''
    def __init__(self, opt):
        super().__init__(opt)
        self.fusion_threshold_mb = 64 * 1024 * 1024
        self.cycle_time_ms = 3.5
        self.token = ["o"]

        self.meta_info = self.opt.clct.para_dict
    
    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []
        
        for node in _dag.nodes():
            if "Comm" in node and "Sync" in node:
                pass

        self.meta_info.standarize_name()

        return search_space, weights
    
    def apply(self, s, __dag, __pkg):
        _, _fusion_threshold_mb, _cycle_time_ms = s
              
class CostModelManager:
    def __init__(self, opt, cost_models):
        self.cost_model_list = [
            # _XLACostModel(opt, cost_models),
            _AMPCostModel(opt),
        ]
        self.mem_model_list = []
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
    def __init__(self, collector, cost_models, memory_budget=None):
        self.clct = collector
        self.platform = self.clct.platform
        self.comm_backend = self.clct.comm_backend

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

        self.base_cost, _, mem_usage = self.evaluate(self.dag)
        SingleLogger().info("Start to search, the original iteration time is %f" % self.base_cost)

        ### Used to cache the node attribtue
        self.node_attr_cache = AttrCache()

        ### Budget, in GB
        self.memory_budget = memory_budget if memory_budget is not None else 1

        ### Some hyper-parameter
        self.enable_defusion = False

        ### DEBUG ONLY
        self.cost_model_error = []

        self.cst_md_mng = CostModelManager(self, cost_models)

    def relabel_dag_node(self, _dag) -> nx.DiGraph:
        def relabel_func(old_label):
            if ("BW" in old_label or "FW" in old_label or "Comm" in old_label) and "^" not in old_label:
                layer_name = parse_layer_name(old_label)
                layer_pid = parse_pid_from_name(old_label)
                # if layer_pid not in self.cost_models or layer_name not in self.cost_models[layer_pid].graph_def_util.operation_names:
                #     return "DEL~"+old_label
                # TODO (huhanpeng): different pids share the same index
                if "Comm" in old_label and layer_name in self.name2index and layer_pid in self.name2index[layer_name]:
                    layer_index = self.name2index[layer_name][layer_pid]
                    new_label = ("[%d]"%layer_index).join(old_label.split(layer_name))
                    return new_label

                layer_index = len(self.index2name)
                self.index2name[layer_index] = layer_name
                self.index2pid[layer_index] = layer_pid
                if layer_name not in self.name2index:
                    self.name2index[layer_name] = {}
                self.name2index[layer_name][layer_pid] = layer_index
                new_label = ("[%d]"%layer_index).join(old_label.split(layer_name))
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
    
    def _debug_convert_to_the_other_machine(self, name_):
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
                new_names.append(self._debug_convert_to_the_other_machine(sub_name))
            new_names = list(np.array(new_names).T)
            return ["+".join(ns) for ns in new_names]

    def _get_original_name_pid_from_index(self, name_):
        if args_.relabel:
            index = self._parse_index_from_name(name_)
            return self.index2name[index], self.index2pid[index]
        else:
            return parse_layer_name(name_), parse_pid_from_name(name_)

    def get_node_attr(self, n, attr_):
        if attr_ in self.node_attr_cache[n]:
            return self.node_attr_cache[n][attr_]
        else:
            return 0

    def cache_node_attr(self, n, attrs):
        ### TODO (huhanpeng): need .copy() ???
        self.node_attr_cache[n] = attrs

    def evaluate(self, _dag, _filename=None):
        # t = time.time()
        ### input _dag is a dependency graph, using the replayer to get the simulated traces and execution graph
        ### Return the iteration time and the execution graph
        _output = False if _filename is None else True
        replayer = Replayer(dag=_dag, _step_num=1, 
                leaf_dirs=self.clct.all_prefix_list(), 
                dump_path=self.clct.pm.path,
                comm_backend=self.clct.comm_backend,
                byteps_graph=self.clct.byteps_graph)
        step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(None, _ouput=_output, _filename=_filename).values()]
        # print("Evaluate time {}".format(time.time() - t))

        return max(step_end_time_ms), replayer.exct_dag, 0

    def candidate_selection(self, GS, topk=None, critical_path=None):
        ### Select nodes on the critical path of the execution graph as the candidates
        ### Return the candidates and the revised dependency graph
        if critical_path is None:
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

    def _get_heat_from_history(self, node):
        heat = 0
        for (h, t) in self.heat_history[node]:
            if h is not None:
                heat += h * np.exp(-0.5*(self.step - t - 1))
        return heat

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        ### TODO (huhanpeng): currently only consider fusion
        ###             Need to add quantization
        search_space = []
        weights = []

        ### OOM
        if self.mem_usage > self.memory_budget:
            for _cost_model in self.cst_md_mng.mem_model_list:
                ss_, wei_ = _cost_model.init_search_space(candidates, _dag, _pkg)
                search_space += ss_
                weights += wei_
            if len(search_space) == 0:
                SingleLogger().WARN("No optimization strategy to reduce memory usage: {} > {}".format(self.mem_usage, self.memory_budget))

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

        ### TODO (huhanpeng): is shallow copy is enough ???
        __dag = _dag.copy()
        __pkg = _pkg.copy()

        if isinstance(strategy, list):
            nodes_introduced = set()
            nodes_removed = set()
            for s in strategy:
                op, target, next_ = s
                success, n_introduced, n_removed = self.cst_md_mng.strategy2model[op].apply(s, __dag, __pkg)
                if not success:
                    raise OptApplyStrategyError
                nodes_introduced.update(n_introduced)
                nodes_removed.update(n_removed)
        else:
            op, target, next_ = strategy
            success, nodes_introduced, nodes_removed = self.cst_md_mng.strategy2model[op].apply(strategy, __dag, __pkg)
            if not success:
                raise OptApplyStrategyError
        return __dag, __pkg, nodes_introduced, nodes_removed

    def pick_strategy(self, search_space, weights=None, invalid_strategies=None):
        ### TODO (huhanpeng): need some priority/heuristic
        valid_search_space = []
        valid_weights = []
        if invalid_strategies:
            for st_idx, st in enumerate(search_space):
                if st not in invalid_strategies:
                    valid_search_space.append(st)
                    if weights is not None:
                        valid_weights.append(weights[st_idx])
        else:
            valid_search_space = search_space.copy()
            if weights is not None:
                valid_weights = weights.copy()
        if not valid_search_space:
            raise OptNoValidStrategyError
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights - np.min(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        if valid_weights is None:
            return random.choice(valid_search_space)
        else:
            # return self.weighted_choice(valid_search_space, weights)
            return random.choices(valid_search_space, weights=valid_weights, k=1)[0]

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
        assert False, "Shouldn't get here"

class MCMCOptimizer(Optimizer):
    ''' Markov Chain Monte Carlo algorithm'''
    def __init__(self, *args, **kwargs):
        super(MCMCOptimizer, self).__init__(*args, **kwargs)
        self.heat_history = {}
        if args_.heat_window_size:
            self.heat_window_size = args_.heat_window_size
        else:
            self.heat_window_size = 5

    def __dump_cluster_mapping(self, dag, output_path):
        cluster_index = 0
        with open(output_path, "w") as f:
            for node in dag.nodes():
                if "+" in node:
                    orig_names, _ = self._get_original_name_pid_from_fused_node(node)
                    for orig_node_name in orig_names:
                        f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1

    def search(self, graph_cache=os.path.join(ROOT_PATH, "graph_cache.pickle"), step_size=1):
        ### TODO (huhanpeng): is shallow copy is enough ???
        if args_.ckpt and graph_cache is not None and os.path.isfile(graph_cache):
            with open(graph_cache, "rb") as f:
                G, PKG, node_attr_cache, initial_partitions_formed = pickle.load(f)
                self.node_attr_cache = node_attr_cache
            SingleLogger().info("Reading init graph from cache.")
        else:
            G = self.dag.copy()
            PKG = PKGraph(G)
            # # randomly contract edges if possible
            # k = int(len(G.edges()) * init_edges_to_contract)
            initial_partitions_formed = 0
            for node in G.nodes():
                if node not in self.node_attr_cache:
                    self.cache_node_attr(node, G.nodes[node])

            for _cost_model in self.cst_md_mng.cost_model_list:
                initial_partitions_formed = _cost_model.init_partition(G, PKG, initial_partitions_formed)
            if graph_cache:
                with open(graph_cache, "wb") as f:
                    pickle.dump([G, PKG, self.node_attr_cache, initial_partitions_formed], f)
                SingleLogger().info("Graph cache dumped to {}.".format(graph_cache))
        if "BPF_DUMP_INIT_CLUSTER_TO" in os.environ:
            self.__dump_cluster_mapping(G, os.environ["BPF_DUMP_INIT_CLUSTER_TO"])
        # initialize heat history
        for node in G.nodes:
            self.heat_history[node] = [(0, 0)] * self.heat_window_size
        SingleLogger().info("="*20 + " Search Starts " + "="*20)
        SingleLogger().info("Successfully initialized {} partitions.".format(initial_partitions_formed))

        if "BPF_DUMP_INIT_GRAPH_TO" in os.environ:
            bpf_dump_init_graph_to = os.environ["BPF_DUMP_INIT_GRAPH_TO"]
        else:
            bpf_dump_init_graph_to = None
        cost, exct_dag, self.mem_usage = self.evaluate(G, _filename=bpf_dump_init_graph_to)

        trajectory = []
        candidates, _ = self.candidate_selection(G, topk=None, critical_path=self.wrap_critical_path(exct_dag))
        search_space, weights = self.init_search_space(candidates, G, PKG)
        SingleLogger().info("\033[94m # of candidates: {}\033[0m".format(len(candidates)))
        best_cost = cost
        best_strategy = trajectory.copy()
        self.step = 0
        while True:
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
                        candidates, _ = self.candidate_selection(G_star, topk=None)
                        search_space, weights = self.init_search_space(candidates, G_star, PKG_star)
                        invalid_strategies = set()
                        continue
                    try:
                        G_star, PKG_star, nodes_introduced, nodes_removed = self.apply_strategies(G_star, PKG_star, strategy)
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
                    if successful_strategies < step_size:
                        candidates, _ = self.candidate_selection(G_star, topk=None)
                        search_space, weights = self.init_search_space(candidates, G_star, PKG_star)
                    invalid_strategies = set()
                    msg = "\033[94m" + "Strategy ({}, {}, {}) successfully applied.".format(*strategy)
                    if len(msg) > 200:
                        msg = msg[:200] + "... successfully applied."
                    SingleLogger().info(msg + "\033[0m")
                ### Start to replay
                if self.step % 20 == 0:
                    cost_star, exct_dag, mem_usage = self.evaluate(G_star, _filename=os.path.join(ROOT_PATH, "searched_graph/{}.json".format(self.step)))
                else:
                    cost_star, exct_dag, mem_usage = self.evaluate(G_star)
                # if step < 200:
                #     MCMC_BETA = 1
                # else:
                #     MCMC_BETA = args.mcmc_beta
                SingleLogger().info("\033[94m Step: {}, Orig cost: {}, New cost: {} \033[0m".format(self.step, cost, cost_star))
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
                        node_history.insert(0, (cost - cost_star, self.step))
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

                if self.accept_or_not(cost, cost_star):
                    invalid_strategies = set()
                    # op, target, next_ = strategy
                    # if op == "+":
                    #     SingleLogger().info("Fuse %s and %s" % (target, next_))
                    # elif op == "-":
                    #     SingleLogger().info("De-fuse %s" % (target))
                    # else:
                    #     raise ValueError("Invalid graph transformation operation: {}".format(op))

                    ### generate history for new nodes
                    combined_history.insert(0, (cost - cost_star, self.step))
                    combined_history.pop()
                    for node in strategy_introduced_nodes:
                        self.heat_history[node] = combined_history.copy()

                    G = G_star
                    PKG = PKG_star
                    cost = cost_star
                    trajectory += strategy_history_in_step
                    self.mem_usage = mem_usage

                    ### clean up heat history for removed nodes
                    for node in strategy_removed_nodes:
                        if node in self.heat_history:
                            self.heat_history.pop(node)

                    ### Cache the best strategy
                    if cost < best_cost:
                        best_cost = cost
                        best_strategy = trajectory.copy()
                    ### Init new search space
                    candidates, _ = self.candidate_selection(G, topk=None, critical_path=self.wrap_critical_path(exct_dag))
                    search_space, weights = self.init_search_space(candidates, G, PKG)
                    break
            SingleLogger().info("\033[94m" + "Speedup to the origin: %6.4f %%"%(100 * (self.base_cost - cost) / self.base_cost) + "'\033[0m'")
            SingleLogger().info("\033[94m" + "Best speedup: %d th acception, speed up to the origin: %6.4f %%"%(len(best_strategy), 100 * (self.base_cost - best_cost) / self.base_cost) + "'\033[0m'")
            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {}".format(100 * (self.base_cost - best_cost) / self.base_cost) + "\n")
            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": best_strategy}, f)
 
    def accept_or_not(self, cost, new_cost):
        # prob = min(1, (math.exp(beta * (cost - new_cost))))
        if cost > new_cost:
            SingleLogger().info("\033[92m" + "Accept a better action, orig cost: {}, new cost: {}".format(cost, new_cost) + "\033[0m")
            return True
        else:
            prob = math.exp(MCMC_BETA * (cost - new_cost))
            r = random.random()
            if r < prob:
                SingleLogger().info("\033[92m" + "Accept a worse action with {} < {} ".format(r, prob) + "\033[0m")
                return True
            else:
                SingleLogger().info("\033[93m" + "Rejected a worse action with {} >= {} ".format(r, prob) + "\033[0m")
                return False

class MCTSOptimizer(Optimizer):
    ''' Monte Carlo Tree Search '''
    def __init__(self, *args, ucb_type="AVG", no_mutation=False, **kwargs):
        super(MCTSOptimizer, self).__init__(*args, **kwargs)
        self.loop_cnt = 0
        self.GS_root = None
        self.opt_GS = None
        self.ucb_type = ucb_type
        if self.ucb_type != "MAX" and self.ucb_type != "AVG":
            raise ValueError("UCB type should be MAX or AVG, but {} is given.".format(self.ucb_type))
        self.no_mutation=no_mutation

    def search(self):
        ### Initialize the root graph state
        self.GS_root = GraphState(depth=0)
        self.GS_root.strategy = []

        while self.check_loop_time() and self.check_loop_num():
            GS = self.tree_policy(self.GS_root)
            reward = self.default_policy(GS)
            SingleLogger().info("Speedup to the origin %6.4f %%"%(100 * reward))
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
            node_string = "  %5.4f %% "%(GS.quality * 100) if GS.quality >= 0 else " -%5.4f %% "%(-GS.quality * 100)
            sys.stdout.write(node_string)
            assert len(node_string) == LENOFNODE
            if GS.childs is None:
                return
            for idx, child in enumerate(GS.childs):
                if idx > 0:
                    sys.stdout.write("\n{}".format(" "*(LENOFNODE + LENOFARROW//2)))
                    sys.stdout.write("{}".format(" "*((LENOFNODE + LENOFARROW) * (GS.depth - cnt))))
                    sys.stdout.write("{}".format(("|" + " " * (LENOFNODE + LENOFARROW - 1))*(cnt)))
                    sys.stdout.write("{}".format("|" if idx < (len(GS.childs) -1) else "\\"))
                    sys.stdout.write("{}".format("-"*(LENOFARROW - LENOFARROW//2 - 1)))
                else:
                    sys.stdout.write("{}".format('-'*LENOFARROW))
                if idx < (len(GS.childs) -1):
                    next_cnt = cnt + 1
                else:
                    next_cnt = cnt
                iter_print(child, next_cnt)

        iter_print(self.GS_root, 0)
        sys.stdout.write("\n")

    def show_opt_strategies(self):
        SingleLogger().info("Best speedup: %d th layer, speed up to the origin: %6.4f %%"%(len(self.opt_GS.strategy), 100 * self.opt_GS.quality))

    def check_loop_num(self):
        self.loop_cnt += 1
        if self.loop_cnt > MAX_LOOP:
            return False # End
        else:
            return True # continue

    def check_loop_time(self):
        return True # continue

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
                c = GS_c.quality + UCB_GAMMA * math.sqrt((2 * math.log(GS.visit_cnt)) / GS_c.visit_cnt)
            elif self.ucb_type == "AVG":
                c = GS_c.quality / GS_c.visit_cnt + UCB_GAMMA * math.sqrt((2 * math.log(GS.visit_cnt)) / GS_c.visit_cnt)
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
            GS.space = [[action, 0] for action in search_space] ### The integer value is used as a counter

    def terminal(self, GS):
        self.check_search_space(GS)
        if GS.depth > MAX_TREE_DEPTH or len(GS.space) == 0:
            return True
        else:
            return False
