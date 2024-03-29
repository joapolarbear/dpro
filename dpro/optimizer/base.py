from multiprocessing import Value
import networkx as nx
import random
import numpy as np
import pickle
import ujson as json

from ..replay import Replayer
from ..trace_utils import *
from ..dag_utils import *
from ..base import bcolors
from ..arg_utils import SingleArg

from ..memory import MemoryEstimator
from ..memory.cost_model import IncreasingBatchSizeCostModel, MemoryGraphPass

from ..cost_model.base import OptApplyStrategyError, OptNoValidStrategyError
from ..cost_model._xla.pk_graph import PKGraph
from ..cost_model.mixed_precision import AMPGraphPass
from ..cost_model.tensor_fusion import TensorFusionGraphPass

args_ = SingleArg().args
try:
    from cost_model.op_fusion import XLAGraphPass
    # import horovod.tensorflow as hvd
except:
    pass

ROOT_PATH = os.path.join(
    args_.workspace if args_.workspace else args_.path, ".opt_ws")

class CostModelManager:
    def __init__(self, opt):
        self.cost_model_list = []
        self.strategy2model = {}
        # if "amp" in args_.sub_option:
        #     self.cost_model_list.append(AMPGraphPass(opt))
        if "tensor_fusion" in args_.sub_option or "tsfs" in args_.sub_option:
            self.cost_model_list.append(TensorFusionGraphPass(opt, ROOT_PATH))
        if "xla" in args_.sub_option:
            self.cost_model_list.append(XLAGraphPass(opt, ROOT_PATH))
        
        if "^memory" in args_.sub_option:
            self.mem_model_list = []
        else:
            self.cost_model_list.append(IncreasingBatchSizeCostModel(opt))
            self.mem_model_list = [MemoryGraphPass(opt)]

        if len(self.cost_model_list) + len(self.mem_model_list) == 0:
            SingleLogger().error("No optimization techniques for computation. ")
            # self.cost_model_list = [
            #    XLAGraphPass(opt, ROOT_PATH),
            #    AMPGraphPass(opt),
            # ]

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
        self.cord_pid = gen_pid_name(self.comm_backend, None, None)

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
            os.makedirs(ROOT_PATH)
        if not os.path.exists(os.path.join(ROOT_PATH, "searched_graph")):
            os.makedirs(os.path.join(ROOT_PATH, "searched_graph"))
        self.ckpt_dir = os.path.join(ROOT_PATH, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.spec_dir = os.path.join(ROOT_PATH, "spec")
        if not os.path.exists(self.spec_dir):
            os.makedirs(self.spec_dir)

        ### Memory Budget, in GB
        self.memory_budget = args_.memory_budget

        ### Some hyper-parameter
        self.enable_defusion = False

        ### DEBUG ONLY
        self.cost_model_error = []

        self.cst_md_mng = CostModelManager(self)
        self.opfs_pass = self.cst_md_mng.strategy2model.get("+", None)
        self.tsfs_pass = self.cst_md_mng.strategy2model.get("++", None)

        self.use_heat = True

        if self.comm_backend == "NCCL" and self.tsfs_pass is not None:
            self.tsfs_pass.enable_partition = False

        self.base_cost, self.exct_dag, self.base_mem_usage = self.evaluate(
            self.dag, _path=os.path.join(ROOT_PATH, "searched_graph/base.json"))

        self.disable_symmetry = False
        self.disable_partial_replay = False

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
        if self.disable_symmetry:
            return []
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
        # assert "+" not in name_, name_
        if args_.relabel:
            index = self._parse_index_from_name(name_)
            return self.index2name[index], self.index2pid[index]
        else:
            return parse_op_name(name_), parse_pid_from_name(name_)

    def evaluate(self, _dag, _path=None, _crit_filename=None,
        recd_topo_order=False, partial=False, name2mapping_fn=None,
        visual_bw2comm=False):

        if self.tsfs_pass is not None and self.clct.byteps_graph is not None:
            self.clct.byteps_graph.grp_part_id2server = self.tsfs_pass.tsfs_state.grp_part_id2server
        # t = time.time()
        ### input _dag is a dependency graph, using the replayer to get the simulated traces and execution graph
        ### Return the iteration time and the execution graph
        _output = False if _path is None else True
        replayer = Replayer(dag=_dag, _step_num=1,
                            leaf_dirs=self.clct.all_prefix_list(),
                            dump_path=self.clct.pm.path,
                            comm_backend=self.comm_backend,
                            byteps_graph=self.clct.byteps_graph,
                            infi_para_update=args_.update_infi_para,
                            recd_topo_order=recd_topo_order,
                            partial=partial,
                            name2mapping_fn=name2mapping_fn
                            )
        step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(
            None, _output=_output, _path=_path, verbose=False).values()]
        
        if visual_bw2comm:
            replayer.paint_bw_comm_depend()
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
        if self.clct.byteps_graph is not None:
            self.clct.byteps_graph.grp_part_id2server = None

        ### Whether to record the topological order
        if recd_topo_order:
            return max(step_end_time_ms), replayer.exct_dag, estimated_memory_usage, replayer.ret_topo_ord()
        else:
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

    def _combine_weight(self, weight: float, heat: float) -> float:
        ''' Tune the weight of a strategy with the heat of involved nodes
        '''
        # return l * (0.05 + heat)
        if heat <= -1:
            SingleLogger().error(bcolors.CRED + "Return negative weight {}".format(heat) + bcolors.ENDC)
        return heat + 1
        # return 1

    def _get_heat_from_history(self, node):
        '''
            Return heat based on the heat history
            A node's heat decreases as the seach process goes on
        '''
        if not self.use_heat:
            return 1
        heat = 0
        HEAT_DECREASE_RATE = 1
        cnt = 0
        for (h, t) in self.heat_history[node]:
            if h is not None and h != 0:
                heat += (np.exp(h) - 1) / (HEAT_DECREASE_RATE * (self.step + 1 - t))
                cnt += 1
        if cnt > 0:
            heat = heat / float(cnt)
        if heat < -1:
            raise ValueError(self.heat_history[node])
        return heat

    def update_fusion_heat_history(self, is_accept, nodes_to_rm, nodes_to_add, fusion=True):
        ''' Update heat history for origal nodes and introduced nodes
            * The heat is for operator fusion
            ** Record `Delta T` as heat for fusion, `- Delta T` for de-fusion
        '''
        new_heat = self.cur_cost - self.cost_star if fusion else self.cost_star - self.cur_cost
        ### Update heat history
        combined_history_list = [None] * self.heat_window_size
        for node in nodes_to_rm:
            if node in self.heat_history:
                node_history = self.heat_history[node]
                for idx, (h, _) in enumerate(node_history):
                    if combined_history_list[idx] is None:
                        combined_history_list[idx] = [h]
                    elif h is None:
                        continue
                    else:
                        combined_history_list[idx].append(h)
                ### update the heat of the original nodes
                node_history.insert(0, (new_heat, self.step))
                node_history.pop()
        combined_history = []
        for h_list in combined_history_list:
            if h_list is not None:
                h_avg = np.average(h_list)
                combined_history.append((h_avg, self.step))
            else:
                combined_history.append((None, None))
        
        if is_accept:
            ### generate history for new nodes
            combined_history.insert(0, (new_heat, self.step))
            combined_history.pop()
            for node in nodes_to_add:
                self.heat_history[node] = combined_history.copy()
        
            ### clean up heat history for removed nodes
            for node in nodes_to_rm:
                if node in self.heat_history:
                    self.heat_history.pop(node)

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

        if len(search_space) == 0:
            if self.mem_usage > self.memory_budget:
                SingleLogger().warn("Ignore OOM, still optimize the training speed")
            else:
                SingleLogger().info("Estimated memory usage does not exceed memory budget: {:.2f}GB < {:.2f}GB".format(
                    self.mem_usage, self.memory_budget))

            ### cm_start_end and cm_weight_dict are used to balance the weights across
            ### different optimization passes
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
                SingleLogger().debug("Weight sum for {}: {}".format(model_type, sum(wei_)))
                weights += wei_

            ### assign a specific portion to each strategy, according to step size
            ### TEMP: xla : tensor_fusion = 2:1
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
        ''' Some strategies may not be accepted, and some Passes are stateful
            * If a strategy is accepted, change the interal state of those Passes
            * Otherwise, keep it the same
        '''
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
            valid_weights = valid_weights  / np.linalg.norm(valid_weights)
            # valid_weights = valid_weights - np.min(valid_weights)
            # valid_weights = valid_weights / np.sum(valid_weights)
            try:
                st_idx = random.choices(valid_search_space_idx, weights=valid_weights, k=1)[0]
            except:
                raise
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

    def _op_to_coarse_grained_comm_time(self, bw_op, _dag, topo_order=None):
        ref_pid = parse_pid_from_name(bw_op)
        comm_t = 0
        def ret_comm_time(comm_op):
            avg_sum = _dag.nodes[comm_op]["avg"]
            for comm_succ in _dag.successors(comm_op):
                _pid = parse_pid_from_name(comm_succ)
                if "Comm" in comm_succ and ref_pid == _pid:
                    avg_sum += ret_comm_time(comm_succ)
            return avg_sum
        if self.comm_backend == "NCCL":
            for _succ in _dag.successors(bw_op):
                if "Comm" in _succ:
                    comm_t += ret_comm_time(_succ)
            return comm_t
        elif self.comm_backend == "BYTEPS":
            ### PS
            local_dfg = self.clct.dag
            bw_op_std_names = [parse_rawname(_op) for _op in bw_op.split("+")]
            comm_succ_list = []
            for bw_op_std_name in bw_op_std_names:
                comm_succ_list += [succ_ for succ_ in local_dfg.successors(bw_op_std_name) if "Comm" in succ_]
            if len(comm_succ_list) == 0:
                return None
            else:
                comm_delay_in_ms = 0
                for comm_succ in comm_succ_list:
                    succ_list = [succ_ for succ_ in local_dfg.successors(comm_succ) if "UPDATE_" in succ_]
                    assert len(succ_list) == 1, (bw_op, comm_succ, succ_list)
                    update_op = gen_long_name(ref_pid, succ_list[0])
                    topo_order = dict(topo_order)
                    bw_op_et = topo_order[bw_op] + _dag.nodes[bw_op]["avg"] * 1000
                    try:
                        update_op_st = topo_order[update_op]
                    except:
                        update_op_st = None
                        for _update_op in topo_order:
                            if "UPDATE_" not in _update_op:
                                continue
                            if update_op in _update_op:
                                update_op_st = topo_order[_update_op]
                                break
                        if update_op_st is None:
                            SingleLogger().warn("No update nodes found")
                            tmp_list = [name.split("->")[1] for name in topo_order.keys() if "UPDATE_" in name]
                            print(bw_op, comm_succ, update_op)
                            import code
                            code.interact(local=locals())
                            raise
                    comm_delay_in_ms = max(comm_delay_in_ms, (update_op_st - bw_op_et) / 1000)
                return comm_delay_in_ms

                # traces_0.rank1->UPDATE_.Distributed_Push_Pull/truediv_212
        else:
            raise ValueError(self.comm_backend)
            
    def estimate_time_involved_nodes(self, _dag, all_involved_nodes, dump_path):
        if self.disable_partial_replay:
            sub_graph = _dag
        else:
            sub_graph = _dag.subgraph(all_involved_nodes)
        sub_graph = _dag.subgraph(all_involved_nodes)
        replayer = Replayer(dag=sub_graph, _step_num=1,
                            leaf_dirs=self.clct.all_prefix_list(),
                            dump_path=self.clct.pm.path,
                            comm_backend=self.comm_backend,
                            byteps_graph=self.clct.byteps_graph,
                            infi_para_update=args_.update_infi_para,
                            recd_topo_order=False,
                            partial=True
                            )
        step_end_time_list_in_us = replayer.replayAndDelay(
            None, _output=True if dump_path is not None else False,
            _path=dump_path, verbose=False)
        return max(step_end_time_list_in_us.values()) / 1000

    def estimate_time_related_to_comp(self, comp_ops, _dag, dump_path=None):
        involved_comm_nodes = set()
        involved_bw_nodes = set(comp_ops)
        involved_update_nodes = set()
        involved_tensors = set()
        for comp_op in comp_ops:
            for succ in _dag.successors(comp_op):
                if "Comm." in succ:
                    tensor_name = parse_op_name(succ)
                    involved_tensors.add(tensor_name)
        
        for _node in _dag.nodes():
            for tensor_name in involved_tensors:
                if "Comm.{}.".format(tensor_name) in _node:
                    involved_comm_nodes.add(_node)
                    for pred in _dag.predecessors(_node):
                        if "BW" in pred:
                            involved_bw_nodes.add(pred)
                    for succ in _dag.successors(_node):
                        if "UPDATE_" in succ:
                            involved_update_nodes.add(succ)

        all_involved_nodes = involved_bw_nodes.union(involved_comm_nodes).union(involved_update_nodes)
        return self.estimate_time_involved_nodes(_dag, all_involved_nodes, dump_path)

    def estimate_time_related_to_comm(self, tensor_names, _dag, dump_path=None):
        involved_comm_nodes = set()
        involved_bw_nodes = set()
        involved_update_nodes = set()
        for _node in _dag.nodes():
            for tensor_name in tensor_names:
                if "Comm.{}.".format(tensor_name) in _node:
                    involved_comm_nodes.add(_node)
                    for pred in _dag.predecessors(_node):
                        if "BW" in pred:
                            involved_bw_nodes.add(pred)
                    for succ in _dag.successors(_node):
                        if "UPDATE_" in succ:
                            involved_update_nodes.add(succ)

        all_involved_nodes = involved_bw_nodes.union(involved_comm_nodes).union(involved_update_nodes)
        # print(tensor_names, len(involved_comm_nodes), len(involved_bw_nodes), len(involved_update_nodes))
        # print(involved_comm_nodes)

        return self.estimate_time_involved_nodes(_dag, all_involved_nodes, dump_path)
    
    def estimate_comm_time_via_replay(self, tensor_name, _dag):
        involved_comm_nodes = [_node for _node in _dag.nodes() if "Comm.{}.".format(tensor_name) in _node]
        involved_bw_nodes = set()
        for comm_op in involved_comm_nodes:
            involved_bw_nodes = involved_bw_nodes.union([pred for pred in _dag.predecessors(comm_op) if "BW" in pred])
        all_involved_nodes = involved_bw_nodes.union(involved_comm_nodes)
        sub_graph = _dag.subgraph(all_involved_nodes)
        # dump_path = "partial_replay.json"
        dump_path = None
        _output = False if dump_path is None else True
        from replay import Replayer
        replayer = Replayer(dag=sub_graph, _step_num=1,
                            leaf_dirs=self.clct.all_prefix_list(),
                            dump_path=self.clct.pm.path,
                            comm_backend=self.comm_backend,
                            byteps_graph=self.clct.byteps_graph,
                            infi_para_update=args_.update_infi_para,
                            recd_topo_order=False,
                            partial=True
                            )
        step_end_time_list_in_us = replayer.replayAndDelay(
            None, _output=_output, _path=dump_path, verbose=False)
        for bw_op in involved_bw_nodes:
            pid = parse_pid_from_name(bw_op)
            step_end_time_list_in_us[pid] -= (_dag.nodes[bw_op]["avg"]) * 1000
        return max(step_end_time_list_in_us.values()) / 1000