from multiprocessing import Value
import os
import re
import time
import json
import pickle
from tqdm import tqdm

from .base import Optimizer, args_, ROOT_PATH
from ..logger_utils import SingleLogger
from ..cost_model._xla.pk_graph import PKGraph, PKGraphCycleError
from ..base import bcolors
from ..trace_utils import gen_long_name, parse_allinfo_from_name, parse_cat_fine_grained, \
    parse_cat_from_name, parse_op_name, parse_pid_from_name, CatName, parse_rawname

DEBUG = False
DISABLE_LAYER_VIEW=False
DISABLE_CRITICAL_PATH=False
DISABLE_SYMMETRY=False
DISABLE_PARTIAL_REPLAY=False

class DPState:
    def __init__(self, opt):
        self.p_e = [] # end time of compute operators
        self.p_d = [] # execution time of compute operators
        self.q_e = [] # end time of communication operators
        self.q_d = [] # execution time of communication operators
        self.q_m = [] # tensor size of communication operators
        self.node_num = 0
        self.opt = opt
    
    def add_comp_op(self, comp_op, exct_time, _dag):
        self.p_d.append(exct_time)
        tensor_set = self.opt.comm_succs_of_comp_in_op_name(comp_op, _dag)
        if self.node_num == 0:
            self.p_e.append(exct_time)
            if len(tensor_set) > 0:
                tensor_time = self.opt._op_to_coarse_grained_comm_time(comp_op, _dag, self.opt.topo_order)
                self.q_d.append(tensor_time)
                self.q_e.append(self.p_e[-1] + tensor_time)
                tensor_size = sum([self.opt.clct.para_dict.tensor_grp_size(tensor) for tensor in tensor_set])
                self.q_m.append(tensor_size)
            else:
                self.q_d.append(0)
                self.q_e.append(0)
                self.q_m.append(0)
        else:
            self.p_e.append(self.p_e[-1] + exct_time)
            if len(tensor_set) > 0:
                tensor_time = self.opt._op_to_coarse_grained_comm_time(comp_op, _dag, self.opt.topo_order)
                self.q_d.append(tensor_time)
                self.q_e.append(max(self.q_e[-1], self.p_e[-1]) + tensor_time)
                tensor_size = sum([self.opt.clct.para_dict.tensor_grp_size(tensor) for tensor in tensor_set])
                self.q_m.append(tensor_size)
            else:
                self.q_d.append(0)
                self.q_e.append(self.q_e[-1])
                self.q_m.append(0)
        self.node_num += 1
    
    def update_state_tsfs(self, tsfs_time):
        self.q_d[-1] = tsfs_time

        # self.q_e[-1] = max(self.p_e[-1], self.q_e[-3]) + self.q_d[-1]
        ### q_e[-3] may be none because it is fused with q_e[-2]
        ### we need to find the first pred comm that q_e is not none
        pred_comm_idx = -3
        while (- pred_comm_idx) <= len(self.q_e) and self.q_e[pred_comm_idx] is None:
            pred_comm_idx -= 1
        if (- pred_comm_idx) > len(self.q_e):
            self.q_e[-1] = self.p_e[-1] + self.q_d[-1]
        else:
            self.q_e[-1] = max(self.p_e[-1], self.q_e[pred_comm_idx]) + self.q_d[-1]

        self.q_m[-1] += self.q_m[-2]
        self.q_e[-2] = None
        self.q_d[-2] = None
        self.q_m[-2] = None
    
    def update_state_opfs(self, opfs_time):
        self.p_d[-1] = opfs_time
        self.p_e[-1] = self.p_e[-2] + self.p_d[-1] - self.p_d[-2]
        self.p_d[-2] = None
        self.p_e[-2] = None
        if self.q_e[-2] is None:
            ### q_n-1 and q_n has been fused
            # TODO (HHP): q_n-2 may also has been fused
            ### find the first pred comm that q_e is not none
            pred_comm_idx = -3
            while (- pred_comm_idx) <= len(self.q_e) and self.q_e[pred_comm_idx] is None:
                pred_comm_idx -= 1
            if (- pred_comm_idx) > len(self.q_e):
                self.q_e[-1] = self.p_e[-1] + self.q_d[-1]
            else:
                self.q_e[-1] = max(self.p_e[-1], self.q_e[pred_comm_idx]) + self.q_d[-1]
        else:
            self.q_e[-2] = max(self.p_e[-1], self.q_e[-3]) + self.q_d[-2]
            self.q_e[-1] = self.q_e[-2] + self.q_d[-1]
        
    def update_state_only_ts_part(self, ts_part_time):
        self.q_d[-1] = ts_part_time
        self.q_e[-1] = max(self.p_e[-1], self.q_e[-2]) + self.q_d[-1]
 

class DPOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super(DPOptimizer, self).__init__(*args, **kwargs)

        ### Map from the original op to the current op in the DFG
        ### E.g, a and b has been fused to be a+b, then there are two mappings, 
        #   a -> a+b and b => a+b
        ### Note the names of these ops are in the long op_long_name format (with pid)
        ### See `current_op_in_graph`
        self.op2cur_op = {}
        
        self.use_heat = False

        if self.opfs_pass is not None:
            self.opfs_pass.explore_fusion = False
            self.opfs_pass.enable_partition = False
        
        if DISABLE_SYMMETRY:
            self.disable_symmetry = True
        if DISABLE_PARTIAL_REPLAY:
            self.disable_partial_replay = True
    
    def current_op_in_graph(self, original_name) -> str:
        return self.op2cur_op[original_name]
    
    def ret_all_comp_ops(self, topo_ord):
        comp_ops = []
        ref_pid = None
        for node in topo_ord:
            if "+" in node:
                pid, std_name, cat, _ = parse_allinfo_from_name(node.split("+")[0])
            else:
                pid, std_name, cat, _ = parse_allinfo_from_name(node)
            if ref_pid is None:
                ref_pid = pid
            if pid == ref_pid and cat == CatName.OPERATOR.value:
                comp_ops.append(node)
        return comp_ops
    
    '''
    def powerset(self, seq):
        """
            Returns all the subsets of this set. This is a generator.
        """
        if len(seq) <= 1:
            yield []
            yield seq
        else:
            for item in self.powerset(seq[1:]):
                yield item
                yield [seq[0]]+item
                
    def all_possible_strategies(self, preds, op_v, G, PKG):
        search_space, weights = self.init_search_space(
            [(op_u, G.nodes[op_u]["avg"]), (op_v, G.nodes[op_v]["avg"])], G, PKG)
        
        rst_sts = []
        for st_op, u, v in search_space:
            if st_op == "+":
                if u in [op_u, op_v] and v in [op_u, op_v]:
                    rst_sts.append((st_op, u, v))
        print(rst_sts)
        ### If the search_space constains two strategies, s1 and s2
        ### Return a combination of them, i.e., [s1, s2, [s1, s2]]
        return self.powerset(rst_sts)
    '''
    def try_to_apply_opfs(self, u, v, _dag, _pkg, applied_sts, verbose=True):
        if self.opfs_pass is None:
            return
        if verbose:
            SingleLogger().info("Fuse operator {} {}".format(u[:60], v[:60]))
        opfs_sts = ("+", u, v)
        try:
            success, nodes_introduced, nodes_removed = \
                self.opfs_pass.apply(opfs_sts, _dag, _pkg)
        except PKGraphCycleError:
            return False, None

        applied_sts.append(opfs_sts)
        return True, nodes_introduced
    
    def try_to_apply_tsfs(self, tensor_name_u, tensor_name_v, _dag, _pkg, applied_sts, verbose=True):
        if self.tsfs_pass is None:
            return []
        if verbose:
            SingleLogger().info("Fuse tensor {} {}".format(tensor_name_u[:60], tensor_name_v[:60]))
        tsfs_sts = ("++", tensor_name_u, tensor_name_v)
        _, _nodes_introduced, _nodes_removed = \
            self.tsfs_pass.apply(tsfs_sts, _dag, _pkg)
        applied_sts.append(tsfs_sts)

        ### DEBUG
        assert len(set([parse_op_name(op) for op in _nodes_introduced])) == 1, _nodes_introduced

        return _nodes_introduced
    
    def try_to_apply_ts_part(self, tensor_name_list, k_star, _dag, _pkg, applied_sts, verbose=True):
        ''' Partition tensors to k_star pieces
        tensor_name_list: list
        '''
        if self.tsfs_pass is None:
            return []
        if verbose:
            SingleLogger().info("Partition tensor {} to {} pieces".format(tensor_name_list, k_star))
        for tensor_name in tensor_name_list:
            self.tsfs_pass._tensor_partition(_dag, _pkg, tensor_name, k_star)
            applied_sts.append(("tspart", tensor_name, k_star))
    
    def comm_succs_of_comp_in_op_name(self, comp_op, _dag):
        return set([parse_op_name(n) for n in _dag.successors(comp_op) if parse_cat_from_name(n) == CatName.COMM.value])

    def comm_succs_of_comp_in_long_name(self, comp_op, _dag):
        return [n for n in _dag.successors(comp_op) if parse_cat_from_name(n) == CatName.COMM.value]

    def search(self, graph_cache=None):

        SingleLogger().info(bcolors.CGREEN + "Start to search using DP" + bcolors.ENDC)

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
        graph_cache = os.path.join(self.ckpt_dir, "graph_cache.pickle") if graph_cache is None else graph_cache
        if args_.ckpt and graph_cache is not None and os.path.isfile(graph_cache):
            ### TODO (hhp): need to guarantee the consistence of checkpoints of both cost models and DFG states
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.load_ckpt()
            with open(graph_cache, "rb") as f:
                G, PKG, self.step, self.trajectory = pickle.load(f)
            SingleLogger().info("Loading checkpoint of step {}".format(self.step))
            self.cur_cost, self.exct_dag, self.mem_usage, self.topo_order = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"),
                recd_topo_order=True)
        else:
            self.cur_cost, self.exct_dag, self.mem_usage, self.topo_order = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"),
                recd_topo_order=True)
            
            # comp_ops = self.ret_all_comp_ops(topo_ord)
            # cur_comp_idx = 1

            self.step = 0
            self.trajectory = []
            SingleLogger().info("No checkpoint found, search from scratch")

        SingleLogger().info("="*20 + " Search Starts " + "="*20)
        SingleLogger().info(bcolors.CGREEN + "Start to search, the original iteration time is %f, init cost is %f" %
                            (self.base_cost, self.cur_cost) + bcolors.ENDC)

        if args_.test_ts_group_num is not None:
            ### Some test cases
            self.tsfs_pass.init_fuse_tensors = False
            num_grp = int(args_.test_ts_group_num)
            part_size_in_B = 4 * 1024 * 1000
            self.tsfs_pass._apply_grp_num(G, PKG, num_grp)
            
            if self.comm_backend == "BYTEPS":
                self.tsfs_pass._apply_partition_size(G, PKG, part_size_in_B)
                ### Update the tensor_id + part_id to server mapping
                self.tsfs_pass.update_tensor2server()

            _cost_star, _exct_dag_star, _mem_usage_star, topo_order = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "test.json"),
                recd_topo_order=True, visual_bw2comm=True)
            SingleLogger().info(bcolors.CGREEN + "New cost {}".format(_cost_star) + bcolors.ENDC)
            exit(0) 

        def write_time(cost=None):
            cost = self.cur_cost if cost is None else cost
            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {},{},{}".format(
                    self.step,
                    100 * (self.base_cost - cost) / self.base_cost,
                    100 * (self.base_cost - cost) / self.base_cost) + "\n")

        def display_and_ckpt():
            SingleLogger().info(bcolors.CBLUE + "Step: %d - Current speedup to the origin: %6.4f %%" % (self.step,
                                                                                                        100 * (self.base_cost - self.cur_cost) / self.base_cost) + bcolors.ENDC)
            write_time()

            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": self.trajectory}, f)

            ### Save checkpoints by default
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.checkpoint()
            with open(graph_cache, "wb") as f:
                pickle.dump([G, PKG, self.step, self.trajectory], f)
            
            if self.opfs_pass is not None:
                self.opfs_pass._dump_cluster_mapping(
                G, os.path.join(self.spec_dir, "cluster_mapping.txt"), partition=True)

            if self.tsfs_pass is not None:
                self.tsfs_pass.dump_tensor_grp_mapping()
                self.tsfs_pass.dump_tensor_partitions()

        def ret_standar_names(_long_name):
            ''' For non-comm ops,return std_name1+std_name2+ ...
                For comm ops, return Comm.op_name....
            '''
            if "+" in _long_name:
                if "Comm" in _long_name:
                    return "{}.{}".format("Comm", parse_op_name(_long_name))
                else:
                    ### fused_op
                    return "+".join([parse_allinfo_from_name(_n)[1] for _n in _long_name.split("+")])
            else:
                # pid, std_name, cat, suffix = parse_allinfo_from_name(_long_name)
                return parse_allinfo_from_name(_long_name)[1]
        
        local_dfg = self.clct.dag

        # n = "server_2::worker_0->Comm.210+211.PUSH_RES~>server_2::worker_0::0"
        # node = "BW.gradient_tape/resnet50/conv5_block3_3_bn/FusedBatchNormGradV3"
        # import code
        # code.interact(local=locals())
        
        ### Invoke Memory-oriented Pass if OOM is detected
        while self.mem_usage > self.memory_budget:
            if len(self.cst_md_mng.mem_model_list) == 0:
                SingleLogger().warn("OOM: memory usage {} > {}, but no memory-oriented optimization pass is found".format(
                    self.mem_usage, self.memory_budget))
                break
            for mem_pass in self.cst_md_mng.mem_model_list:
                candidate_strategies, candidate_weights = mem_pass.init_search_space(None, G, None)
                st = self.pick_strategies(candidate_strategies, weights=candidate_weights)
                mem_pass.apply(st, G, None)
        
        no_speed_up_step = 0
        search_start_t = time.time()
        NO_SPEEDUP_STEP_BUDGET = 5
        SEARCH_TIME_BUDGET = 3600 * 5

        prev_cost = self.cur_cost
        
        while no_speed_up_step <= NO_SPEEDUP_STEP_BUDGET and time.time() - search_start_t < SEARCH_TIME_BUDGET:
            critical_path = self.wrap_critical_path(self.exct_dag)

            with open(os.path.join(ROOT_PATH, "crit_path_step{}.txt".format(self.step)), "w") as f:
                for _node, _len in critical_path:
                    f.write("{} {:.3f}\n".format(_node, _len))

            comp_op_on_critical_path_std_name = [
                ret_standar_names(op) for op, _ in critical_path 
                    if parse_cat_fine_grained(op) in ["operator.FW", "operator.BW", "operator.FW+BW"]]
            ref_pid = parse_pid_from_name(critical_path[0][0])
            topo_order_one_pid = [comp_op for comp_op, _ in self.topo_order if 
                parse_pid_from_name(comp_op) == ref_pid and 
                    parse_cat_fine_grained(comp_op) in ["operator.FW", "operator.BW", "operator.FW+BW"]]
            
            G_star = G.copy()
            PKG_star = PKG.copy()

            prev_node = None
            dp_state = DPState(opt=self)
            applied_sts = []

            # for node_n, exct_time_n in critical_path:
            SingleLogger().info("Step {}".format(self.step))
            for topo_idx, node_n in tqdm(enumerate(topo_order_one_pid), total=len(topo_order_one_pid)):
                exct_time_n = G_star.nodes[node_n]["avg"]
                model_changed = False
                if prev_node is None:
                    ### The first node
                    prev_node = node_n
                    dp_state.add_comp_op(node_n, exct_time_n, G_star)
                    continue
                
                dp_state.add_comp_op(node_n, exct_time_n, G_star)
                
                fused_comp_op = None
                if self.opfs_pass is not None and self.tsfs_pass is not None:
                    if not DISABLE_LAYER_VIEW and "BW." not in node_n:
                        continue
                    _pred = prev_node
                    t_null = self.estimate_time_related_to_comp([_pred, node_n], G_star)

                    ### Try to apply tensor fusion and operator fusion
                    G_prime = G_star.copy()
                    PKG_prime = PKG_star.copy()
                    old_tsfs_state = None
                    sts = []
                    opfs_succeed, nodes_introduced = self.try_to_apply_opfs(_pred, node_n, G_prime, PKG_prime, sts, verbose=False)
                    if opfs_succeed:
                        _fused_comp_op = nodes_introduced.pop()
                        fused_comp_op = "+".join([gen_long_name(ref_pid, parse_rawname(long_name)) for long_name in _fused_comp_op.split("+")])
                        tensor_op_names = self.comm_succs_of_comp_in_op_name(fused_comp_op, G_prime)
                    else:
                        tensor_u = self.comm_succs_of_comp_in_op_name(prev_node, G_star)
                        tensor_v = self.comm_succs_of_comp_in_op_name(node_n, G_star)
                        tensor_op_names = tensor_u.union(tensor_v)
                        G_prime = G_star.copy()
                        PKG_prime = PKG_star.copy()
                    if len(tensor_op_names) > 1:
                        old_tsfs_state = self.tsfs_pass.tsfs_state.copy()

                        assert len(tensor_op_names) == 2, tensor_op_names
                        # long_name_u = self.comm_succs_of_comp_in_long_name(_pred, G_star)
                        # long_name_v = self.comm_succs_of_comp_in_long_name(node_n, G_star)
                        tensor_op_names = list(tensor_op_names)
                        nodes_introduced = self.try_to_apply_tsfs(
                            "Comm." + tensor_op_names[0],
                            "Comm." + tensor_op_names[1],
                            G_prime, PKG_prime, sts, verbose=False)

                        fused_tensor_name = parse_op_name(nodes_introduced[0])
                        k_star, ts_part_time = self.tsfs_pass.best_partition_partial_replay(
                            fused_tensor_name, G_prime, no_throrem=True)
                        # k_star_fuse, tsfs_time = self.tsfs_pass.best_partition(dp_state.q_m[-2] + dp_state.q_m[-1])
                        if self.tsfs_pass.enable_partition:
                            ### Do NOT fuse but tensor partition may be applied
                            self.try_to_apply_ts_part([fused_tensor_name], k_star, 
                                G_prime, PKG_prime, sts, verbose=False)
                        self.tsfs_pass.update_tensor2server()
                    if opfs_succeed:
                        t_fuse = self.estimate_time_related_to_comp([_fused_comp_op], G_prime)
                    else:
                        t_fuse = self.estimate_time_related_to_comp([prev_node, node_n], G_prime)

                    if t_fuse < t_null:
                        G_star = G_prime
                        PKG_star = PKG_prime
                        model_changed = True
                        applied_sts += sts
                        SingleLogger().info(bcolors.CYELLOW + "Fuse {} {} and {}".format(_pred[:60], node_n[:60], str(tensor_op_names)) + bcolors.ENDC)
                    else:
                        ### Fusion is worse, retrieve the original tensor fusion pass state
                        if old_tsfs_state is not None:
                            self.tsfs_pass.tsfs_state = old_tsfs_state
                        fused_comp_op = None

                elif ret_standar_names(node_n) in comp_op_on_critical_path_std_name:
                    ### Computation operators are on the critical path
                    _pred = prev_node
                    if self.opfs_pass is not None:
                        if_fusion_better, opfs_time = self.opfs_pass.is_fusion_better(
                            _pred, node_n, G_star, PKG_star, dp_state, no_theorem=True)
                        if if_fusion_better:
                            SingleLogger().info(bcolors.CGREEN + "Fusion {} {} is better ...".format(dp_state.node_num-1, dp_state.node_num) + bcolors.ENDC)
                            succeed, nodes_introduced = self.try_to_apply_opfs(_pred, node_n, G_star, PKG_star, applied_sts)
                            if not succeed:
                                continue
                            fused_comp_op = nodes_introduced.pop()
                            fused_comp_op = "+".join([gen_long_name(ref_pid, parse_rawname(long_name)) for long_name in fused_comp_op.split("+")])
                            # SingleLogger().info("Success !!!")
                            model_changed = True

                            ### Update DP states after operator fusion
                            dp_state.update_state_opfs(opfs_time)
                            
                            if not DISABLE_LAYER_VIEW and self.tsfs_pass is not None:
                                tensor_op_names = self.comm_succs_of_comp_in_op_name(fused_comp_op, G_star)
                                if len(tensor_op_names) > 1:
                                    assert len(tensor_op_names) == 2, tensor_op_names
                                    # long_name_u = self.comm_succs_of_comp_in_long_name(_pred, G_star)
                                    # long_name_v = self.comm_succs_of_comp_in_long_name(node_n, G_star)
                                    tensor_op_names = list(tensor_op_names)
                                    nodes_introduced = self.try_to_apply_tsfs(
                                        "Comm." + tensor_op_names[0],
                                        "Comm." + tensor_op_names[1],
                                        G_star, PKG_star, applied_sts)

                                    fused_tensor_name = parse_op_name(nodes_introduced[0])
                                    k_star, ts_part_time = self.tsfs_pass.best_partition_partial_replay(
                                        fused_tensor_name, G_star, no_throrem=True)
                                    # k_star_fuse, tsfs_time = self.tsfs_pass.best_partition(dp_state.q_m[-2] + dp_state.q_m[-1])
                                    if self.tsfs_pass.enable_partition:
                                        ### Do NOT fuse but tensor partition may be applied
                                        self.try_to_apply_ts_part([fused_tensor_name], k_star, G_star, PKG_star, applied_sts)

                                    self.tsfs_pass.update_tensor2server()
                                    ### Update DP states after tensor fusion/partition
                                    # dp_state.update_state_tsfs(tsfs_time)
                        else:
                            ### No fusion is better
                            SingleLogger().debug(bcolors.CBLUE + "Do not fuse {} {} ...".format(
                                dp_state.node_num-1, dp_state.node_num) + bcolors.ENDC)
                            if self.tsfs_pass is not None:
                                tensor_name_list = self.comm_succs_of_comp_in_op_name(node_n, G_star)
                                if self.tsfs_pass.enable_partition and len(tensor_name_list) > 0:
                                    assert not DISABLE_LAYER_VIEW and len(tensor_name_list) == 1, (node_n, tensor_name_list)
                                    tensor_name_v = tensor_name_list.pop()
                                    k_star, ts_part_time = self.tsfs_pass.best_partition_partial_replay(
                                        tensor_name_v, G_star, no_throrem=True)
                                    self.try_to_apply_ts_part([tensor_name_v], k_star, G_star, PKG_star, applied_sts)
                                    model_changed = True

                                    ### Update DP states with only tensor partition
                                    ### TODO: ts_part_time includes BW, COMM and UPDATE time
                                    ts_part_time = ts_part_time - exct_time_n
                                    dp_state.update_state_only_ts_part(ts_part_time)

                                    self.tsfs_pass.update_tensor2server()
                else:
                    # print(prev_node, node_n)
                    ### Communication operators are on the critical path
                    tensor_u = self.comm_succs_of_comp_in_op_name(prev_node, G_star)
                    tensor_v = self.comm_succs_of_comp_in_op_name(node_n, G_star)
                    if tensor_u is None or tensor_v is None or len(tensor_u) == 0 or len(tensor_v) == 0:
                        ### TODO: handle the special case when q_n is NULL
                        # SingleLogger().warn("Have no comm succes")
                        continue
                    
                    if self.tsfs_pass is None:
                        continue

                    if DISABLE_LAYER_VIEW:
                        op_name_u = tensor_u.pop()
                        if len(tensor_u) > 0:
                            op_name_v = tensor_u.pop()
                        else:
                            op_name_v = tensor_v.pop()
                    else:
                        assert len(tensor_u) == 1, (prev_node, tensor_u)
                        assert len(tensor_v) == 1, (node_n, tensor_v)
                        op_name_u = tensor_u.pop()
                        op_name_v = tensor_v.pop()

                    if op_name_u != op_name_v:
                        print(op_name_u, op_name_v)
                        is_fuse_better, k_star, t_sync_fuse, t_sync_null = self.tsfs_pass.if_fusion_better(
                            op_name_u, op_name_v, dp_state, G_star, no_theorem=True)

                        if is_fuse_better:
                            SingleLogger().info(bcolors.CGREEN + "Tensor Fusion: fusing {} {} is better".format(op_name_u, op_name_v) + bcolors.ENDC)
                            ### Tensor fusion is better, apply this strategy
                            pid = parse_pid_from_name(node_n)
                            if self.comm_backend == "NCCL":
                                long_name_u = gen_long_name(pid, "Comm.{}.Sync".format(op_name_u))
                                long_name_v = gen_long_name(pid, "Comm.{}.Sync".format(op_name_v))
                            elif self.comm_backend == "BYTEPS":
                                long_name_u = self.comm_succs_of_comp_in_long_name(prev_node, G_star)[0]
                                long_name_v = self.comm_succs_of_comp_in_long_name(node_n, G_star)[0]
                            else:
                                raise ValueError(self.comm_backend)
                            _bw_pred_list_u = [pred for pred in G_star.predecessors(long_name_u) if parse_pid_from_name(pred) == ref_pid]
                            _bw_pred_list_v = [pred for pred in G_star.predecessors(long_name_v) if parse_pid_from_name(pred) == ref_pid]

                            nodes_introduced = self.try_to_apply_tsfs(long_name_u, long_name_v, G_star, PKG_star, applied_sts)
                            model_changed = True

                            ### Apply the partition startegy
                            if self.tsfs_pass.enable_partition:
                                fused_tensor_name = parse_op_name(nodes_introduced[0])
                                ### Partition fused tensor to k_star pieces
                                self.try_to_apply_ts_part([fused_tensor_name], k_star, G_star, PKG_star, applied_sts)

                            ### Update DP states after tensor fusion
                            tsfs_time = t_sync_fuse
                            dp_state.update_state_tsfs(tsfs_time)

                            ### TODO (HHP): fuse corresponding operatos
                            if not DISABLE_LAYER_VIEW and self.opfs_pass is not None:
                                assert len(_bw_pred_list_v) == 1 and len(_bw_pred_list_u) == 1, (_bw_pred_list_v, _bw_pred_list_u)
                                succeed, nodes_introduced = self.try_to_apply_opfs(_bw_pred_list_u[0], _bw_pred_list_v[0], G_star, PKG_star, applied_sts)
                                if succeed:
                                    fused_comp_op = nodes_introduced.pop()
                                    fused_comp_op = "+".join([gen_long_name(ref_pid, parse_rawname(long_name)) for long_name in fused_comp_op.split("+")])

                                    ### Update DP states after operator fusion
                                    opfs_time = self.opfs_pass._get_node_avg(_bw_pred_list_u[0]+"+"+_bw_pred_list_v[0])
                                    dp_state.update_state_opfs(opfs_time)

                            self.tsfs_pass.update_tensor2server()
                        else:
                            SingleLogger().info(bcolors.CBLUE + "Tensor Fusion: fusing {} {} is worse".format(op_name_u, op_name_v) + bcolors.ENDC)
                            ### Do NOT fuse tensors but tensor partition may be applied
                            if self.tsfs_pass.enable_partition:
                                SingleLogger().info(bcolors.CBLUE + "Partition {} to {} pieces".format(op_name_v, k_star) + bcolors.ENDC)
                                ### Partition fused tensor to k_star pieces
                                self.try_to_apply_ts_part([op_name_v], k_star, G_star, PKG_star, applied_sts)
                                model_changed = True
                                ### Update DP states with only tensor partition
                                dp_state.update_state_only_ts_part(t_sync_null)

                                self.tsfs_pass.update_tensor2server()

                prev_node = node_n if fused_comp_op is None else fused_comp_op

                ### debug
                if model_changed:
                    _cost_star, _exct_dag_star, _mem_usage_star, _ = self.evaluate(G_star, recd_topo_order=True)
                    write_time(cost=_cost_star)
                    if prev_cost >= _cost_star:
                        SingleLogger().info("Step {}-{}: Cost from {:.3f} to {:.3f}".format(self.step, topo_idx, prev_cost, _cost_star))
                    else:
                        SingleLogger().info(bcolors.CRED + "Step {}-{}: Cost from {:.3f} to {:.3f}".format(self.step, topo_idx, prev_cost, _cost_star) + bcolors.ENDC)
                    prev_cost = _cost_star

            _cost_star, _exct_dag_star, _mem_usage_star, topo_order = self.evaluate(G_star, recd_topo_order=True)
            SingleLogger().info("Cost from {:.3f} to {:.3f}".format(self.cur_cost, _cost_star))
            if abs((_cost_star - self.cur_cost) / self.cur_cost) < 0.001:
                no_speed_up_step += 1

            self.cur_cost = _cost_star
            self.exct_dag = _exct_dag_star
            self.mem_usage = _mem_usage_star
            self.topo_order = topo_order
            G = G_star
            PKG = PKG_star
            self.trajectory += applied_sts
            display_and_ckpt()
            self.step += 1

        display_and_ckpt()

