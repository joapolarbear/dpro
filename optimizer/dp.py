import os
import time
import json
import pickle

from optimizer.base import Optimizer, args_, ROOT_PATH
from logger_utils import SingleLogger
from cost_model._xla.pk_graph import PKGraph
from base import bcolors
from trace_utils import gen_long_name, parse_allinfo_from_name, parse_cat_fine_grained, \
    parse_cat_from_name, parse_op_name, parse_pid_from_name, CatName, parse_rawname

DEBUG = False

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
        self.q_e[-1] = max(self.p_e[-1], self.q_e[-3]) + self.q_d[-1]
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
            self.q_e[-1] = max(self.p_e[-1], self.q_e[-3]) + self.q_d[-1]
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

        self.opfs_pass = self.cst_md_mng.strategy2model.get("+", None)
        self.tsfs_pass = self.cst_md_mng.strategy2model.get("++", None)

        if self.opfs_pass is not None:
            self.opfs_pass.explore_fusion = False
            self.opfs_pass.enable_partition = False
    
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
    def try_to_apply_opfs(self, u, v, _dag, _pkg, applied_sts):
        if self.opfs_pass is None:
            return
        SingleLogger().info("Fuse operator {} {}".format(u, v))
        opfs_sts = ("+", u, v)
        success, nodes_introduced, nodes_removed = \
            self.opfs_pass.apply(opfs_sts, _dag, _pkg)
        self.opfs_pass.flush(is_accept=True)
        applied_sts.append(opfs_sts)
        return nodes_introduced
    
    def try_to_apply_tsfs(self, long_name_u, long_name_v, _dag, _pkg, applied_sts):
        if self.tsfs_pass is None:
            return []
        SingleLogger().info("Fuse tensor {} {}".format(long_name_u, long_name_v))
        tsfs_sts = ("++", long_name_u, long_name_v)
        _, _nodes_introduced, _nodes_removed = \
            self.tsfs_pass.apply(tsfs_sts, _dag, _pkg)
        self.tsfs_pass.flush(is_accept=True)
        applied_sts.append(tsfs_sts)

        ### DEBUG
        assert len(set([parse_op_name(op) for op in _nodes_introduced])) == 1, _nodes_introduced

        return _nodes_introduced
    
    def try_to_apply_ts_part(self, tensor_name_list, k_star, _dag, _pkg, applied_sts):
        ''' Partition tensors to k_star pieces
        tensor_name_list: list
        '''
        if self.tsfs_pass is None:
            return []
        SingleLogger().info("Partition tensor {} to {} pieces".format(tensor_name_list, k_star))
        for tensor_name in tensor_name_list:
            self.tsfs_pass._tensor_partition(_dag, _pkg, tensor_name, k_star)
            self.tsfs_pass.flush(True)
            applied_sts.append(("tspart", tensor_name, k_star))
    
    def comm_succs_of_comp_in_op_name(self, comp_op, _dag):
        return set([parse_op_name(n) for n in _dag.successors(comp_op) if parse_cat_from_name(n) == CatName.COMM.value])

    def comm_succs_of_comp_in_long_name(self, comp_op, _dag):
        return [n for n in _dag.successors(comp_op) if parse_cat_from_name(n) == CatName.COMM.value]

    def search(self, graph_cache=os.path.join(ROOT_PATH, "graph_cache.pickle")):

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
            self.tsfs_pass.flush(True)
            self.tsfs_pass._apply_partition_size(G, PKG, part_size_in_B)
            self.tsfs_pass.flush(True)

            _cost_star, _exct_dag_star, _mem_usage_star, topo_order = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "test.json"),
                recd_topo_order=True)
            SingleLogger().info(bcolors.CGREEN + "New cost {}".format(_cost_star) + bcolors.ENDC)
            exit(0) 

        def display_and_ckpt():
            SingleLogger().info(bcolors.CBLUE + "Step: %d - Current speedup to the origin: %6.4f %%" % (self.step,
                                                                                                        100 * (self.base_cost - self.cur_cost) / self.base_cost) + bcolors.ENDC)
            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {},{},{}".format(
                    self.step,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost) + "\n")

            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": self.trajectory}, f)

            ### Save checkpoints by default
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.checkpoint()
            with open(graph_cache, "wb") as f:
                pickle.dump([G, PKG, self.step, self.trajectory], f)

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
        
        while no_speed_up_step <= NO_SPEEDUP_STEP_BUDGET and time.time() - search_start_t < SEARCH_TIME_BUDGET:
            critical_path = self.wrap_critical_path(self.exct_dag)
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
            comm_bound = False

            # for node_n, exct_time_n in critical_path:
            for node_n in topo_order_one_pid:
                exct_time_n = G_star.nodes[node_n]["avg"]
                if prev_node is None:
                    ### The first node
                    prev_node = node_n
                    dp_state.add_comp_op(node_n, exct_time_n, G_star)
                    continue
                
                dp_state.add_comp_op(node_n, exct_time_n, G_star)
                
                fused_comp_op = None
                if ret_standar_names(node_n) in comp_op_on_critical_path_std_name:
                    ### Computation operators are on the critical path
                    _pred = prev_node
                    if self.opfs_pass is not None:
                        if_fusion_better, opfs_time = self.opfs_pass.is_fusion_better(_pred, node_n, G_star, PKG_star, dp_state)
                        if if_fusion_better:
                            SingleLogger().info("Fusion {} {} is better ...".format(dp_state.node_num-1, dp_state.node_num))
                            nodes_introduced = self.try_to_apply_opfs(_pred, node_n, G_star, PKG_star, applied_sts)
                            fused_comp_op = gen_long_name(ref_pid, parse_rawname(nodes_introduced.pop()))
                            SingleLogger().info("Success !!!")

                            ### Update DP states after operator fusion
                            dp_state.update_state_opfs(opfs_time)
                            
                            if self.tsfs_pass is not None:
                                ### TODO (HHP): fuse corresponding tensors
                                
                                tensor_long_name = self.comm_succs_of_comp_in_long_name(fused_comp_op, G_star)
                                if len(tensor_long_name) > 1:
                                    assert len(tensor_long_name) == 2, tensor_long_name
                                    # long_name_u = self.comm_succs_of_comp_in_long_name(_pred, G_star)
                                    # long_name_v = self.comm_succs_of_comp_in_long_name(node_n, G_star)
                                    tensor_long_name = list(tensor_long_name)
                                    nodes_introduced = self.try_to_apply_tsfs(tensor_long_name[0], tensor_long_name[1], G_star, PKG_star, applied_sts)
                                    k_star_fuse, tsfs_time = self.tsfs_pass.best_partition(dp_state.q_m[-2] + dp_state.q_m[-1])
                                    if self.tsfs_pass.enable_partition:
                                        fused_tensor_name = parse_op_name(nodes_introduced[0])
                                        ### Do NOT fuse but tensor partition may be applied
                                        self.try_to_apply_ts_part([fused_tensor_name], k_star, G_star, PKG_star, applied_sts)

                                    ### Update DP states after tensor fusion/partition
                                    dp_state.update_state_tsfs(tsfs_time)
                                    
                        else:
                            SingleLogger().debug("Do not fuse {} {} ...".format(dp_state.node_num-1, dp_state.node_num))
                            ### Not to fuse operators
                            if self.tsfs_pass is not None:
                                tensor_name_v = self.comm_succs_of_comp_in_op_name(node_n, G_star)
                                k_star, ts_part_time = self.tsfs_pass.best_partition(dp_state.q_m[-1])
                                if self.tsfs_pass.enable_partition and len(tensor_name_v) > 0:
                                    assert len(tensor_name_v) == 1
                                    self.try_to_apply_ts_part([tensor_name_v.pop()], k_star_fuse)
                                
                                    ### Update DP states with only tensor partition
                                    dp_state.update_state_only_ts_part(ts_part_time)
                else:
                    # print(prev_node, node_n)
                    ### Communication operators are on the critical path
                    tensor_u = self.comm_succs_of_comp_in_op_name(prev_node, G_star)
                    tensor_v = self.comm_succs_of_comp_in_op_name(node_n, G_star)
                    if tensor_u is None or tensor_v is None or len(tensor_u) == 0 or len(tensor_v) == 0:
                        ### TODO: handle the special case when q_n is NULL
                        # SingleLogger().warn("Have no comm succes")
                        continue
                    
                    assert len(tensor_u) == 1, (prev_node, tensor_u)
                    assert len(tensor_v) == 1, (node_n, tensor_v)
                    op_name_u = tensor_u.pop()
                    op_name_v = tensor_v.pop()
                    if op_name_u != op_name_v and self.tsfs_pass is not None:
                        print(op_name_u, op_name_v)
                        is_fuse_better, k_star, t_sync_fuse, t_sync_null = self.tsfs_pass.if_fusion_better(op_name_u, op_name_v, dp_state, G_star)
                        if is_fuse_better:
                            SingleLogger().info("Tensor Fusion: fusing {} {} is better".format(op_name_u, op_name_v))
                            ### Tensor fusion is better, apply this strategy
                            pid = parse_pid_from_name(node_n)
                            if self.comm_backend == "NCCL":
                                long_name_u = gen_long_name(pid, "Comm.{}.Sync".format(op_name_u))
                                long_name_v = gen_long_name(pid, "Comm.{}.Sync".format(op_name_v))
                            else:
                                long_name_u = self.comm_succs_of_comp_in_long_name(prev_node, G_star)[0]
                                long_name_v = self.comm_succs_of_comp_in_long_name(node_n, G_star)[0]
                            _in_bw_list_u = list(G_star.predecessors(long_name_u))
                            _in_bw_list_v = list(G_star.predecessors(long_name_v))

                            nodes_introduced = self.try_to_apply_tsfs(long_name_u, long_name_v, G_star, PKG_star, applied_sts)
                            
                            ### Apply the partition startegy
                            if self.tsfs_pass.enable_partition:
                                fused_tensor_name = parse_op_name(nodes_introduced[0])
                                ### Partition fused tensor to k_star pieces
                                self.try_to_apply_ts_part([fused_tensor_name], k_star, G_star, PKG_star, applied_sts)

                            ### Update DP states after tensor fusion
                            tsfs_time = t_sync_fuse
                            dp_state.update_state_tsfs(tsfs_time)

                            ### TODO (HHP): fuse corresponding operatos
                            if self.opfs_pass is not None:
                                assert len(_in_bw_list_v) == 0 and len(_in_bw_list_u) == 0
                                self.try_to_apply_opfs(_in_bw_list_u[0], _in_bw_list_v[0], G_star, PKG_star, applied_sts)

                                ### Update DP states after operator fusion
                                opfs_time = self.opfs_pass._get_node_avg(_in_bw_list_u[0]+"+"+_in_bw_list_v[0])
                                dp_state.update_state_opfs(opfs_time)

                        else:
                            SingleLogger().info("Tensor Fusion: fusing {} {} is worse".format(op_name_u, op_name_v))
                            ### Do NOT fuse tensors but tensor partition may be applied
                            if self.tsfs_pass.enable_partition:
                                pid = parse_pid_from_name(node_n)
                                ### Partition fused tensor to k_star pieces
                                self.try_to_apply_ts_part([op_name_v], k_star, G_star, PKG_star, applied_sts)
                            
                                ### Update DP states with only tensor partition
                                dp_state.update_state_only_ts_part(t_sync_null)

                prev_node = node_n if fused_comp_op is None else fused_comp_op

            _cost_star, _exct_dag_star, _mem_usage_star, topo_order = self.evaluate(G_star, recd_topo_order=True)
            SingleLogger().info("Cost from {:.3f} to {:.3f}".format(self.cur_cost, _cost_star))
            if abs((_cost_star - self.cur_cost) / self.cur_cost) < 0.001:
                no_speed_up_step += 1

            self.cur_cost = _cost_star
            self.exct_dag = _exct_dag_star
            self.mem_usage = _mem_usage_star
            G = G_star
            PKG = PKG_star
            self.trajectory += applied_sts
            display_and_ckpt()
            self.step += 1

        display_and_ckpt()
        if self.opfs_pass is not None:
            self.opfs_pass._dump_cluster_mapping(
                G, os.path.join(ROOT_PATH, "cluster_mapping.txt"), partition=True)

        if self.tsfs_pass is not None:
            self.tsfs_pass.dump_tensor_grp_mapping()
