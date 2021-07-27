import os
import time
import json
import pickle

from optimizer.base import Optimizer, args_, ROOT_PATH
from logger_utils import SingleLogger
from cost_model._xla.pk_graph import PKGraph
from base import bcolors
from trace_utils import parse_allinfo_from_name, parse_pid_from_name, CatName

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

        if "+" in self.cst_md_mng.strategy2model:
            self.cst_md_mng.strategy2model["+"].explore_fusion = False
            self.cst_md_mng.strategy2model["+"].enable_partition = False
    
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
                G, PKG, self.step, self.trajectory, comp_ops, cur_comp_idx = pickle.load(f)
            SingleLogger().info("Loading checkpoint of step {}".format(self.step))
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"))
        else:
            self.cur_cost, self.exct_dag, self.mem_usage = self.evaluate(
                G, _path=os.path.join(ROOT_PATH, "searched_graph/init.json"),
                recd_topo_order=False)
            
            # comp_ops = self.ret_all_comp_ops(topo_ord)
            # cur_comp_idx = 1

            self.step = 0
            self.trajectory = []
            SingleLogger().info("No checkpoint found, search from scratch")

        SingleLogger().info("="*20 + " Search Starts " + "="*20)
        SingleLogger().info(bcolors.CGREEN + "Start to search, the original iteration time is %f, init cost is %f" %
                            (self.base_cost, self.cur_cost) + bcolors.ENDC)

        def display_and_ckpt():
            SingleLogger().info(bcolors.CBLUE + "Step: %d - Current speedup to the origin: %6.4f %% (%d/%d)" % (self.step,
                                                                                                        100 * (self.base_cost - self.cur_cost) / self.base_cost, cur_comp_idx, len(comp_ops)) + bcolors.ENDC)
            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {},{},{}".format(
                    self.step,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost) + "\n")

            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": self.trajectory}, f)

            # if args_.ckpt:
            ### Save checkpoints by default
            for _cost_model in self.cst_md_mng.cost_model_list:
                _cost_model.checkpoint()
            with open(graph_cache, "wb") as f:
                pickle.dump([G, PKG, self.step, self.trajectory, comp_ops, cur_comp_idx], f)

        def update_comp_idx(strategy_rsts, cur_idx, comp_ops):
            if strategy_rsts is None:
                return cur_idx + 1
            
            sts, _nodes_introduced, _nodes_removed = strategy_rsts
            for st_op, u, v in sts:
                if st_op == "+":
                    assert u in _nodes_removed
                    assert v in _nodes_removed
                    pid = parse_pid_from_name(u)
                    _nodes_introduced = [n for n in _nodes_introduced if parse_pid_from_name(n) == pid]
                    assert len(_nodes_introduced) == 1, _nodes_introduced
                    comp_ops.insert(cur_idx, _nodes_introduced[0])
                    comp_ops.remove(u)
                    comp_ops.remove(v)
                    return cur_idx
            return cur_idx + 1

        def ret_standar_names(cur_n):
            if "+" in cur_n:
                if "Comm" in cur_n:
                    _, std_name, cat, _ = parse_allinfo_from_name(cur_n)
                    tensor_name_list = std_name.split(".")[1]
                    return ["{}.{}".format(cat, _n) for _n in tensor_name_list.split("+")]
                else:
                    ### fused_op
                    return [parse_allinfo_from_name(_n)[1] for _n in cur_n.split("+")]
            else:
                # pid, std_name, cat, suffix = parse_allinfo_from_name(cur_n)
                return [parse_allinfo_from_name(cur_n)[1]]
        
        local_dfg = self.clct.dag
        
        while True:
            critical_path = self.wrap_critical_path(self.exct_dag)
            prev_nodes = None
            for cur_n, l in critical_path:
                if prev_nodes is None:
                    prev_nodes = ret_standar_names(cur_n)
                    continue
                
                # print(prev_nodes, cur_n)
                cur_nodes = ret_standar_names(cur_n)

                ### Some consecutive communication OPs on the critical path
                # if "+".join(prev_nodes) == "+".join(cur_nodes):
                #     if "Comm" in prev_nodes[0]:
                #         print(prev_nodes)
                
                # for pred in prev_nodes:
                #     for _cur_n in cur_nodes:
                #         if (pred, _cur_n) in local_dfg.edges:
                #             print(pred, _cur_n)
                #         if "Comm" in pred and "Comm" in _cur_n:
                #             print(pred, _cur_n)
                #     assert pred in local_dfg.nodes, pred

                prev_nodes = cur_nodes

            exit(0)

        while cur_comp_idx < len(comp_ops):
            if cur_comp_idx == 0:
                cur_comp_idx += 1
                continue
            op_v = comp_ops[cur_comp_idx]
            preds = list(G.predecessors(op_v))

            best_cost, best_st, best_G, best_PKG = self.cur_cost, None, None, None 
            has_better = False
            for sts in self.all_possible_strategies(preds, op_v, G, PKG):
                if len(sts) == 0:
                    continue
                G_star = G.copy()
                PKG_star = PKG.copy()
                _nodes_introduced, _nodes_removed = self.apply_strategies(
                    G_star, PKG_star, sts)
                _cost_star, _exct_dag_star, _mem_usage_star = self.evaluate(
                    G_star)
                if best_cost is None or _cost_star < best_cost:
                    best_cost, best_st, best_G, best_PKG = _cost_star, (sts, _nodes_introduced, _nodes_removed), G_star, PKG_star
                    has_better = True
            
            if op_v in G.successors(op_u):
                print("Exist u -> v")
            else:
                print("No u -> v")

            if has_better:
                G = best_G
                PKG = best_PKG
                self.trajectory += best_st[0]
                self.cur_cost = best_cost
                SingleLogger().info(bcolors.CGREEN + "Take strategies: {}".format(str([st[0] for st in best_st[0]])) + bcolors.ENDC)
            else:
                SingleLogger().info(bcolors.CYELLOW + "Take no action" + bcolors.ENDC)
            
            # print(best_cost, best_st)
            cur_comp_idx = update_comp_idx(best_st, cur_comp_idx, comp_ops)
            display_and_ckpt()
            self.step += 1
        
        display_and_ckpt()
        if "+" in self.cst_md_mng.strategy2model:
            self.cst_md_mng.strategy2model["+"]._dump_cluster_mapping(
                G, os.path.join(ROOT_PATH, "cluster_mapping.txt"), partition=True)

        if "++" in self.cst_md_mng.strategy2model:
            self.cst_md_mng.strategy2model["++"].dump_tensor_grp_mapping()
