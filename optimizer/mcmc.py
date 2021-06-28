import math
import os
import time
import json
import pickle
import traceback
import random

from optimizer.base import Optimizer, args_, ROOT_PATH
from logger_utils import SingleLogger
from cost_model._xla.pk_graph import PKGraph
from base import bcolors
from cost_model.base import OptApplyStrategyError, OptNoValidStrategyError, OptQueryCostModelError

MCMC_BETA = args_.mcmc_beta

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
        if G is None:
            G = self.dag.copy()
            PKG = PKGraph(G)

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
        SingleLogger().info(bcolors.CGREEN + "Start to search, the original iteration time is %f, init cost is %f" %
                            (self.base_cost, self.cur_cost) + bcolors.ENDC)
        candidates, _ = self.candidate_selection(
            G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
        search_space, weights = self.init_search_space(candidates, G, PKG)
        SingleLogger().info(bcolors.CBLUE + "# of candidates: {}, space: {}".format(
            len(candidates), len(search_space)) + bcolors.ENDC)

        def display_and_ckpt():
            SingleLogger().info(bcolors.CBLUE + "Step: %d - Current speedup to the origin: %6.4f %%" % (self.step,
                100 * (self.base_cost - self.cur_cost) / self.base_cost) + bcolors.ENDC)
            SingleLogger().info(bcolors.CBLUE + "Step: %d - Best speedup: %d th step, speed up to the origin: %6.4f %%" % (self.step,
                self.best_step, 100 * (self.base_cost - self.best_cost) / self.base_cost) + bcolors.ENDC + "\n")

            with open(os.path.join(ROOT_PATH, "search_trajectory.txt"), "a") as f:
                f.write(str(time.time()) + ": {},{},{}".format(
                    self.step,
                    100 * (self.base_cost - self.cur_cost) / self.base_cost,
                    100 * (self.base_cost - self.best_cost) / self.base_cost) + "\n")

            with open(os.path.join(ROOT_PATH, "best_strategy.txt"), "w") as f:
                json.dump({"best_strategy": self.best_strategy}, f)

            # if args_.ckpt:
            ### Save checkpoints by default
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
            SingleLogger().info(bcolors.CBLUE + "Group Num: {}: default fusion: {} ms, Cur cost: {} ms".format(
                st[1], self.cur_cost, self.cost_star) + bcolors.ENDC)
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
                    
                    ### 1. Pick strategies
                    try:
                        strategy = self.pick_strategy(search_space, weights=weights, invalid_strategies=invalid_strategies)
                        msg = bcolors.CBLUE + "Picked strategy ({}, {}, {}).".format(*strategy)
                        if len(msg) > 200:
                            msg = msg[:200] + "..."
                        SingleLogger().debug(msg + bcolors.ENDC)
                    except OptNoValidStrategyError:
                        # no valid strategies available, refresh search space
                        SingleLogger().info(bcolors.CBLUE + "Search space exhausted." + bcolors.ENDC)
                        candidates, _ = self.candidate_selection(G_star, topk=None, critical_path=None)
                        search_space, weights = self.init_search_space(candidates, G_star, PKG_star)
                        invalid_strategies = set()
                        continue

                    ### 2. Apply strategies
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
                            # invalid_strategies.add(st)
                            is_succ_apply = False
                            break
                    if not is_succ_apply:
                        continue
                    successful_strategies += 1
                    strategy_history_in_step += strategies
                    strategy_introduced_nodes.update(nodes_introduced)
                    strategy_removed_nodes.update(nodes_removed)

                    ### 3. Evaluate the cost through the replayer
                    self.step += 1
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
                    
                    ### 4. Check whether to init search space again
                    if successful_strategies < step_size and len(search_space) == 0:
                        candidates, _ = self.candidate_selection(
                            G_star, topk=None, critical_path=self.wrap_critical_path(self.exct_dag_star))
                        search_space, weights = self.init_search_space(
                            candidates, G_star, PKG_star)

                    ### 5. Log the strategy and its cost
                    invalid_strategies = set()
                    msg = bcolors.CBLUE + "Step: {} - ".format(
                        self.step) + "Strategy ({}, {}, {}) successfully applied.".format(*strategy)
                    if len(msg) > 200:
                        msg = msg[:200] + "... successfully applied."
                    SingleLogger().info(msg + bcolors.ENDC)

                SingleLogger().info(bcolors.CBLUE + "Step: {} - cost from {:.5f} -> {:.5f}".format(
                    self.step, self.cur_cost, self.cost_star) + bcolors.ENDC)
                
                if strategy[0] in ["gradient_accumulation", "recomputation"]:
                    is_accept = True
                else:    
                    is_accept = self.accept_or_not(self.cur_cost, self.cost_star)

                ### update Graph Pass internal states
                self.cost_model_flush(is_accept)
                ### update heat history 
                self.update_fusion_heat_history(is_accept, strategy_removed_nodes, 
                    strategy_introduced_nodes, fusion=(strategy[0]!="-"))

                if is_accept:
                    invalid_strategies = set()
                    G = G_star
                    PKG = PKG_star
                    self.trajectory += strategy_history_in_step
                    self.cur_cost = self.cost_star
                    self.exct_dag = self.exct_dag_star
                    self.mem_usage = self.mem_usage_star

                    ### Cache the best strategy
                    if self.cur_cost < self.best_cost:
                        self.best_cost = self.cur_cost
                        self.best_strategy = self.trajectory.copy()
                        self.best_step = self.step
                        if "+" in self.cst_md_mng.strategy2model:
                            self.cst_md_mng.strategy2model["+"]._dump_cluster_mapping(
                                G, os.path.join(ROOT_PATH, "cluster_mapping.txt"), partition=True)
                        
                        if "++" in self.cst_md_mng.strategy2model:
                            self.cst_md_mng.strategy2model["++"].dump_tensor_grp_mapping()
                        # DEBUG: log the best graph for debugging
                        self.evaluate(G, 
                            _path=os.path.join(ROOT_PATH, "best.json".format(self.step)),
                            _crit_filename=os.path.join(ROOT_PATH, "best_crit.json".format(self.step)))
                    ### Init the new search space
                    candidates, _ = self.candidate_selection(
                        G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
                    search_space, weights = self.init_search_space(
                        candidates, G, PKG)
                    break
            
            display_and_ckpt()
            if len(search_space) == 0:
                ### Init new search space
                candidates, _ = self.candidate_selection(
                    G, topk=None, critical_path=self.wrap_critical_path(self.exct_dag))
                search_space, weights = self.init_search_space(
                    candidates, G, PKG)
                ### End of search
                if len(search_space) == 0:
                    break

        display_and_ckpt()
    
    def accept_or_not(self, cost, new_cost):
        # prob = min(1, (math.exp(beta * (cost - new_cost))))
        try:
            prob = math.exp(MCMC_BETA * math.log(self.step + 1) * (cost - new_cost))
        except OverflowError:
            prob = float('inf')

        # if cost > new_cost:
        if prob > 1:
            SingleLogger().info(
                bcolors.CGREEN + "Accept a better action, orig cost: {:.5f}, new cost: {:.5f}".format(cost, new_cost) + bcolors.ENDC)
            return True
        else:
            # prob = math.exp(MCMC_BETA * (cost - new_cost))
            r = random.random()
            if r < prob:
                SingleLogger().info(
                    bcolors.CGREEN + "Accept a worse action with random value: {:.5f} < {:.5f} ".format(r, prob) + bcolors.ENDC)
                return True
            else:
                SingleLogger().info(
                    bcolors.CYELLOW + "Rejected a worse action with random value: {:.5f} >= {:.5f} ".format(r, prob) + bcolors.ENDC)
                return False
