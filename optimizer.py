import networkx as nx
import random
import math
import time

from replay import Replayer
from trace_utils import *
from dag_utils import *

class GraphExpand(Enum):
    NOT=0
    PARTIAL=1
    FULLY=2

args = arg_utils.SingleArg().args
MAX_TREE_DEPTH = 1000
MAX_LOOP = 1000
UCB_GAMMA = args.ucb_gamma
MCMC_BETA = args.mcmc_beta


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

class Optimizer:
    def __init__(self, collector, cost_models, memory_budget=None):
        self.clct = collector
        self.cost_models = cost_models
        self.platform = self.clct.platform
        self.comm_backend = self.clct.comm_backend
        self.index2name = {}
        self.index2pid = {}
        ### Get the dependency graph
        self.dag = self.relabel_dag_node(self.clct.trail_dag)

        self.base_cost, _ = self.evaluate(self.dag)
        SingleLogger().info("Start to search, the original iteration time is %f" % self.base_cost)

        ### Used to cache the node attribtue
        self.node_attr_cache = {}

        ### Budget
        self.memory_budget = memory_budget

        ### Some hyper-parameter
        self.enable_defusion = False

    def relabel_dag_node(self, _dag):
        if self.platform == "TENSORFLOW":
            def relabel_func(old_label):
                if ("BW" in old_label or "FW" in old_label or "Comm" in old_label) and "^" not in old_label:
                    layer_name = parse_layer_name(old_label)
                    layer_pid = parse_pid_from_name(old_label)
                    # if layer_pid not in self.cost_models or layer_name not in self.cost_models[layer_pid].graph_def_util.operation_names:
                    #     return "DEL~"+old_label
                    layer_index = len(self.index2name)
                    new_name = ("[%d]"%layer_index).join(old_label.split(layer_name))
                    self.index2name[layer_index] = layer_name
                    self.index2pid[layer_index] = layer_pid
                    new_label = ("[%d]"%layer_index).join(old_label.split(layer_name))
                    return ("[%d]"%layer_index).join(old_label.split(layer_name))
                else:
                    return old_label
        elif self.platform == "MXNET":
            def relabel_func(old_label):
                if ("BW" in old_label or "FW" in old_label or "Comm" in old_label) and "^" not in old_label:
                    layer_name = parse_layer_name(old_label)
                    layer_pid = parse_pid_from_name(old_label)
                    # if layer_pid not in self.cost_models or layer_name not in self.cost_models[layer_pid]:
                    #     return "DEL~"+old_label
                    layer_index = self.clct.para_dict.parse_layer_index(layer_name)
                    self.index2name[layer_index] = layer_name
                    self.index2pid[layer_index] = layer_pid
                    return ("[%d]"%layer_index).join(old_label.split(layer_name))
                # else:
                #     return old_label
        else:
            raise NotImplementedError("Unsupported platform {}".format(self.platform))
        # new_dag = nx.relabel_nodes(_dag, relabel_func)
        # nodes_to_remove = []
        # for node, attrs in new_dag.nodes.items():
        #     if "DEL~" in node:
        #         nodes_to_remove.append(node)
        # for node in nodes_to_remove:
        #     new_dag.remove_node(node)
        return nx.relabel_nodes(_dag, relabel_func)

    def concat_name(self, u_, v_):
        return "%s+%s"%(u_, v_)
    
    def _parse_index_from_name(self, name_):
        return int(name_.split("[")[1].split("]")[0])

    def _get_original_name_pid_from_index(self, name_):
        index = self._parse_index_from_name(name_)
        return self.index2name[index], self.index2pid[index]

    def _get_original_name_pid_from_fused_node(self, u_):
        single_pid = None
        orig_names = []
        for node_name in self._get_defused_node_names(u_):
            orig_name, pid = self._get_original_name_pid_from_index(node_name)
            orig_names.append(orig_name)
            if single_pid is None:
                single_pid = pid
            else:
                if single_pid != pid:
                    raise RuntimeError("Fused DAG node {} contains ops from different machines.".format(u_))
        return orig_names, single_pid
    
    def _query_cost_model(self, fused_u_):
        # query cost model for exec time of a fused node u
        nodes_in_u, u_pid = self._get_original_name_pid_from_fused_node(fused_u_)
        nodes_to_fuse = set(nodes_in_u)
        exec_time = self.cost_models[u_pid].predict(nodes_to_fuse) / 1000
        if exec_time < 0:
            raise RuntimeError("Failed to query cost model.")
        return exec_time

    def combine_avg(self, u, v):
        # call cost model to obtain the combined time
        fused_name = self.concat_name(u, v)
        return self._query_cost_model(fused_name)

    def combine_gap(self, ug, vg):
        ### TODO (huhanpeng): key component
        ### Use max to avoid one input is zero, 
        ### some how for the new gap x, ug < x < ug + vg, vg < x < ug + vg
        return max(max((ug + vg) / 0.8, ug), vg)
        return 0

    def get_node_attr(self, n, attr_):
        if attr_ in self.node_attr_cache[n]:
            return self.node_attr_cache[n][attr_]
        else:
            return 0

    def cache_node_attr(self, n, attrs):
        ### TODO (huhanpeng): need .copy() ???
        self.node_attr_cache[n] = attrs
        # print("cache attributes for %s" % n)

    def op_fusion(self, _dag, u_, v_):
        ### u_ and v_ are both FW nodes
        self._fuse_pair(_dag, u_, v_)
        # self._fuse_pair(_dag, self.convert_fw2bw(v_), self.convert_fw2bw(u_))
        

    def _fuse_pair(self, _dag, u_, v_):
        # print("fuse {} {}".format(u_, v_))
        ### Cache the node attributes in case they will be used when de-fuse
        if u_ not in self.node_attr_cache:
            self.cache_node_attr(u_, _dag.nodes[u_])
        if v_ not in self.node_attr_cache:
            self.cache_node_attr(v_, _dag.nodes[v_])

        new_name = self.concat_name(u_, v_)

        ### Add new nodes and get the attibute
        if new_name in self.node_attr_cache:
            _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            _dag.add_node(new_name)
            ### Calculate the new attribute
            self.combine_nodes_attr(_dag, new_name, u_, v_)
            ### cache the attribute
            self.cache_node_attr(new_name, _dag.nodes[new_name])

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

    def combine_nodes_attr(self, _dag, target, u_, v_):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
        _dag.nodes[target]["avg"] = self.combine_avg(u_, v_)
        _dag.nodes[target][GAP_STR_OP2OP] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2OP), self.get_node_attr(v_, GAP_STR_OP2OP))
        _dag.nodes[target][GAP_STR_OP2COMM] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2COMM), self.get_node_attr(v_, GAP_STR_OP2COMM))

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
            target["avg"] = self._query_cost_model(new_name)
            ### set and cache the attribute
            nx.set_node_attributes(_dag, {new_name:attrs})
            self.cache_node_attr(new_name, _dag.nodes[new_name])

    def convert_fw2bw(self, _name):
        assert "FW" in _name
        _name = "BW".join(_name.split("FW"))
        if "+" in _name:
            ns = _name.split("+")
            ns.reverse()
            return "+".join(ns)
        else:
            return _name

    def op_defusion(self, _dag, target, pos):
        self._defuse_pair(_dag, target, pos)
        target = "BW".join(target.split("FW"))
        ns = target.split("+")
        ns.reverse()
        pos = len(ns) - pos - 2
        target = "+".join(ns)
        assert target in _dag.nodes
        self._defuse_pair(_dag, target, pos)
    
    def _get_defused_node_names(self, fused_node_):
        return fused_node_.split("+")

    def _defuse_pair(self, _dag, target, pos):
        ns = target.split("+")
        pos += 1
        left = "+".join(ns[:pos])
        right = "+".join(ns[pos:])

        ### For the edges related to target, re-connect them to left or right
        ### NOTE, Assumption: for an edge a->b, a is fused with others become ...+...+a, b is fused with others b+...+...
        left_most = ns[0]
        right_most = ns[-1]

        ### Handle in_edges
        for in_node, _ in _dag.in_edges(target):
            if "+" in in_node:
                ### In_node is fused with others
                in_node_last = in_node.split("+")[-1]
            else:
                ### in_node is a node in original dag
                in_node_last = in_node

            if (in_node_last, left_most) in self.dag:
                _dag.add_edge(in_node, left)
            else:
                _dag.add_edge(in_node, right)
        ### Handle the edges between left and right
        _dag.add_edge(left, right)
        ### Handle out_edges
        for out_node in _dag.successors(target):
            if "+" in out_node:
                out_node_first = out_node.split("+")[0]
            else:
                out_node_first = out_node

            if (right_most, out_node_first) in self.dag:
                _dag.add_edge(right, out_node)
            else:
                _dag.add_edge(left, out_node)

        self.parse_node_attr(_dag, left)
        self.parse_node_attr(_dag, right)
        _dag.remove_node(target)

        assert left in _dag.nodes
        assert "avg" in _dag.nodes[left]
        assert right in _dag.nodes
        assert "avg" in _dag.nodes[right]
        

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

        return max(step_end_time_ms), replayer.exct_dag

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

            iter_time, exct_dag = self.evaluate(new_dag)
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
        
        filtered_critical_path = []
        for (node, length) in critical_path:
            if "+" in node:
                # fused node
                filtered_critical_path.append((node, length))
            elif "[" in node and "]" in node:
                orig_name, pid = self._get_original_name_pid_from_index(node)
                if orig_name in self.cost_models[pid].graph_def_util.operation_names:
                    filtered_critical_path.append((node, length))

        if topk is None:
            return filtered_critical_path, new_dag
        else:
            critical_path = sorted(filtered_critical_path, key=lambda x: x[1], reverse=True)
            return critical_path[:topk], new_dag

    def wrap_critical_path(self, _dag, verbose=False):
        # t = time.time()

        cal_edge_cost(_dag)
        ret = dag_longest_path(_dag, None, weight="cost", default_weight=0, _debug_level=(1 if verbose else 0))

        # print("critical path time {}".format(time.time() - t))

        return ret

    def init_search_space(self, candidates, _dag):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        ### TODO (huhanpeng): currently only consider fusion
        ### 			Need to add quantization
        search_space = []
        weights = []
        prun_cnt = 0
        for n, l in candidates:
            if self.enable_defusion and "+" in n:
                ### This a fused node
                ns = n.split("+")
                cat = parse_cat_fine_grained(ns[0])
                pid = parse_pid_from_name(ns[0])
                if cat == "operator.FW":
                    ### defusion process
                    pos = 0
                    while True:
                        search_space.append(("-", n, pos))
                        weights.append(l)
                        if pos >= (len(ns) - 2):
                            break
                        pos += 1
            else:
                ### Nodes that have never been fused
                cat = parse_cat_fine_grained(n)
                pid = parse_pid_from_name(n)

            # if cat != "operator.FW":
            # 	### TODO (huhanpeng): only pick FW, then fuse corresponding BW
            # 	continue

            for succ_ in _dag.successors(n):
                _pid = parse_pid_from_name(succ_)
                _cat = parse_cat_fine_grained(succ_)
                if pid != _pid or cat != _cat:
                    continue

                ### Assumption 1: for edge a->b, only if the indegree of b is 1, the node can be fused
                # bw_v = self.convert_fw2bw(n)
                # if len(_dag.in_edges(succ_)) > 1 or len(_dag.in_edges(bw_v)) > 1:
                if len(_dag.in_edges(succ_)) > 1:
                    continue

                # bw_u = self.convert_fw2bw(succ_)
                # assert bw_u in _dag.nodes and bw_v in _dag.nodes
                
                # TODO (huhanpeng): this process is only for NCCL now
                # if args.comm_backend != "NCCL":
                # 	raise NotImplementedError()

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
                        if self.comm_backend == "NCCL":
                            comm_t += ret_comm_time(bw_u_succ)
                        else:
                            comm_t += _dag.nodes[bw_u_succ]["avg"]
                
                # if comm_t >= _dag.nodes[bw_v]["avg"]:
                if comm_t >= _dag.nodes[succ_]["avg"]:
                    prun_cnt += 1
                    # G_star = self.apply_strategies(self.dag, ("+", n, succ_))
                    # iter_time, _ = self.evaluate(G_star)
                    # SingleLogger().info("Prune fusing %s and %s, performace %f %%"%(n, succ_, 100*(self.base_cost - iter_time)/self.base_cost))
                    continue

                search_space.append(("+", n, succ_))
                weights.append(l)
        # SingleLogger().info("Init search space len={} from {} candidates, prune {}".format(len(search_space), len(candidates), prun_cnt))
        return search_space, weights

    def apply_strategies(self, _dag, strategy):
        # print(strategy)

        ### TODO (huhanpeng): is shallow copy is enough ???
        __dag = _dag.copy()
        def __apply(s):
            op, target, next_ = s
            ### TODO (huhanpeng): need further add other optimization techiniques
            if op == "+":
                ### Fuse two nodes
                self.op_fusion(__dag, target, next_)
            elif op == "-":
                self.op_defusion(__dag, target, next_)

        if isinstance(strategy, list):
            for s in strategy:
                __apply(s)
        else:
            __apply(strategy)
        return __dag

    def pick_strategy(self, search_space, weights=None, invalid_strategies = None):
        ### TODO (huhanpeng): need some priority/heuristic
        valid_search_space = search_space.copy()
        if invalid_strategies is not None:
            for inv_st in invalid_strategies:
                if inv_st in valid_search_space:
                    valid_search_space.remove(inv_st)
        if not valid_search_space:
            raise RuntimeError("No valid strategy available.")
        if weights is None:
            return random.choice(valid_search_space)
        else:
            return self.weighted_choice(valid_search_space, weights)

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
        self.enable_defusion = True

    def search(self):
        ### TODO (huhanpeng): is shallow copy is enough ???
        G = self.dag.copy()
        cost = self.base_cost
        trajectory = []
        candidates, _ = self.candidate_selection(G, topk=None)
        search_space, weights = self.init_search_space(candidates, G)
        SingleLogger().info("\033[94m # of candidates: {}\033[0m".format(len(candidates)))
        candidate_orig_names = [self._get_original_name_pid_from_index(cand[0])[0] for cand in candidates]
        best_cost = cost
        best_strategy = trajectory.copy()
        step = 0
        while True:
            invalid_strategies = set()
            while True and len(search_space) > 0:
                try:
                    strategy = self.pick_strategy(search_space, weights=weights, invalid_strategies=invalid_strategies)
                except:
                    # no valid strategies available, refresh search space
                    candidates, _ = self.candidate_selection(G, topk=None)
                    search_space, weights = self.init_search_space(candidates, G)
                    break
                try:
                    G_star = self.apply_strategies(G, strategy)
                except:
                    # strategy invalid
                    invalid_strategies.add(strategy)
                    continue
                SingleLogger().info("\033[94m" + "Strategy ({}, {}, {}) successfully applied.".format(*strategy) + "\033[0m")
                ### Start to replay
                cost_star, exct_dag = self.evaluate(G_star, _filename="/root/searched_graph/most_resent.json")
                cost_star, exct_dag = self.evaluate(G_star)
                SingleLogger().info("\033[94m Step: {}, Orig cost: {}, New cost: {} \033[0m".format(step, cost, cost_star))
                step += 1
                if self.accept_or_not(cost, cost_star):
                    invalid_strategies = set()
                    op, target, next_ = strategy
                    if op == "+":
                        SingleLogger().info("Fuse %s and %s" % (target, next_))
                    elif op == "-":
                        SingleLogger().info("De-fuse %s at %dth op" % (target, next_))
                    else:
                        raise ValueError("Invalid graph transformation operation: {}".format(op))

                    G = G_star
                    cost = cost_star
                    trajectory.append(strategy)

                    ### Cache the best strategy
                    if cost < best_cost:
                        best_cost = cost
                        best_strategy = trajectory.copy()
                    ### Init new search space
                    candidates, _ = self.candidate_selection(G, topk=None, critical_path=self.wrap_critical_path(exct_dag))
                    search_space, weights = self.init_search_space(candidates, G)
                    break
            SingleLogger().info("\033[94m" + "Speedup to the origin: %6.4f %%"%(100 * (self.base_cost - cost) / self.base_cost) + "'\033[0m'")
            SingleLogger().info("\033[94m" + "Best speedup: %d th acception, speed up to the origin: %6.4f %%"%(len(best_strategy), 100 * (self.base_cost - best_cost) / self.base_cost) + "'\033[0m'")
            with open("/root/search_trajectory.txt", "a") as f:
                f.write(str(time.time()) + ": {}".format(100 * (self.base_cost - best_cost) / self.base_cost) + "\n")


    def accept_or_not(self, cost, new_cost):
        # prob = min(1, (math.exp(beta * (cost - new_cost))))
        if cost > new_cost:
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
            if args.ucb_visual:
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