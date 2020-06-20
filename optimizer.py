import networkx as nx
import random
import math

from replay import Replayer
from trace_utils import *
from dag_utils import *

class GraphExpand(Enum):
    NOT=0
    PARTIAL=1
    FULLY=2

MAX_TREE_DEPTH = 1000
MAX_LOOP = 100
UCB_GAMMA = 1

class GraphState:
	def __init__(self, depth):
		self.visit_cnt = 1
		self.quality = 0

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
	def __init__(self, collector, memory_budget=None):
		self.clct = collector
		self.dag = self.relabel_dag_node(self.clct.trail_dag)

		self.base_cost = self.evaluate(self.dag)
		### Used to cache the node attribtue
		self.node_attr_cache = {}

		### Budget
		self.memory_budget = memory_budget

	def relabel_dag_node(self, _dag): 
		def relabel_func(old_label):
			if "BW" in old_label or "FW" in old_label or "Comm" in old_label:
				layer_name = parse_layer_name(old_label)
				layer_index = self.clct.para_dict.parse_layer_index(layer_name)
				return ("[%d]"%layer_index).join(old_label.split(layer_name))
			else:
				return old_label
		return nx.relabel_nodes(_dag, relabel_func)

	def concat_name(self, u_, v_):
		return "%s+%s"%(u_, v_)

	def combine_avg(self, ua, va):
		### TODO (huhanpeng): key component
		# raise NotImplementedError()
		return (ua + va) / 0.8

	def combine_gap(self, ug, vg):
		### TODO (huhanpeng): key component
		### Use max to avoid one input is zero, 
		### some how for the new gap x, ug < x < ug + vg, vg < x < ug + vg
		return max(max((ug + vg) / 0.8, ug), vg)

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
		_dag.nodes[target]["avg"] = self.combine_avg(self.get_node_attr(u_, "avg"), self.get_node_attr(v_, "avg"))
		_dag.nodes[target][GAP_STR_OP2OP] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2OP), self.get_node_attr(v_, GAP_STR_OP2OP))
		_dag.nodes[target][GAP_STR_OP2COMM] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2COMM), self.get_node_attr(v_, GAP_STR_OP2COMM))

	def combine_attr(self, target, attr1, attr2):
		### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
		target["avg"] = self.combine_avg(attr1["avg"], attr2["avg"])

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
				self.combine_attr(attrs, attrs, self.node_attr_cache[ns[idx]])
			### set and cache the attribute
			nx.set_node_attributes(_dag, {new_name:attrs})
			self.cache_node_attr(new_name, _dag.nodes[new_name])

	def op_defusion(self, _dag, target, pos):
		assert "+" in target
		ns = target.split("+")
		pos += 1
		left = "+".join(ns[:pos])
		right = "+".join(ns[pos:])

		### For the edges related to traget, re-connect them to left or right
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
		

	def evaluate(self, _dag):
		### Get the replay results of current state
		replayer = Replayer(dag=_dag, _step_num=1, 
				leaf_dirs=self.clct.all_prefix_list(), 
				dump_path=self.clct.pm.path,
				comm_backend=self.clct.comm_backend,
				byteps_graph=self.clct.byteps_graph)
		step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(None, _ouput=False).values()]
		return max(step_end_time_ms)

	def candidate_selection(self, GS):
		if isinstance(GS, GraphState):
			new_dag = self.apply_strategies(self.dag, GS.strategy)
		elif isinstance(GS, nx.DiGraph):
			new_dag = GS
		else:
			raise ValueError("Invalid type for input (type: {}), only GraphState and nx.DiGraph are allowed".format(type(GS)))

		cal_edge_cost(new_dag)
		critical_path = dag_longest_path(new_dag, None, weight="cost", default_weight=0, _debug_level=0)

		### Need to pick some candidates
		### TODO (huhanpeng): ??? how to decide which nodes to select as candiates
		MOST_FOCUSED_CANDIATES = 10
		critical_path = sorted(critical_path, key=lambda x: x[1], reverse=True)
		return [n for n, l in critical_path[:MOST_FOCUSED_CANDIATES]], new_dag

	def init_search_space(self, candidates, _dag):
		### TODO (huhanpeng): currently only consider fusion and split
		### 			Need to add quantization
		### 	Currently assumption: if a->b, belong to the same pid and cat, in_degree(b) = 1 then we can fuse a and b.
		search_space = []
		for n in candidates:
			if "+" in n:
				### This a fused node
				ns = n.split("+")
				cat = parse_cat_fine_grained(ns[0])
				pid = parse_pid_from_name(ns[0])
				pos = 0
				while True:
					search_space.append(("-", n, pos))
					if pos >= (len(ns) - 2):
						break
					pos += 1
			else:
				### Nodes that have never been fused
				cat = parse_cat_fine_grained(n)
				pid = parse_pid_from_name(n)

			if cat not in ["operator.FW", "operator.BW"]:
				### TODO (huhanpeng): only fuse FW or BW nodes now
				continue

			for succ_ in _dag.successors(n):
				_pid = parse_pid_from_name(succ_)
				_cat = parse_cat_fine_grained(succ_)
				if pid != _pid or cat != _cat:
					continue
				if len(_dag.in_edges(succ_)) == 1:
					search_space.append(("+", n, succ_))
		return search_space

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

	def pick_strategy(self, search_space):
		### TODO (huhanpeng): need some priority/heuristic
		return random.choice(search_space)

	def search(self):
		raise NotImplementedError()

class MCMCOptimizer(Optimizer):
	''' Markov Chain Monte Carlo algorithm'''
	def search(self):
		### TODO (huhanpeng): is shallow copy is enough ???
		G = self.dag.copy()
		cost = self.base_cost

		while True:
			candiates = self.candidate_selection(G)
			search_space = self.init_search_space(candidates, new_dag)
			while True and len(search_space) > 0:
				G_star = self.apply_strategies(G, self.pick_strategy(search_space))

				### Start to replay
				cost_star = self.evaluate(G_star)

				if self.accept_or_not(cost, cost_star):
					if op == "+":
						SingleLogger().info("Fuse %s and %s" % (target, next_))
					elif op == "-":
						SingleLogger().info("Split %s at %dth op" % (target, next_))
					G = G_star
					cost = cost_star
					break

	def accept_or_not(self, cost, new_cost):
		### beta = 1 means if new_cost is smaller, definitely change to the new strategy, otherwise, there is some probability
		beta = 1
		prob = min(1, (math.exp(beta * (cost - new_cost))))
		if random.random() < prob:
			return True
		else:
			return False

class MCTSOptimizer(Optimizer):
	''' Monte Carlo Tree Search '''
	def __init__(self, *args, **kwargs):
		super(MCTSOptimizer, self).__init__(*args, **kwargs)
		self.loop_cnt = 0
		self.GS_root = None

	def search(self):
		### Initialize the root graph state
		self.GS_root = GraphState(depth=0)
		self.GS_root.strategy = []

		while self.check_loop_time() and self.check_loop_num():
			GS = self.tree_policy(self.GS_root)
			reward = self.default_policy(GS)
			self.backpropagation(GS, reward)
			self.visualize_tree()
		return 

	def visualize_tree(self):
		def iter_print(GS):
			sys.stdout.write("*")
			if GS.childs is None:
				return
			for idx, child in enumerate(GS.childs):
				if idx > 0:
					sys.stdout.write("\n{}{}".format(" "*int(1 + 4/2) + "|    "*int(GS.depth), "├─" if idx < (len(GS.childs) -1) else "└─"))
				else:
					sys.stdout.write("{}".format('─'*4))
				iter_print(child)

		iter_print(self.GS_root)
		sys.stdout.write("\n")


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
		while not self.terminal(GS):
			self.check_search_space(GS)
			action = self.pick_strategy(GS.space)[0]

			GS_c = GraphState(depth=(GS.depth+1))
			GS_c.strategy = GS.strategy.copy()
			GS_c.strategy.append(action)
			GS = GS_c

		### Evaluate the final graph
		if GS.iter_time is None:
			new_dag = self.apply_strategies(self.dag, GS.strategy)
			cost = self.evaluate(new_dag)
			GS.iter_time = cost
		else:
			cost = GS.iter_time
		SingleLogger().info("Evaluate the strategy %s" % (str(GS.strategy)))
		return math.exp(self.base_cost - cost)

	def backpropagation(self, GS, reward):
		GS.quality += reward
		GS.visit_cnt += 1
		if GS.depth == 0:
			return
		else:
			self.backpropagation(GS.parent, reward)
		
	def best_UCB(self, GS):
		GS_opt = c_opt = None
		for GS_c in GS.childs:
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
		self.check_search_space(GS)
		### TODO (huhanpeng): how to pick with some heuristic
		for idx in range(len(GS.space)):
			if GS.space[idx][1] == 0:
				GS.space[idx][1] += 1
				return GS.space[idx][0]
		return None

	def check_search_space(self, GS):
		### TODO (huhanpeng): we can do some pruning here
		if GS.space is None:
			candidates, new_dag = self.candidate_selection(GS)
			GS.space = [[action, 0] for action in self.init_search_space(candidates, new_dag)] ### The integer value is used as a counter

	def terminal(self, GS):
		self.check_search_space(GS)
		if GS.depth > MAX_TREE_DEPTH or len(GS.space) == 0:
			return True
		else:
			return False






