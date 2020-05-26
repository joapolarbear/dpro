import networkx as nx
import random
import math

from replay import Replayer
from trace_utils import *

class State:
	def __init__(self):
		self.reward = 0
		self.cnt = 0
		self.childs = None
		self.parent = None

class Optimizer:
	def __init__(self, collector):
		self.clct = collector
		self.dag = self.relabel_dag_node(self.clct.trail_dag)

		self.search_space = None
		### Used to cache the node attribtue
		self.node_attr_cache = {}

	def relabel_dag_node(self, G): 
		def relabel_func(old_label):
			if "BW" in old_label or "FW" in old_label or "Comm" in old_label:
				layer_name = parse_layer_name(old_label)
				layer_index = self.clct.para_dict.parse_layer_index(layer_name)
				return ("[%d]"%layer_index).join(old_label.split(layer_name))
			else:
				return old_label
		return nx.relabel_nodes(G, relabel_func)

	def concat_name(self, u_, v_):
		return "%s+%s"%(u_, v_)

	def combine_avg(self, ua, va):
		### TODO: huhanpeng, key component
		# raise NotImplementedError()
		return (ua + va) / 0.8

	def combine_gap(self, ug, vg):
		### TODO: huhanpeng, key component
		# raise NotImplementedError()
		return (ug + vg) / 0.8

	def get_node_attr(self, n, attr_):
		if attr_ in self.node_attr_cache[n]:
			return self.node_attr_cache[n][attr_]
		else:
			return 0

	def op_fusion(self, G, u_, v_):
		### Cache the node attributes in case they will be used when de-fuse
		if u_ not in self.node_attr_cache:
			self.node_attr_cache[u_] = G.nodes[u_]
		if v_ not in self.node_attr_cache:
			self.node_attr_cache[v_] = G.nodes[v_]

		new_name = self.concat_name(u_, v_)

		### Add new nodes and get the attibute
		if new_name in self.node_attr_cache:
			G.add_node(new_name, **self.node_attr_cache[new_name])
		else:
			G.add_node(new_name)
			### Calculate the new attribute
			self.combine_nodes_attr(G, new_name, u_, v_)
			### cache the attribute
			self.node_attr_cache[new_name] = G.nodes[new_name]

		### Update edges
		for in_, _ in G.in_edges(u_):
			if in_ != v_:
				G.add_edge(in_, new_name)
		for in_, _ in G.in_edges(v_):
			if in_ != u_:
				G.add_edge(in_, new_name)

		for out_ in G.successors(u_):
			if out_ != v_:
				G.add_edge(new_name, out_)
		for out_ in G.successors(v_):
			if out_ != u_:
				G.add_edge(new_name, out_)

		### Remove current nodes
		G.remove_node(u_)
		G.remove_node(v_)

	def combine_nodes_attr(self, G, target, u_, v_):
		### In graph G, combine the attributes of u_ and v_, store the results in G as the attributes of target
		G.nodes[target]["avg"] = self.combine_avg(self.get_node_attr(u_, "avg"), self.get_node_attr(v_, "avg"))
		G.nodes[target][GAP_STR_OP2OP] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2OP), self.get_node_attr(v_, GAP_STR_OP2OP))
		G.nodes[target][GAP_STR_OP2COMM] = self.combine_gap(self.get_node_attr(u_, GAP_STR_OP2COMM), self.get_node_attr(v_, GAP_STR_OP2COMM))

	def parse_dependency_from_name(self, G, fuse_name, attr=True):
		ns = fuse_name.split("+")
		prev_name = None
		for _name in ns:
			### Handle the edge, different from that in op_fusion
			### Here we based on the self.dag, the original dag
			for in_, _ in self.dag.in_edges(_name):
				if in_ not in ns:
					G.add_edge(in_, fuse_name)
			for out_ in self.dag.successors(_name):
				if out_ not in ns:
					G.add_edge(fuse_name, out_)

			### Handle the attribuet
			if attr:
				if prev_name is None:
					prev_name = _name
					continue
				self.combine_nodes_attr(G, fuse_name, prev_name, _name)
				prev_name = fuse_name

	def add_new_node(self, G, new_name):
		if new_name in self.node_attr_cache:
			G.add_node(new_name, **self.node_attr_cache[new_name])
			self.parse_dependency_from_name(G, new_name, attr=False)
		else:
			G.add_node(new_name)
			self.parse_dependency_from_name(G, new_name, attr=True)
			### cache the attribute
			self.node_attr_cache[new_name] = G.nodes[new_name]

	def op_defusion(self, G, target, pos):
		assert "+" in target
		ns = target.split("+")
		pos += 1
		left = "+".join(ns[:pos])
		right = "+".join(ns[pos:])
		self.add_new_node(G, left)
		self.add_new_node(G, right)
		G.remove_node(target)

	def MCTS(self):
		pass

	def MCMC_search(self):
		### TODO (huhanpeng): is shallow copy is enough ???
		G = self.dag.copy()
		### Get the replay results of current state
		replayer = Replayer(dag=G, _step_num=1, leaf_dirs=self.clct.all_prefix_list(), dump_path=self.clct.pm.path)
		step_end_time_ms = [t / 1000 for t in replayer.replayAndDelay(None, _ouput=False).values()]
		cost = max(step_end_time_ms)

		while True:
			self.init_search_space(G)
			while True:
				op, target, next_ = self.pick_strategy()

				G_star = G.copy()
				if op == "+":
					### Fusion two nodes
					self.op_fusion(G_star, target, next_)
				elif op == "-":
					self.op_defusion(G_star, target, next_)

				### Start to replay
				replayer = Replayer(dag=G_star, _step_num=1, leaf_dirs=self.clct.all_prefix_list(), dump_path=self.clct.pm.path)
				cost_star = max([t / 1000 for t in replayer.replayAndDelay(None, _ouput=False).values()])

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

	def init_search_space(self, G):
		self.search_space = []
		for n in G:
			if "+" in n:
				### This a fused node
				ns = n.split("+")
				cat = parse_cat_fine_grained(ns[0])
				pid = parse_pid_from_name(ns[0])
				split_n_idx = 0
				while True:
					self.search_space.append(("-", n, split_n_idx))
					if split_n_idx >= (len(ns) - 2):
						break
					split_n_idx += 1
			else:
				### Nodes that have never been fused
				cat = parse_cat_fine_grained(n)
				pid = parse_pid_from_name(n)

			if cat not in ["operator.FW", "operator.BW"]:
				### TODO (huhanpeng): only fuse FW or BW nodes now
				continue

			for succ_ in G.successors(n):
				_pid = parse_pid_from_name(succ_)
				_cat = parse_cat_fine_grained(succ_)
				if pid != _pid or cat != _cat:
					continue
				if len(G.in_edges(succ_)) == 1:
					self.search_space.append(("+", n, succ_))

	def pick_strategy(self):
		return random.choice(self.search_space)






