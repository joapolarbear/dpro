import os 
import ujson as json
import networkx as nx
import traceback
import time
import bisect

from dag_utils import QueueType
from trace_utils import *
from progress_utils import progressBar
import logger_utils
import debug_utils

FIXED_GAP_us = 5

class Deivce:
	def __init__(self, device_name, _replayer, infi_para=False):
		self.replayer = _replayer
		self.device_time = 0
		self.device_name = device_name
		#! infi_para devices allow to excute in infinite parallelism
		self.infi_para = infi_para

	def reset(self):
		self.device_time = 0

	def exct(self, name, _last_end_time, step_idx):
		''' Execute one op on this device 

		Parameters
		----
		name: str
			The name of the op to run
		_last_end_time: float
			The time when this op can start to run, 
			i.e., all the dependent ops have been done
		'''
		### for debug
		_ts = time.time()

		if not self.infi_para:
			_last_end_time = max(_last_end_time, self.device_time)

		if "Sync" in name or name == "END":
			#! No event is generated, but has successors
			self.mark_as_exct(name, _last_end_time, _last_end_time)
			return 

		#! Some BW nodes of dag is not profiled, ignore them.
		try:
			avg = self.replayer.traceM.lookup_stat(None, None, name)
		except:
			self.replayer.logger.warning("%s is not in _name2sta" % name)
			self.mark_as_exct(name, _last_end_time, _last_end_time)
			return

		#! really start to execute
		pid = parse_pid_from_name(name)
		cat = parse_cat_from_name(name)
		raw_name = parse_rawname_from_name(name)
		delay, ratio = self.get_delay_para(name)
		duration = (1000.0 * max(avg + delay, 0)) * ratio
		
		start_t = _last_end_time

		event = {
					"name": raw_name,
					"ts": start_t,
					"dur": duration,
					"pid": pid,
					"cat": cat,
					"ph": "X",
					"tid": cat,
					"args": {
						"name": name,
						"cnt": step_idx
					}
				}
		_id = 0
		# for prev, _ in self.replayer.dag.in_edges(name):
		# 	event["args"]["input%d"%_id] = prev
		# 	_id += 1
		self.replayer.rst_traces.append(event)

		self.mark_as_exct(name, start_t, start_t + duration)
		### TODO (huhanpeng): modify after fine-tune update
		### Should be safe now, would be overitted by an UPDATE OP with larger UPDATE index
		if "UPDATE" in name:
			#! current UPDATE of this GPU ends
			pid = parse_pid_from_name(name)
			self.replayer.step_end_time[pid] = start_t + duration

		#! TODO: for debug
		debug_utils.DebugRecorder().debug_record(name, _ts, self.device_name, "run")

	def mark_as_exct(self, name, _start_t, _end_time):
		''' Mark that the op has been executed '''
		self.device_time = _end_time
		self.replayer.node_status.pop(name)
		prev_cat = parse_cat_from_name(name)
		for _succ in self.replayer.dag.successors(name):
			if _succ in self.replayer.node_status:
				_status = self.replayer.node_status[_succ]
				### Calculate the start time
				if "SEND" in name and "RECV" in _succ:
					### For Send->Recv edge, there exist some overlap
					### TODO (huhanpeng): how do decide the end time of the RECV event
					try:
						avg = self.replayer.traceM.lookup_stat(None, None, _succ)
					except:
						self.replayer.logger.warning("%s is not in _name2sta" % _succ)
						avg = 0
					_status["last_end"] = _end_time if _status["last_end"] is None else max(_end_time - avg, _status["last_end"])
				else:
					### Apply the gap between two nodes
					# gap = self.replayer.dag.edges[name, _succ]["gap"] if "gap" in self.replayer.dag.edges[name, _succ] else 0
					gap = 0
					next_cat = parse_cat_from_name(_succ)
					for key, value in self.replayer.dag.nodes[name].items():
						if "GAP" in key:
							### e.g. "gap.operator.operator"
							key_s = key.split("GAP")
							if prev_cat == key_s[0] and next_cat == key_s[1]:
								gap += value
					_status["last_end"] = _end_time + gap if _status["last_end"] is None else max(_end_time + gap, _status["last_end"])

				### Whether the dependency has met
				_status["in_degree"] -= 1
				if _status["in_degree"] == 0:
					self.replayer.insert_next_node(_succ, _status["last_end"])

	def get_delay_para(self, name_):
		#! Get the delay parameters.
		delay = 0
		ratio = 1.0
		if self.replayer.delay_dict is not None:
			if name_ in self.replayer.delay_dict:
				delay = self.replayer.delay_dict[name_]["delay"]
				ratio = self.replayer.delay_dict[name_]["ratio"]
			elif "DELAY_ALL_CMP" in self.replayer.delay_dict and parse_cat_from_name(name_) == "operator":
				delay = self.replayer.delay_dict["DELAY_ALL_CMP"]["delay"]
				ratio = self.replayer.delay_dict["DELAY_ALL_CMP"]["ratio"]
			elif "DELAY_ALL_COMM" in self.replayer.delay_dict and parse_cat_from_name(name_) == "Comm":
				delay = self.replayer.delay_dict["DELAY_ALL_COMM"]["delay"]
				ratio = self.replayer.delay_dict["DELAY_ALL_COMM"]["ratio"]
			elif "DELAY_ALL" in self.replayer.delay_dict:
				delay = self.replayer.delay_dict["DELAY_ALL"]["delay"]
				ratio = self.replayer.delay_dict["DELAY_ALL"]["ratio"]
		return delay, ratio

class Replayer:
	def __init__(self, collector, _step_num):
		self.clct = collector
		self.traceM = collector.traceM
		self.dag = collector.trail_dag
		self.step_num = _step_num

		assert self.traceM.dir_level == DirLevel.TRIAL
		self.logger = logger_utils.SingleLogger()
		self.leaf_dirs = self.clct.all_prefix_list()
		### Delay information, the unit of 'delay' field should be ms
		self.delay_dict = None

		### maintain node status
		self.node_status = {}
		self.accessed = None

		self.device_dict = {}
		self.reset_replayer()

	def pre_prepare(self):
		''' Initialize nodes that need to be replayed first
		'''
		self.accessed = set()
		self.node_status = dict([(n, {"in_degree": self.dag.in_degree(n), "last_end": None}) for n in self.dag.nodes()])
		#! prepare next_nodes
		for n, _status in self.node_status.items():
			if _status["in_degree"] == 0 and n not in self.accessed:
				try:
					assert "Comm" not in n
				except:
					print(n)
					raise
				pid = parse_pid_from_name(n)
				_last_end = self.step_end_time[pid] if _status["last_end"] is None else _status["last_end"]
				self.insert_next_node(n, _last_end)
	
	def replay_one_iter(self, step_idx):	
		self.pre_prepare()
		assert len(self.next_nodes) != 0
		while True:
			if len(self.next_nodes) == 0:
				break
			(n, t) = self.next_nodes.pop(0)
			device = self.name2device(n)
			device.exct(n, t, step_idx)
		assert len(self.node_status) == 0

	def replay(self, _output=True):
		self.reset_replayer()
		_ts = time.time()
		for step_idx in range(self.step_num):
			self.replay_one_iter(step_idx)
		self.logger.info("Take %f s to replay one iteration" % ((time.time() - _ts)/float(self.step_num)))
		if _output:
			self.output_traces()
		
	def replayAndDelay(self, delay_dict_, _ouput=False):
		self.reset_replayer()
		self.delay_dict = delay_dict_
		self.replay_one_iter(0)
		if _ouput:
			self.output_traces()
		return self.step_end_time

	def insert_next_node(self, n, t):
		''' This is acutally equal to a scheduler of an **Engine**
		n: node string
		t: start time of this node
		'''
		self.accessed.add(n)
		def ret_priority(n_):
			### The smaller the rank is, the higher priority the node has
			if "FW" in n_:
				return 0
			elif "OUTPUT" in n_:
				return 1
			elif "BW" in n_:
				return 2
			elif "UPDATE_" in n_:
				return 3
			else:
				return 4

		def _schedule(_a, _b):
			_ap = ret_priority(_a[0])
			_bp = ret_priority(_b[0])
			if _ap == _bp:
				### The same priority, compare the start time		
				return _a[1] < _b[1]
			else:
				return _ap < _bp
			

		#! TODO (huhanpeng): if OPs are ranked, 
		# just to substitute func to compare their ranks.
		self.insort_right(self.next_nodes, (n, t), func=_schedule)

	def insort_right(self, a, x, lo=0, hi=None, func=None):
		"""Insert item x in list a, and keep it sorted assuming a is sorted.
		If x is already in a, insert it to the right of the rightmost x.
		Optional args lo (default 0) and hi (default len(a)) bound the
		slice of a to be searched.
		"""
		def fun_cmp(x1, x2):
			if func is None:
				return x1 < x2
			else:
				return func(x1, x2)

		if lo < 0:
			raise ValueError('lo must be non-negative')
		if hi is None:
			hi = len(a)
		while lo < hi:
			mid = (lo+hi)//2
			if fun_cmp(x, a[mid]):
				hi = mid
			else:
				lo = mid+1
		a.insert(lo, x)

	def name2device(self, n):
		pid = parse_pid_from_name(n)
		cat = parse_cat_from_name(n)
		if "SEND" in n:
			device_id = gen_long_name(pid, cat, "SEND")
		elif "RECV" in n:
			device_id = gen_long_name(pid, cat, "RECV")
		else:
			device_id = gen_long_name(pid, cat)

		if device_id not in self.device_dict:
			self.device_dict[device_id] = self.create_device(device_id)

		return self.device_dict[device_id]		
		
	def create_device(self, device_name, infi_para=False):
		d = Deivce(device_name, self, infi_para=infi_para)
		return d

	def reset_replayer(self):
		self.step_end_time = dict([(_d, 0.0) for _d in self.leaf_dirs])
		### nodes to be executed, in a **partial order**
		self.next_nodes = []
		self.rst_traces = []
		### Reset all devices
		for _, device_ in self.device_dict.items():
			device_.reset()

	def output_traces(self):
		#! Output the synthetic traces.
		rst = {
			"traceEvents": self.rst_traces,
			"displayTimeUnit": "ms"
		}
		TraceManager(self.rst_traces, DirLevel.TRIAL).get_iter_time()
		with open(os.path.join(self.clct.pm.path, "synthetic.json"), 'w') as f:
			json.dump(rst, f, indent=4)

"""
class Replayer:
	'''Used to replay distributed training

	Parameters
	----------
	_all_name2sta: dict
		A dict storing all static info on onw worker.
		The value of key "traces" is list of dict, 
		The i th dict element stores the name2sta of the GPU with local_rank=i
	_dirs: list
		The list of GPU folder on this worker.
	_dag: networkx.classes.digraph.DiGraph
		The combined execution order graph on one worker.
	_step_num: int
		The number of steps to replay.
	_path: str
		The path to store the synthetic traces.
	_logger:logging.Logger
		The logger used to output logging
	'''
	def __init__(self, _all_name2sta, _dirs, _dag, _step_num, _path, _logger):
		self.all_name2sta = _all_name2sta
		self.dag = _dag
		self.step_num = _step_num
		self.path = _path
		self.logger = _logger
		self._dirs = _dirs
		self.step_end_time = dict([(int(_d), 0.0) for _d in self._dirs])

		#! Inital next_nodes, start replay from I/O nodes of each GPU
		self.next_nodes = None
		self.rst_traces = []
		self.loop_cnt = 0
		self.delay_dict = None

	def record_end_time(self, _name, _end_time):
		''' Record the latest end time for current node
		
		Parameters
		----------
		_name: str
			The name of the node we want to record.
		_end_time: int or long
			The latest end time.
		'''
		if "rank" not in _name:
			if _name not in self.all_name2sta:
				self.all_name2sta[_name] = {"latest_end" : _end_time}
			else:
				self.all_name2sta[_name]["latest_end"] = _end_time
			return 0

		_local_rank, _raw_name = split_name(_name)
		_name2sta = self.all_name2sta["traces"][_local_rank]
		if _raw_name not in _name2sta:
			_name2sta[_raw_name] = {"latest_end" : _end_time}
		else:
			_name2sta[_raw_name]["latest_end"] = _end_time
		return 0

	def has_arrived(self, _name, level=0):
		'''Check whether a node has been replayed.
		The node may be in other GPUs, so use all_name2sta
		
		Parameters
		----------
		level: int
			If set to 0, return arrive_flag as True only if the OP has been output
			Else if set to -1, return arrive_flag as True if the OP has been accessed
		Return
		------
		_arrive: bool
			A node has been replayed if set True
		_local_rank: int
			The local rank of the GPU which generates this op.
		_raw_name: str
			If the node is not a shared node, e.g. Sync nodes, 
			return its original name without local rank
		_name2sta: dict
		step_end_time: int
			The latest step end time of current GPU.
		'''
		assert level == 0 or level == -1
		if "rank" not in _name:
			#! shared nodes across GPUs
			_arrive = _name in self.all_name2sta and "latest_end" in self.all_name2sta[_name] and self.all_name2sta[_name]["latest_end"] >= 0
			return _arrive, None, None, None, 0
		else:
			_local_rank, _raw_name = split_name(_name)
			_name2sta = self.all_name2sta["traces"][_local_rank]
			_arrive = _raw_name in _name2sta and "latest_end" in _name2sta[_raw_name] and _name2sta[_raw_name]["latest_end"] >= 0
			return _arrive, _local_rank, _raw_name, _name2sta, self.step_end_time[_local_rank]

	def _reproduce_one_op(self, name, reserve=False, FIXED_GAP_us=10):
		''' Process one op, if the op has been replayed, ignore it, 
		or process all its dependent upstream nodes in a DFS manner,
		and process its successor nodes in a BFS manner, by adding them to a set.

		Parameters
		----------
		reserve: bool --> TODO, to be deleted
			If this is set True, denotes this is a call from a child node to its parent node
				!!!require the graph ends with one node (otherwise some nodes may be missed)!!!
				!!!A dense graph (or the recursion depth is too large)!!!
		FIXED_GAP_us: int
			A synthetic fixed gap between two steps.
		'''
		self.logger.debug("Name: %s, call parents?: %s" % (name, "True" if reserve else "False"))
		#! avoid repeated processing
		arrive_flag, _local_rank, raw_name, _name2sta, _last_end_time = self.has_arrived(name)
		if arrive_flag:
			#! When being re-called from its successors, this node must have finished
			#! Directly return the end time of the node
			return lookup_stat(self.all_name2sta, name, "latest_end")

		self.loop_cnt += 1
		#! Mark as arrival with -1
		self.record_end_time(name, -1)
		for u, v in self.dag.in_edges(name):
			arrive_flag, _, _, _, _ = self.has_arrived(u)
			if arrive_flag:
				_last_end_time = max(_last_end_time, lookup_stat(self.all_name2sta, u, "latest_end"))
			else:
				#! Use recursive/dfs to process parents, be
				_last_end_time = max(_last_end_time, self._reproduce_one_op(u, reserve=True))

		def call_successor(_name):
			# if not reserve: 
			for _succ in self.dag.successors(_name):
				arrive_flag, _, _, _, _ = self.has_arrived(_succ, level=-1)
				if not arrive_flag:
					self.next_nodes.add(_succ)

		#! Get the delay parameters.
		delay = 0
		ratio = 1.0
		if self.delay_dict is not None:
			if name in self.delay_dict:
				delay = self.delay_dict[name]["delay"]
				ratio = self.delay_dict[name]["ratio"]
			elif "DELAY_ALL_CMP" in self.delay_dict and ("FW" in name or "BW" in name or "STEP" in name):
				delay = self.delay_dict["DELAY_ALL_CMP"]["delay"]
				ratio = self.delay_dict["DELAY_ALL_CMP"]["ratio"]
			elif "DELAY_ALL_COMM" in self.delay_dict and ("PUSH" in name or "PULL" in name):
				delay = self.delay_dict["DELAY_ALL_COMM"]["delay"]
				ratio = self.delay_dict["DELAY_ALL_COMM"]["ratio"]
			elif "DELAY_ALL" in self.delay_dict:
				delay = self.delay_dict["DELAY_ALL"]["delay"]
				ratio = self.delay_dict["DELAY_ALL"]["ratio"]

		#! All dependent nodes have been processed
		if "I/O" in name:
			cat = tid = "I/O"
			pid = name
		elif "Comm" in name:
			cat = "Comm"
			_name_split = name.split(".")
			assert len(_name_split) >= 2
			if _name_split[-2] in QueueType:
				#! sub-task
				pid = ".".join(_name_split[:-2])
				tid = _name_split[-1]
			else:
				#! main task
				pid = name
				tid = "total"
		elif "FW" in name or "BW" in name or "STEP" in name:
			pid = "rank%d.operator"%_local_rank
			cat = "operator"
			tid = "tmp"
		elif "OUTPUT" in name or "Sync" in name:
			#! No event is generated, but has successors
			self.record_end_time(name, _last_end_time)
			call_successor(name)
			return _last_end_time	
		else:
			raise ValueError("Unknown node name: " + name)
		
		#! Some BW nodes of dag is not profiled, ignore them.
		if raw_name not in _name2sta or "avg" not in _name2sta[raw_name]:
			self.logger.warning("%s is not in _name2sta" % name)
			self.record_end_time(name, _last_end_time)
			call_successor(name)
			return _last_end_time

		_dur = (1000.0 * (_name2sta[raw_name]["avg"] + delay)) * ratio
		self.rst_traces.append({
				"name": name,
				"ts": _last_end_time + FIXED_GAP_us ,
				"dur": _dur,
				"pid": pid,
				"cat": cat,
				"ph": "X",
				"tid": tid
			})
		
		self.record_end_time(name, _last_end_time + FIXED_GAP_us + _dur)
		if "STEP" in name:
			#! current STEP of this GPU ends
			self.step_end_time[_local_rank] = _last_end_time + FIXED_GAP_us + _dur
		else:
			call_successor(name)
		return _last_end_time + FIXED_GAP_us + _dur

	def replay(self):
		self.resetReplayer(_continue=False)
		for step_idx in range(self.step_num):
			time_before_gen = time.time()
			while len(self.next_nodes) > 0:
				self._reproduce_one_op(self.next_nodes.pop())

			if step_idx == 0:
				self.logger.info("One step time: %s ms" % (str([_t / 1000.0 for _t in self.step_end_time.values()])))
				self.logger.info("Take %f s and %d loops to produce %d events" % 
				(time.time() - time_before_gen, self.loop_cnt, len(self.rst_traces)))

			#! prepare for the next step
			self.resetReplayer()
			
		self.outputTraces()

	def outputTraces(self):
		#! Output the synthetic traces.
		rst = {
			"traceEvents": self.rst_traces,
			"displayTimeUnit": "ms"
		}
		get_iter_time(self.rst_traces, self.logger)
		with open(os.path.join(self.path, "synthetic.json"), 'w') as f:
			json.dump(rst, f, indent=4)

	def resetReplayer(self, _continue=True):
		''' Reset the replayer to prepare for a new step.
		Parameters
		----------
		_continue: bool
			if set True, do not reset the time counter, continue for the next step
		'''
		self.next_nodes = set(["rank%s."%i + "I/O" for i in self._dirs])
		for key, value in self.all_name2sta.items():
			if key == "traces":
				for _name2sta in value.values():
					for _, _v in _name2sta.items():
						_v["latest_end"] = -2
			else:
				value["latest_end"] = -2
		self.loop_cnt = 0
		if not _continue:
			self.step_end_time = dict([(int(_d), 0.0) for _d in self._dirs])

	def replayAndDelay(self, delay_dict, _output=True):
		''' Replay one step with latency
		Parameters
		----------
		delay_dict: dict
			E.g. {nodename: {"delay": 10, "ratio": 1.0}}
			The key should be the node name to which we want to add delay.
			The value is a dict, the final time of this op is `(t + delay) * ratio`
			`delay` means the absolute time (in ms) you want to delay.
			`ratio` denotes the relative times of time you want to delay, 1.0 means the same.
			If nodename == DELAY_ALL, all nodes whould be delayed
		'''
		self.delay_dict = delay_dict
		self.resetReplayer(_continue=False)
		while len(self.next_nodes) > 0:
			self._reproduce_one_op(self.next_nodes.pop())
		#! prepare for the next step
		self.resetReplayer()
		step_end_time = [_t / 1000.0 for _t in self.step_end_time.values()]
		if _output:
			self.outputTraces()
			self.logger.info("One step time: %s ms" % (str(step_end_time)))
		return step_end_time
"""

