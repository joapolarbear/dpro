import os 
import json
import argparse
import networkx as nx
import traceback
import time


import logger_utils
from trace_utils import read_traces, return_stat, export2xlsx
from dag_utils import gen_dag_from_gml_and_traces, dag_longest_path, visualize_gml, gen_gpu_dag

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, 
					choices=["statistic", "graph", "combine", "compare", "critical", "timeline", "gpu_graph", "reproduce"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
parser.add_argument("--path", type=str, required=True, help="The path of the dir you want to analyze. Must be a dirctory.")
parser.add_argument("--path2", type=str, required=False, help="The path of the file you want to analyze.")

parser.add_argument("--sort", type=bool, default=True, help="Sorted in descending order")
parser.add_argument("--head", type=int, default=None, help="Print the first few lines")
parser.add_argument("--xlsx", type=bool, default=False, help="Output XLSX file of the statistic results")
parser.add_argument("--del_queue", action="store_true", help="If set True, delete the queue time in communicateion traces. ")
parser.add_argument("--logging_level", type=int, default="20", help="Logging level")
parser.add_argument("--clean", action="store_true", help="Flush the log file")
parser.add_argument("--gen_step_num", type=int, default="1", help="Default step numbers to reproduce.")
args = parser.parse_args()

logger = logger_utils.get_logger(args)
logger.info(args)

def return_path_dict(root_path):
	''' Map the paths of each file from its name
	Args:
		root_path: the root path for one GPU
	'''
	assert os.path.isdir(root_path)
	root_path = os.path.abspath(root_path)
	__root, _, files = list(os.walk(root_path))[0]
	path_dict = {}
	for __file in files:
		cur_path = os.path.join(__root, __file)
		if "bps_trace" in __file:
			path_dict["trace_path"] = cur_path		
		elif __file == 'dag.gml':
			# mygraph = nx.read_gml(cur_path)
			path_dict['gml_path'] = cur_path
		elif __file == 'temp.json':
			pass
		else:
			pass
	path_dict["local_rank"] = int(__root.split("/")[-1])
	return path_dict

assert os.path.isdir(args.path)

""" Read traces """
if args.option != 'critical' and args.option != 'combine' and args.option != 'compare':
	path_dict = return_path_dict(args.path)
	traces = read_traces(path_dict["trace_path"])
	name2sta, cat2sta = return_stat(traces)

if args.option == "statistic":
	""" Output the statistic results """
	# \TODO: device id
	def output(_name2sta):
		logger.info("Profile Statistics.")
		logger.info("===================")
		logger.info("%-60s\t Total Count\t Time (ms)\t Min Time (ms)\t Max Time (ms)\t Avg Time (ms)\t Variance (ms^2)" % "Name")
		logger.info("%-60s\t -----------\t ---------\t -------------\t -------------\t -------------\t ---------------" % "----")
		line_cnt = 0
		for name, statistic in _name2sta:
			if (args.head and line_cnt >= args.head):
				break		
			logger.info("%-60s\t %11d\t %9.4f\t %12.4f\t %13.4f\t %13.4f\t %13.4f" % 
					(name,
					statistic["cnt"],
					statistic["time"],
					statistic["min_t"],
					statistic["max_t"],
					statistic["avg"],
					statistic["var"]
					))
			line_cnt += 1

	# output(sorted(name2sta.items(), lambda x, y: cmp(x[1]["avg"], y[1]["avg"])))
	if args.sort:
		sort_sta = sorted(name2sta.items(), key=lambda x: x[1]["avg"], reverse=True)
	else:
		sort_sta = name2sta.items()
	output(sort_sta)
	if args.xlsx:
		export2xlsx(name2sta, '/'.join(args.path.split('/')[:-1]))

	# Group by category
	logger.info("")
	logger.info("Group by category")
	logger.info("===================")
	line_cnt = 0
	for cat, statistic in cat2sta.items():
		if (args.head and line_cnt >= args.head):
				break
		logger.info("Category: %-10s\t The most time-consuming OP: %-30s -> %13.4f (ms)" % (cat, statistic["max_name"], statistic["max_t"] / 1000.0))
		line_cnt += 1

if args.option == "graph":
	mygraph = nx.read_gml(path_dict["gml_path"])
	visualize_gml(mygraph)

if args.option == "critical":
	''' 
	Args:
		-- args.path: the dir of a worker, which contains multiple folders 
						storing traces of GPUs of this worker
	'''
	root, dirs, _ = list(os.walk(args.path))[0]

	#! used to store all dags generated from GPUs
	graphs = []
	for _dir in dirs:
		local_rank = int(_dir)
		path_dict = return_path_dict(os.path.join(root, _dir))
		traces = read_traces(path_dict["trace_path"])
		name2sta, cat2sta = return_stat(traces)
		dag = gen_dag_from_gml_and_traces(name2sta, path_dict["gml_path"], args.del_queue, local_rank, logger)
		dag_longest_path(dag, local_rank, logger, weight="weight", default_weight=0)
		graphs.append(dag)

	graph = nx.compose_all(graphs)
	dag_longest_path(graph, -1, logger, weight="weight", default_weight=0)

if args.option == "timeline":
	raise NotImplementedError()

if args.option == "reproduce":
	''' Re-generate the timeline according to the dependency 
	graph with time for each node.
	Args:
		--path: the root path for one GPU
		--para_degree: required parallelism degree.
		--compute_delay:
		--comm_delay: 
		--gen_step_num: number of steps we want to generate.
	'''
	gpu_dag, max_para_degree = gen_gpu_dag(traces, name2sta, path_dict, args.del_queue, logger, _pretty=True)
	rst_traces = []

	def _del_prefix(name):
		return ".".join(name.split(".")[1:])

	#! Use set to avoid repeated node names.
	_successors = set()
	_running_nodes = set()
	_pre_depend_nodes = set()

	FIXED_GAP_us = 10
	

	def check_dependency(name, _overall_time = 0):
		''' Check whether all the parents of `name` has finished.
		If not, add corresponding parent to `_pre_depend_nodes.
		Args:
			name: the name of node
			_overall_time: used when the node has no dependency
		Return:
			ret: if set True, all the parents of `name` has finished.
			_last_end_time: the end time of last finished parents
		'''
		assert "rank" in name or "Sync" in name
		# if "bertmodel0_word_embed_embedding0" in name:
		# 	print([u for u, _ in gpu_dag.in_edges(name)])
		ret = None
		_last_end_time = 0
		if len(gpu_dag.in_edges(name)) == 0:
			#! I/O or other nodes with no parents
			return True, _overall_time
		for u, v in gpu_dag.in_edges(name):
			if _del_prefix(u) in name2sta and "arrive" in name2sta[_del_prefix(u)] and name2sta[_del_prefix(u)]["arrive"] == 1:
				try:
					_last_end_time = max(_last_end_time, name2sta[_del_prefix(u)]["latest_end"])
				except:
					print(u)
					raise ValueError()
				continue
			else:
				_pre_depend_nodes.add(u)
				ret = False
		if ret is None:
			ret = True
		return ret, _last_end_time

	def record_end_time(_name, _end_time):
		''' Record the latest end time for current node
		Args:
			_name: the name of the node we want to record
			_end_time: the latest end time
		'''
		if _name not in name2sta:
			name2sta[_name] = {"latest_end" : _end_time}
		else:
			name2sta[_name]["latest_end"] = _end_time
		return 0

	def execute_one_node(_cur_name, _start_time):
		''' Execute one node, generate a trace, 
		* add this node to `_running_nodes`,
		* add the successors of this node to `_successors`
		Args:
			_cur_name: the name of the node to be executed
			_start_time: the start time of this node, **excluding the gap**
		'''
		raw_name = _del_prefix(_cur_name)
		_running_nodes.add(_cur_name)

		if "I/O" in _cur_name:
			pid = cat = tid = "I/O"
		elif "Comm" in _cur_name:
			cat = "Comm"
			pid = _cur_name
			# TODO for each partition.
			tid = "total"
		elif "FW" in _cur_name or "BW" in _cur_name:
			pid = "operator"
			cat = "operator"
			tid = "tmp"
			if raw_name not in name2sta or "avg" not in name2sta[raw_name]:
				logger.warning("%s is not in name2sta" % _cur_name)
				record_end_time(raw_name, _start_time)
				return _start_time
		elif "OUTPUT" in _cur_name:
			#! No event is generated
			_successors.update(set(gpu_dag.successors(_cur_name)))
			record_end_time(raw_name, _start_time)
			return _start_time
		elif _cur_name == "Sync":
			#! No event is generated, no successors nodes
			return _start_time			
		else:
			raise ValueError("Unknown node name" + _cur_name)

		_dur = 1000 * name2sta[raw_name]["avg"]
		rst_traces.append({
				"name": _cur_name,
				"ts": _start_time + FIXED_GAP_us ,
				"dur": _dur,
				"pid": pid,
				"cat": cat,
				"ph": "X",
				"tid": tid
			})

		record_end_time(raw_name, _start_time + FIXED_GAP_us + _dur)
		_successors.update(set(gpu_dag.successors(_cur_name)))
		return _start_time + FIXED_GAP_us + _dur

	overall_time = 0.0
	
	for _ in range(args.gen_step_num):
		step_end_time = 0.0
		finished_nodes = set()
		next_nodes = {"rank%d."%path_dict["local_rank"] + "I/O"}
		step_start_t = time.time()
		while len(next_nodes) > 0:
			#! check the op in the `next_nodes`, if the dependency can be met
			#  these OPs can be executed in parallel
			# TODO: take max_para_degree into account
			for name in next_nodes:
				is_run, start_time = check_dependency(name, overall_time)
				if is_run:
					_node_end_time = execute_one_node(name, start_time)
					step_end_time = max(step_end_time, _node_end_time)

			#! if some nodes' successors are in `_running_nodes`, 
			#	then the dependency of these sunccesors should not be satisfied, conflict.
			assert len(_running_nodes.intersection(_successors)) == 0
		
			# #! if `_running_nodes` is empty, infinite loop
			# assert len(_running_nodes) != 0
			for _name in _running_nodes:
				if _del_prefix(_name) not in name2sta:
					name2sta[_del_prefix(_name)] = {"arrive" : 1}
				else:
					name2sta[_del_prefix(_name)]["arrive"] = 1

			prev_len = len(next_nodes)
			finished_nodes.update(_running_nodes)
			_pre_depend_nodes.difference_update(finished_nodes)
			next_nodes.difference_update(_running_nodes)
			next_nodes.update(_successors)
			next_nodes.update(_pre_depend_nodes)

			assert not (len(next_nodes) == prev_len and len(_running_nodes) == 0)

			_running_nodes = set()
			_successors = set()
			_pre_depend_nodes = set()

		#! prepare for the next step
		logger.info("Time to geenrate one step of traces: %f s" % (time.time() - step_start_t))
		overall_time = step_end_time
		for _, v in name2sta.items():
			v["arrive"] = 0

	#! Output the synthetic traces.
	rst = {
		"traceEvents": rst_traces,
		"displayTimeUnit": "ms"
	}
	with open(os.path.join(args.path, "synthetic.json"), 'w') as f:
		json.dump(rst, f, indent=4)

if args.option == "gpu_graph":
	''' Construct the real graph running on GPU
	and calculate the maximum parallelism degree.
	Args:
		--path: the root path for one GPU
	'''
	gpu_dag = gen_gpu_dag(traces, name2sta, path_dict, args.del_queue, logger)
	

'''below options use special --path'''
# TODO
if args.option == "combine":
	traces = read_traces(args.path)
	traces2 = read_traces(args.path2)
	rank = args.path.split('/')[-2]
	rank2 = args.path2.split('/')[-2]

	rst_path = '/'.join(args.path.split("/")[:-2]) + '/' + "combined.json"
	rst_traces = {"traceEvents": []}
	for event in traces:
		event['pid'] = rank + '.' + str(event['pid'])
		rst_traces["traceEvents"].append(event)
	for event in traces2:
		event['pid'] = rank2 + '.' + str(event['pid'])
		rst_traces["traceEvents"].append(event)

	with open(rst_path, 'w') as f:
		json.dump(rst_traces, f, indent=4)

if args.option == "compare":
	if not (args.path and args.path2):
		raise ValueError("To compare two files, two paths must be given")
	traces = [read_traces(args.path), read_traces(args.path2)]
	name2sta = [return_stat(_traces)[0] for _traces in traces]
	name2compare = {}
	for name, statistic in name2sta[0].items():
		if name not in name2sta[1]:
			continue
		name2compare[name] = {
				"avg_absolute": name2sta[1][name]["avg"] - statistic["avg"],
				"avg_relative": (name2sta[1][name]["avg"] - statistic["avg"]) / statistic["avg"]
			}

	if args.sort:
		sort_sta = sorted(name2compare.items(), key=lambda x: x[1]["avg_relative"], reverse=True)
	else:
		sort_sta = name2compare.items()

	print("Compare following two files:")
	print("File 1: " + args.path)
	print("File 2: " + args.path2)
	print("===================")
	print("%-60s\t Absolute Avg Time Increase (ms)\t Relative Avg Time Increase" % "Name")
	line_cnt = 0
	for name, compare in sort_sta:
		if (args.head and line_cnt >= args.head):
			break	
		print("%-60s\t %24.4f\t %24.4f" %
				(name, compare["avg_absolute"], compare["avg_relative"]))
		line_cnt += 1













