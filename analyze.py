import os 
import json
import argparse
import networkx as nx
import traceback
import time


import logger_utils
from trace_utils import read_traces, return_stat, export2xlsx
from dag_utils import gen_dag_from_gml_and_traces, dag_longest_path, visualize_gml, gen_gpu_dag
from dag_utils import QueueType

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
parser.add_argument("--step_num", type=int, default="1", help="Default step numbers to reproduce.")
parser.add_argument("--pretty", action="store_true", help="Output necessary info if set")
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

""" Read traces and prepare statitic info"""
if args.option not in ['critical', 'combine', 'compare', "reproduce"]:
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
		dag = gen_dag_from_gml_and_traces(name2sta, path_dict["gml_path"], local_rank, args.del_queue, logger)
		dag_longest_path(dag, local_rank, logger, weight="weight", default_weight=0)
		graphs.append(dag)

	graph = nx.compose_all(graphs)
	dag_longest_path(graph, -1, logger, weight="weight", default_weight=0)

if args.option == "timeline":
	raise NotImplementedError()

def _del_prefix(name):
	#! delete the prefix rank0.
	return ".".join(name.split(".")[1:])

def split_name(_name):
	try:
		name_split = _name.split(".")
		_local_rank = int(name_split[0].split("rank")[1])
		raw_name = ".".join(name_split[1:])
	except:
		print(_name)
		raise
	return _local_rank, raw_name

def record_end_time(_all_name2sta, _name, _end_time):
	''' Record the latest end time for current node
	Args:
		_name: the name of the node we want to record
		_end_time: the latest end time
	'''
	if "rank" not in _name:
		if _name not in _all_name2sta:
			_all_name2sta[_name] = {"latest_end" : _end_time}
		else:
			_all_name2sta[_name]["latest_end"] = _end_time
		return 0

	_local_rank, _raw_name = split_name(_name)
	_name2sta = _all_name2sta["traces"][_local_rank]
	if _raw_name not in _name2sta:
		_name2sta[_raw_name] = {"latest_end" : _end_time}
	else:
		_name2sta[_raw_name]["latest_end"] = _end_time
	return 0

def has_arrived(_all_name2sta, _name):
	'''The node may be in other GPUs, so use _all_name2sta
	'''
	if "rank" not in _name:
		return _name in _all_name2sta and "latest_end" in _all_name2sta[_name] and _all_name2sta[_name]["latest_end"] >= 0
	_local_rank, _raw_name = split_name(_name)
	_name2sta = _all_name2sta["traces"][_local_rank]
	return _raw_name in _name2sta and "latest_end" in _name2sta[_raw_name] and _name2sta[_raw_name]["latest_end"] >= 0

def lookup_stat(_all_name2sta, _name, _field="avg"):
	''' look up data from the entire worker stat info
	'''
	if "rank" not in _name:
		return _all_name2sta[_name][_field]
	_local_rank, _raw_name = split_name(_name)
	return _all_name2sta["traces"][_local_rank][_raw_name][_field]

step_end_time = 0
loop_cnt = 0

def _reproduce_one_op(_rst_traces, _all_name2sta, _wk_dag, name, next_nodes_list, reserve=False, FIXED_GAP_us=10):
	'''
	Args:
		reserve: child call its parents
			!!!require the graph ends with one node (otherwise some nodes may be missed)!!!
			!!!A dense graph (or the recursion depth is too large)!!!
		FIXED_GAP_us: fixed gap between two steps
	'''
	logger.debug("Name: %s, call parents?: %s" % (name, "True" if reserve else "False"))
	
	#! avoid repeated processing
	if has_arrived(_all_name2sta, name):
		#! When being re-called from its successors, this node must have finished
		#! Directly return the end time of the node
		return lookup_stat(_all_name2sta, name, "latest_end")

	global step_end_time, loop_cnt
	loop_cnt += 1
	_last_end_time = step_end_time
	for u, v in _wk_dag.in_edges(name):
		if has_arrived(_all_name2sta, u):
			_last_end_time = max(_last_end_time, lookup_stat(_all_name2sta, u, "latest_end"))
		else:
			#! Use recursive/dfs to process parents, be
			_last_end_time = max(_last_end_time, _reproduce_one_op(_rst_traces, _all_name2sta, _wk_dag, u, next_nodes_list, reserve=True))

	def call_successor(_name):
		if not reserve:
			for _succ in _wk_dag.successors(_name):
				next_nodes_list.add(_succ)
	if "I/O" in name or "Comm" in name or "FW" in name or "BW" in name:
		_local_rank, raw_name = split_name(name)
		_name2sta = _all_name2sta["traces"][_local_rank]

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
	elif "FW" in name or "BW" in name:
		pid = "rank%d.operator"%_local_rank
		cat = "operator"
		tid = "tmp"
	elif "OUTPUT" in name or "Sync" in name:
		#! No event is generated
		record_end_time(_all_name2sta, name, _last_end_time)
		call_successor(name)
		return _last_end_time
	elif name == "END":
		#! No event is generated, no successors nodes
		step_end_time = _last_end_time
		return _last_end_time			
	else:
		raise ValueError("Unknown node name: " + name)

	
	if raw_name not in _name2sta or "avg" not in _name2sta[raw_name]:
		logger.warning("%s is not in _name2sta" % name)
		record_end_time(_all_name2sta, name, _last_end_time)
		call_successor(name)
		return _last_end_time

	_dur = 1000 * _name2sta[raw_name]["avg"]
	_rst_traces.append({
			"name": name,
			"ts": _last_end_time + FIXED_GAP_us ,
			"dur": _dur,
			"pid": pid,
			"cat": cat,
			"ph": "X",
			"tid": tid
		})
	record_end_time(_all_name2sta, name, _last_end_time + FIXED_GAP_us + _dur)
	call_successor(name)
	return _last_end_time + FIXED_GAP_us + _dur

if args.option == "reproduce":
	''' Re-generate the timeline according to the dependency 
	graph with time for each node.
	Args:
		--path: the root path for one GPU
		--compute_delay:
		--comm_delay: 
		--step_num: number of steps we want to generate.
	'''
	root, dirs, _ = list(os.walk(args.path))[0]
	#! used to store all dags generated from GPUs
	worker_dag_list = []
	
	all_name2sta = {"traces": []}

	dirs = sorted(dirs)

	for _dir in dirs:
		local_rank = int(_dir)
		path_dict = return_path_dict(os.path.join(root, _dir))
		traces = read_traces(path_dict["trace_path"])
		name2sta, cat2sta = return_stat(traces)
		gpu_dag, max_para_degree = gen_gpu_dag(traces, name2sta, path_dict, args.del_queue, logger, _pretty=args.pretty)
		worker_dag_list.append(gpu_dag)
		all_name2sta["traces"].append(name2sta)

	def _name2rootname(_all_name2sta, _name, _QueueType, _root_rank):
		name_split = _name.split(".")
		_local_rank = int(name_split[0].split("rank")[1])
		name_split[0] = "rank%d"%_root_rank
		name_split[-2] = _QueueType
		_raw_name = ".".join(name_split[1:-2])
		relative = int(name_split[-1]) - int(min(_all_name2sta["traces"][_local_rank][_raw_name]["key"]))
		name_split[-1] = str(int(min(_all_name2sta["traces"][_root_rank][_raw_name]["key"])) + relative)
		return  ".".join(name_split)

	#! Combine all worker_dag_list on one worker, build the dependency
	wk_dag = nx.compose_all(worker_dag_list)
	root_rank = len(dirs) - 1
	sync_edges = []
	for u, v in wk_dag.edges:
		if "COORDINATE_PUSH" in u and "COPYH2D" in v:
			sync_edges.append((u, v, True))
		elif "REDUCE" in u and ("BROADCAST" in v or "COORDINATE_BROADCAST" in v):
			sync_edges.append((u, v, False))
	for u, v, is_distr in sync_edges:
		if is_distr:
			wk_dag.add_edge(u, _name2rootname(all_name2sta, u, "PUSH", root_rank), weight=lookup_stat(all_name2sta, u))
			wk_dag.add_edge(_name2rootname(all_name2sta, u, "PULL", root_rank), v, weight=lookup_stat(all_name2sta, _name2rootname(all_name2sta, u, "PULL", root_rank)))
		else:
			name_split = u.split(".")
			name_split[-2] = "Sync"
			sync_name = ".".join(name_split[2:])
			wk_dag.add_edge(u, sync_name, weight=lookup_stat(all_name2sta, u))
			wk_dag.add_edge(sync_name, v, weight=0.0)

	rst_traces = []
	for step_idx in range(args.step_num):
		time_before_gen = time.time()
		next_nodes = set(["rank%d."%i + "I/O" for i in range(len(dirs))])
		while len(next_nodes) > 0:
			_reproduce_one_op(rst_traces, all_name2sta, wk_dag, next_nodes.pop(), next_nodes)
		#! prepare for the next step
		if step_idx == 0:
			logger.info("One step time: %f ms" % (step_end_time / 1000.0))
			logger.info("Take %f s and %d loops to produce %d events" % 
			(time.time() - time_before_gen, loop_cnt, len(rst_traces)))
		for key, value in all_name2sta.items():
			if key == "traces":
				for _name2sta in value:
					for _, _v in _name2sta.items():
						_v["latest_end"] = -1
			else:
				value["latest_end"] = -1
		loop_cnt = 0

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













