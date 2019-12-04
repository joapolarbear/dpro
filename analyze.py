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

""" Read traces and prepare statitic info"""
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

def _has_arrived(raw_name):
	return raw_name in name2sta and "arrive" in name2sta[raw_name] and name2sta[raw_name]["arrive"] == 1

def _has_finished(raw_name):
	return _has_arrived(raw_name) and "latest_end" in name2sta[raw_name] and name2sta[raw_name]["latest_end"] >= 0

step_end_time = 0
loop_cnt = 0

def _reproduce_one_op(gpu_dag, name, next_nodes_list, reserve=False, FIXED_GAP_us=10):
	'''
	Args:
		reserve: child call its parents
			!!!require the graph ends with one node (otherwise some nodes may be missed)!!!
			!!!A dense graph (or the recursion depth is too large)!!!
		FIXED_GAP_us: fixed gap between two steps
	'''
	logger.debug("Name: %s, call parents?: %s" % (name, "True" if reserve else "False"))
	raw_name = _del_prefix(name)
	#! avoid repeated processing
	if _has_arrived(raw_name):
		#! When being re-called from its successors, this node must have finished
		#! Directly return the end time of the node
		return name2sta[raw_name]["latest_end"]
	else:
		#! mark arrival here
		if raw_name not in name2sta:
			name2sta[raw_name] = {"arrive" : 1}
		else:
			name2sta[raw_name]["arrive"] = 1

	global step_end_time, loop_cnt
	loop_cnt += 1
	_last_end_time = step_end_time
	for u, v in gpu_dag.in_edges(name):
		if _has_arrived(_del_prefix(u)):
			_last_end_time = max(_last_end_time, name2sta[_del_prefix(u)]["latest_end"])
		else:
			#! Use recursive/dfs to process parents, be
			_last_end_time = max(_last_end_time, _reproduce_one_op(gpu_dag, u, next_nodes_list, reserve=True))

	def call_successor(_name):
		if not reserve:
			for _succ in gpu_dag.successors(_name):
				# assert not _has_finished(_succ)
				next_nodes_list.add(_succ)

	#! All dependent nodes have been processed
	if "I/O" in name:
		pid = cat = tid = "I/O"
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
		pid = "operator"
		cat = "operator"
		tid = "tmp"
		if raw_name not in name2sta or "avg" not in name2sta[raw_name]:
			logger.warning("%s is not in name2sta" % name)
			record_end_time(raw_name, _last_end_time)
			call_successor(name)
			return _last_end_time
	elif "OUTPUT" in name:
		#! No event is generated
		record_end_time(raw_name, _last_end_time)
		call_successor(name)
		return _last_end_time
	elif name == "Sync":
		#! No event is generated, no successors nodes
		step_end_time = _last_end_time
		return _last_end_time			
	else:
		raise ValueError("Unknown node name" + name)

	_dur = 1000 * name2sta[raw_name]["avg"]
	rst_traces.append({
			"name": name,
			"ts": _last_end_time + FIXED_GAP_us ,
			"dur": _dur,
			"pid": pid,
			"cat": cat,
			"ph": "X",
			"tid": tid
		})
	record_end_time(raw_name, _last_end_time + FIXED_GAP_us + _dur)
	call_successor(name)
	return _last_end_time + FIXED_GAP_us + _dur

if args.option == "reproduce":
	''' Re-generate the timeline according to the dependency 
	graph with time for each node.
	Args:
		--path: the root path for one GPU
		--compute_delay:
		--comm_delay: 
		--gen_step_num: number of steps we want to generate.
	'''
	gpu_dag, max_para_degree = gen_gpu_dag(traces, name2sta, path_dict, args.del_queue, logger, _pretty=True)
	rst_traces = []

	for step_idx in range(args.gen_step_num):
		time_before_gen = time.time()
		next_nodes = {"rank%d."%path_dict["local_rank"] + "I/O"}
		while len(next_nodes) > 0:
			_reproduce_one_op(gpu_dag, next_nodes.pop(), next_nodes)
		#! prepare for the next step
		if step_idx == 0:
			logger.info("One step time: %f ms" % (step_end_time / 1000.0))
		for _, v in name2sta.items():
			v["arrive"] = 0
		logger.info("One step of statistic: %f s, %d loops, %d events" % 
			(time.time() - time_before_gen, loop_cnt, len(rst_traces)))
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













