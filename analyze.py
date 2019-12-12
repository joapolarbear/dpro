import os 
import json
import argparse
import networkx as nx
import traceback
import time


import logger_utils
from trace_utils import read_traces, return_stat, export2xlsx, lookup_stat, return_path_dict
from dag_utils import DAGManager, dag_longest_path, visualize_gml
from replay import Replayer

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, 
					choices=["statistic", "graph", "combine", "compare", "critical", "timeline", "reproduce"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
parser.add_argument("--path", type=str, required=True, help="The path of the dir you want to analyze. Must be a dirctory.")
parser.add_argument("--path2", type=str, required=False, help="The path of the file you want to analyze.")

parser.add_argument("--sort", type=bool, default=True, help="Sorted in descending order")
parser.add_argument("--head", type=int, default=None, help="Print the first few lines")
parser.add_argument("--xlsx", type=bool, default=False, help="Output XLSX file of the statistic results")
parser.add_argument("--del_queue", action="store_true", help="If set True, delete the queue time in communication traces. ")
parser.add_argument("--logging_level", type=int, default="20", help="Logging level")
parser.add_argument("--clean", action="store_true", help="Flush the log file")
parser.add_argument("--step_num", type=int, default="1", help="Default step numbers to reproduce.")
parser.add_argument("--pretty", action="store_true", help="Output necessary info if set")
parser.add_argument("--filter", type=str, default=None, help="Used to show part of communication operations, seperated with `,`.")
args = parser.parse_args()

logger = logger_utils.get_logger(args)
logger.info(args)

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
		dagmanager = DAGManager(os.path.join(root, _dir), local_rank, logger, args.del_queue)
		dagmanager.gen_dag_from_gml_and_traces()
		dag_longest_path(dagmanager.dag, local_rank, logger, weight="weight", default_weight=0)
		graphs.append(dagmanager.dag)

	graph = nx.compose_all(graphs)
	dag_longest_path(graph, -1, logger, weight="weight", default_weight=0)

if args.option == "timeline":
	raise NotImplementedError()

if args.option == "reproduce":
	''' Re-generate the timeline according to the dependency 
	graph with time for each node.
	Args:
		--path: the root path for one GPU
		--compute_delay:
		--comm_delay: 
		--step_num: number of steps we want to generate.
	'''	
	#! used to store all dags generated from GPUs
	worker_dag_list = []
	all_name2sta = {"traces": []}
	root, dirs, _ = list(os.walk(args.path))[0]
	dirs = sorted(dirs)

	for _dir in dirs:
		local_rank = int(_dir)
		dagmanager = DAGManager(os.path.join(root, _dir), local_rank, logger, args.del_queue)
		max_para_degree = dagmanager.gen_gpu_dag(_pretty=args.pretty)
		worker_dag_list.append(dagmanager.gpu_dag)
		all_name2sta["traces"].append(dagmanager.name2sta)

	def _name2rootname(_all_name2sta, _name, _QueueType, _root_rank):
		''' Find the corresponding op name in the root GPU
		'''
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

	#! Replay traces
	replayer = Replayer(_all_name2sta=all_name2sta, _local_size=len(dirs), _wk_dag=wk_dag, _step_num=args.step_num, _path=args.path, _logger=logger)
	replayer.replay()
	
'''below options use special --path'''
# TODO
if args.option == "combine":
	comm_filter = args.filter.split(",") if args.filter is not None else None
	rst_path = None
	rst_traces = {"traceEvents": []}
	def add_traces(_traces, _local_rank):
		for event in _traces:
			if event["cat"] == "Comm" and comm_filter is not None and event["args"]["name"] not in comm_filter:
				continue
			event['pid'] = "rank%d."%_local_rank + str(event['pid'])
			rst_traces["traceEvents"].append(event)

	if os.path.isdir(args.path):
		root, dirs, _ = list(os.walk(args.path))[0]
		dirs = sorted(dirs)
		rst_path = os.path.join(root, "combined.json")
		for _dir in dirs:
			path_dict = return_path_dict(os.path.join(root, _dir))
			local_rank = path_dict["local_rank"]
			traces = read_traces(path_dict["trace_path"])
			add_traces(traces, local_rank)
	else:
		rst_path = '/'.join(args.path.split("/")[:-2]) + '/' + "combined.json"
			
		traces = read_traces(args.path)
		local_rank = args.path.split('/')[-2]		
		add_traces(traces, local_rank)

		traces2 = read_traces(args.path2)
		local_rank2 = args.path2.split('/')[-2]
		add_traces(traces2, local_rank2)

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













