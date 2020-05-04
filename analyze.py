import os 
import json

import networkx as nx
import traceback
import time
import sys

import logger_utils
from trace_utils import *
from dag_utils import *
from collect import Collector
from replay import Replayer
from progress_utils import progressBar
import arg_utils
import debug_utils

args = arg_utils.SingleArg().args
logger = logger_utils.SingleLogger(args.path.split(',')[0], 
	args.option, args.logging_level, 
	is_clean=args.clean, 
	show_progress=args.progress)
logger.info(args)
QueueType("NCCL")
debug_utils.DebugRecorder()

sys.setrecursionlimit(1000000)

path_list = args.path.split(',')
""" Read traces and prepare statitic info"""
if args.option not in ['critical', 'combine', 'compare', "replay", "topo_sort", "collect", "3dcompare"]:
	pm = PathManager(path_list[0])
	traces = read_traces(pm.search(FileName.TRACE))
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
		export2xlsx([name2sta], os.path.dirname(path_dict["trace_path"]))

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
	mygraph = nx.read_gml(pm.search(FileName.DAG))
	visualize_gml(mygraph)

if args.option == "critical":
	''' 
	Args:
		-- args.path: the dir of a worker, which contains multiple folders 
						storing traces of GPUs of this worker
	'''
	assert pm.dir_level == DirLevel.WORKER
	#! used to store all dags generated from GPUs
	graphs = []
	for _dir in pm.dirs:
		dagmanager = DAGManager(os.path.join(pm.path, _dir))
		dagmanager.gen_dag_with_prefix_weight()
		dag_longest_path(dagmanager.dag, dagmanager.pm, weight="weight", default_weight=0)
		graphs.append(dagmanager.dag)

	graph = nx.compose_all(graphs)
	dag_longest_path(graph, pm, weight="weight", default_weight=0)

if args.option == "timeline":
	raise NotImplementedError()

if args.option == "replay":
	''' Re-generate the timeline according to the dependency 
	graph with time for each node.
	Args:
		--path: the root path for 
		--step_num: number of steps we want to generate.
	'''	
	clct = Collector(path_list[0])
	trail_dag = clct.collect_dag(args)
	# clct.re_align_traces(trail_dag)
	clct.dump_traces()

	clct.add_gaps(trail_dag)

	# dag_longest_path(trail_dag, clct.pm, weight="weight", default_weight=0)

	### Replay traces
	logger.info("# Start to Replay")
	replayer = Replayer(
				collector=clct,
				dag=trail_dag, 
				_step_num=args.step_num)
	if args.sub_option is None:
		''' Directly replay '''
		replayer.replay()
	elif args.sub_option == "smlt_delay_cmp":
		''' Replay with computation delays'''
		delay_dict = {"DELAY_ALL_CMP": {"delay": 0, "ratio": args.delay_ratio}}
		step_end_time = replayer.replayAndDelay(delay_dict, _output=True)
	elif args.sub_option == "smlt_delay_comm":
		''' Replay with communication delays'''
		delay_dict = {"DELAY_ALL_COMM": {"delay": 0, "ratio": args.delay_ratio}}
		step_end_time = replayer.replayAndDelay(delay_dict, _output=True)
	elif args.sub_option == "map_delay":
		''' Replay and add delays to each node respectively.'''
		node_lists = list(wk_dag.nodes())
		total_len = len(node_lists)
		pgsbar = progressBar(start=0, end=total_len)
		idx = 0
		while idx < total_len:
			nodename = node_lists[idx]
			delay_dict = {nodename: {"delay": 10, "ratio": 1.0}}
			step_end_time = replayer.replayAndDelay(delay_dict, _ouput=False)
			logger.info("Delay %s ==> %s ==> %s critical path." % (nodename, str(step_end_time), "in" if nodename in critical_path else "not in"))
			if args.progress:
				pgsbar.showBar(idx)
			idx += 10
		
if args.option == "topo_sort":
	pm = PathManager(path_list[0])
	assert pm.dir_level == DirLevel.GPU
	local_rank = int(pm.path.split("/")[-1])
	dagmanager = DAGManager(pm.path, local_rank)
	dagmanager.gen_fw_bw_dag()

'''below options use special --path'''
if args.option == "compare":
	if len(path_list) < 2:
		raise ValueError("To compare two files, two paths must be given")
	if os.path.isfile(path_list[0]):
		traces = [read_traces(path_list[0]), read_traces(path_list[1])]
	else:
		clct = [Collector(path_list[0]), Collector(path_list[1])]
		traces = [c.iter_combine() for c in clct]
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

	name2sta.append(name2compare)
	if args.xlsx:
		def gen_sheet_name(l):
			if len(l) >= 31:
				l = l[-31:]
			return "_".join(l.split("/")[1:])

		sheet_name = [gen_sheet_name(l) for l in path_list]
		sheet_name.append("comparison")
		export2xlsx(name2sta, 
			os.path.abspath(path_list[0]) if os.path.isdir(path_list[0]) else os.path.dirname(path_list[0]), 
			filename="compare",
			sheet_name=sheet_name)

	logger.info("Compare following two files:")
	logger.info("File 1: " + path_list[0])
	logger.info("File 2: " + path_list[1])
	logger.info("===================")
	logger.info("%-100s\t Absolute Avg Time Increase (ms)\t Relative Avg Time Increase" % "Name")
	line_cnt = 0
	for name, compare in sort_sta:
		if (args.head and line_cnt >= args.head):
			break	
		logger.info("%-100s\t %24.4f\t %24.4f" %
				(name, compare["avg_absolute"], compare["avg_relative"]))
		line_cnt += 1

if args.option == "collect":
	clct = Collector(path_list[0])
	if args.sub_option == "combine":
		clct.iter_combine()
	elif args.sub_option == "iter_time":
		clct.iter_time()

### Output debug traces
debug_utils.DebugRecorder().dump_traces(path_list[0])



