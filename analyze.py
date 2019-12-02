import os 
import json
import argparse
import networkx as nx
import traceback


import logger_utils
from trace_utils import read_traces, return_stat, export2xlsx
from dag_utils import gen_dag_from_gml_and_traces, dag_longest_path, visualize_gml

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
args = parser.parse_args()

logger = logger_utils.get_logger(args)
logger.info(args)

def return_path_dict(root_path):
	''' Map the paths of each file from its name
	Args:
		root_path: the root path for one GPU
	'''
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
	raise NotImplementedError()
	dag = gen_dag_from_gml_and_traces(name2sta, path_dict["gml_path"], args.del_queue, path_dict["local_rank"], logger)
	synthetic_traces = gen_timeline_from_dag(dag)

if args.option == "gpu_graph":
	''' Construct the real graph running on GPU
	and calculate the maximum parallelism degree.
	Args:
		--path: the root path for one GPU
	'''
	traces = sorted(traces, key=lambda x: (x["ts"], x["name"]), reverse=False)
	mygraph = gen_dag_from_gml_and_traces(name2sta, path_dict["gml_path"], args.del_queue, path_dict["local_rank"], logger)
	prefix = "rank%d."%path_dict["local_rank"]

	in_process_events = []
	max_para_degree = 1
	first = True
	start_time = None
	def relative_time(time):
		return (time - start_time) / 1000.0
	gpu_dag = nx.DiGraph()
	input_node_raw_names = [".".join(name.split(".")[2:]) for name in mygraph.successors(prefix + "I/O")]
	
	#! Go through one step of traces
	for event in traces:
		if first:
			logger.info("The first event - name: %s, ts: %s, dur: %s" %
				(event["name"], str(event["ts"]), str(event["dur"])))
			start_time = event["ts"]
			first = False

		#! only consider FW and BW nodes
		if event["cat"] != "operator":
			continue

		i = 0
		while True:
			if i >= len(in_process_events):
					break
			prev_event = in_process_events[i]
			assert event["ts"] >= prev_event["ts"]
			if event["ts"] >= prev_event["ts"] + prev_event["dur"]:
				#! prev event has ended, should be deleted from in_process_events
				del in_process_events[i]
				#! TODO: only add once, to verify
				gpu_dag.add_edge(prefix + prev_event["name"], prefix + event["name"], weight=name2sta[prev_event["name"]]["avg"])
			else:
				parent_list_of_prev = [(u, gpu_dag.edges[(u, v)]["weight"]) for u, v in gpu_dag.in_edges(prefix + prev_event["name"])]
				for u, w in parent_list_of_prev:
					gpu_dag.add_edge(u, prefix + event["name"], weight=w)
				i += 1

		if len(in_process_events) + 1 > max_para_degree:
			max_para_degree = len(in_process_events) + 1

		def in_process_events2str():
			s = ''
			for _event in in_process_events:
				_n, _ts, _te = _event["name"], _event["ts"], _event["ts"] + _event["dur"]
				s += "\n\t\t\t\t%-60s: %s~%s (%-13.4f ~ %-13.4f)" % (_n, str(_ts), str(_te), relative_time(_ts), relative_time(_te))
			return s

		if len(in_process_events) > 0:
			logger.info("%s (%-13.4f): D=%d => %-60s%s" %
				(event["ts"], relative_time(event["ts"]),
					len(in_process_events)+1,
					event["name"], 
					in_process_events2str()))

		#! Only go through one step
		if "BW" in event["name"] and event["name"].split("BW.")[1] in input_node_raw_names:
			#! the last computation nodes in one step
			pass
		else:
			in_process_events.append(event)
		if len(in_process_events) == 0:
			break

	logger.info("max_para_degree: %d" % max_para_degree)

	#! Then, read IO, Comm, OUTPUT, and Sync nodes
	def is_computation_node(_node):
		return "BW" in _node or "FW" in _node
	for u, v in mygraph.edges:
		if is_computation_node(u) and is_computation_node(v):
			#! ignore edges whose u v are both computation nodes
			continue
		gpu_dag.add_edge(u, v, weight=mygraph.edges[(u, v)]["weight"])

	dag_longest_path(gpu_dag, path_dict["local_rank"], logger, weight="weight", default_weight=0)


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













