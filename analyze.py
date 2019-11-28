import os 
import json
import argparse
import networkx as nx
import traceback
import xlsxwriter

import logger_utils

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, 
					choices=["statistic", "graph", "combine", "compare", "critical"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
# parser.add_argument("--graph", type=bool, default=False, help="show the dependency graph")
parser.add_argument("--path", type=str, required=True, help="The path of the file you want to analyze.")
parser.add_argument("--path2", type=str, required=False, help="The path of the file you want to analyze.")

parser.add_argument("--sort", type=bool, default=True, help="Sorted in descending order")
parser.add_argument("--head", type=int, default=None, help="Print the first few lines")
parser.add_argument("--xlsx", type=bool, default=False, help="Output XLSX file of the statistic results")
parser.add_argument("--del_queue", action="store_true", help="If set True, delete the queue time in communicateion traces. ")
parser.add_argument("--logging_level", type=int, default="20", help="Logging level")
parser.add_argument("--logging_file", type=str, default="log.txt", help="Logging file")

args = parser.parse_args()
logger = logger_utils.get_logger(args)
logger.info(args)

QueueType = [
  "COORDINATE_REDUCE",
  "REDUCE",
  "COPYD2H",
  "PCIE_REDUCE",
  "COORDINATE_PUSH",
  "PUSH",
  "PULL",
  "COPYH2D",
  "COORDINATE_BROADCAST",
  "BROADCAST",
  "QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST"
]

def read_traces(traces_path):
	with open(traces_path, 'r') as fp:
		_traces = json.load(fp)
	if isinstance(_traces, dict):
		traces = _traces.get("traceEvents")
	elif isinstance(_traces, list):
		traces = _traces
	else:
		raise ValueError("The output file not follow the stardard chrome tracing format!: " + traces_path)
	return traces

def return_stat(traces):
	""" Basic Statistic """
	name2sta = {}
	cat2sta = {}
	for event in traces:
		name = event["name"]
		if name in name2sta:
			name2sta[name]["cnt"] += 1
			name2sta[name]["time"] += event["dur"] / 1000.0
			name2sta[name]["min_t"] = min(name2sta[name]["min_t"], event["dur"] / 1000.0)
			name2sta[name]["max_t"] = max(name2sta[name]["max_t"], event["dur"] / 1000.0)
		else:
			name2sta[name] = {"cnt": 1, "time": event["dur"] / 1000.0, 
				"min_t": event["dur"] / 1000.0, "max_t": event["dur"] / 1000.0,
				# \TODO: add `cat` field for communication traces
				# "cat": event["cat"] 
				"cat": event["name"].split(".")[0]
				}
	"""calculate the avg """
	for name, statistic in name2sta.items():
		statistic["avg"] = statistic["time"] / statistic["cnt"]
		statistic["var"] = 0.0
		cat = statistic["cat"]
		if cat in cat2sta:
			if statistic["avg"] > cat2sta[cat]["max_t"]:
				cat2sta[cat]["max_t"] = statistic["avg"]
				cat2sta[cat]["max_name"] = name
		else:
			cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name}
	"""calculate the variance"""
	for event in traces:
		name = event["name"]
		name2sta[name]["var"] += pow(event["dur"] / 1000.0 - name2sta[name]["avg"], 2)
	for name, statistic in name2sta.items():
		statistic["var"] = statistic["var"] / float(statistic["cnt"])
	return name2sta, cat2sta

def visualize_gml(graph, layout="circular"):
	import matplotlib.pyplot as plt
	if layout == "spectral":
		pos = nx.spectral_layout(graph, dim=2, scale=0.5)
	elif layout == "circular":
		pos = nx.circular_layout(graph)
	elif layout == "random":
		pos = nx.random_layout(graph)
	nx.draw(graph, pos, with_labels=True, font_size=6)
	plt.show()
	# import matplotlib.pyplot as plt; plt.ion()
	# import netgraph
	# netgraph.draw(graph)
	# plot_instance = netgraph.InteractiveGraph(graph, node_positions=pos)
	# node_positions = plot_instance.node_positions

def return_time_line(traces):
	name2time_line = {}
	for event in traces:
		name = event["name"]
		if name in name2time_line:
			name2time_line[name].append([(event["ts"], event["dur"])])
		else:
			name2time_line[name] = [(event["ts"], event["dur"])]
	for name in name2time_line:
		name2time_line[name].sort()
	return name2time_line

def export2xlsx(_dict, _dir, _order=False):
	workbook = xlsxwriter.Workbook(_dir + '/statistic.xlsx')
	worksheet = workbook.add_worksheet()
	row = 0
	header = []
	for name, statistic in _dict.items():
		if row == 0:
			# -- Output the header of the sheet
			col = 0
			worksheet.write(row, col, "Name")
			for key in statistic:
				col += 1
				header.append(key)
				worksheet.write(row, col, key)
		row += 1
		col = 0
		worksheet.write(row, col, name)
		for key in header:
			col += 1
			worksheet.write(row, col, statistic[key])
	workbook.close()

def gen_dag_from_gml_and_traces(trace_path, gml_path, rank=0):
	traces = read_traces(trace_path)
	name2sta, cat2sta = return_stat(traces)
	mygraph = nx.read_gml(gml_path)
	dag = nx.DiGraph()
	def add_prefix(name):
		return "rank%d."%rank + name
	def _read_stat(node_name, _assert=False):
		return name2sta[node_name]["avg"] if node_name in name2sta else 0.0

	for u, v in mygraph.edges:
		if "Comm" in u:
			if args.del_queue == True:
				prev_nodes = [_u for _u, _ in mygraph.in_edges(u)]
				assert len(prev_nodes) == 1
				prev_node = prev_nodes[0]
				for suffix in QueueType[-1:]:
					cur_node = u + '.' + suffix
					if _read_stat(cur_node) == 0:
						continue
					dag.add_edge(add_prefix(prev_node), add_prefix(cur_node), weight=_read_stat(prev_node))
					prev_node = cur_node
				dag.add_edge(add_prefix(prev_node), "Sync", weight=_read_stat(prev_node))
			else:
				dag.add_edge(add_prefix(u), "Sync", weight=_read_stat(u))
		else:
			dag.add_edge(add_prefix(u), add_prefix(v), weight= _read_stat(u))	
	for e in dag.edges.data("weight"):
		logger.debug(e)
	# visualize_gml(dag, layout="circular")
	return dag

def return_path_dict(root_path):
	#! Map the paths of each file from its name
	_, _, files = list(os.walk(root_path))[0]
	path_dict = {}
	for file in files:
		cur_path = os.path.join(root, _dir, file)
		if "bps_trace" in file:
			path_dict["trace_path"] = cur_path		
		elif file == 'dag.gml':
			# mygraph = nx.read_gml(cur_path)
			path_dict['gml_path'] = cur_path
		elif file == 'temp.json':
			pass
		else:
			pass
	return path_dict

if args.option == "statistic":
	""" Read traces """
	traces = read_traces(args.path)
	name2sta, cat2sta = return_stat(traces)

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
	mygraph = nx.read_gml(args.path)
	visualize_gml(mygraph)

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
		# try:
		# 	assert 
		# except:
		# 	raise ValueError("Op name: %s is not in the second file" % name)
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

if args.option == "critical":
	# -- args.path is the dir of a worker, 
	# -- which contains multiple folders storing traces of GPUs of this worker
	assert(os.path.isdir(args.path))
	root, dirs, _ = list(os.walk(args.path))[0]

	def dag_longest_path(G, local_rank, weight='weight', default_weight=0):
		critical_path = nx.algorithms.dag.dag_longest_path(G, weight=weight, default_weight=default_weight)
		prefix = "Critical Path of " + ("the Entire Graph: " if local_rank == -1 else "GPU-%d: " % local_rank)
		logger.info(prefix + " => ")
		path_length = 0
		for (u, v) in nx.utils.pairwise(critical_path):
			path_length += G[u][v].get(weight, default_weight)
			logger.info("%s -> %s: %f ms" % (u, v, G[u][v].get(weight, default_weight)))
		# logger.info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
		logger.info("Length of the " + prefix + "%12.4f ms\n" % path_length)
	#! used to store all dags generated from GPUs
	graphs = []
	for _dir in dirs:
		path_dict = return_path_dict(os.path.join(root, _dir))
		local_rank = int(_dir)
		dag = gen_dag_from_gml_and_traces(path_dict["trace_path"], path_dict["gml_path"], local_rank)
		graphs.append(dag)
		dag_longest_path(dag, local_rank, weight="weight", default_weight=0)

	graph = nx.compose_all(graphs)
	dag_longest_path(graph, -1, weight="weight", default_weight=0)

if args.option == "time_line":
	path_dict = return_path_dict(args.path)
	if "/" == args.path[-1]:
		local_rank = int(args.path.split("/")[-2])
	else:
		local_rank = int(args.path.split("/")[-1])
	dag = gen_dag_from_gml_and_traces(path_dict["trace_path"], path_dict["gml_path"], local_rank)
	traces = read_traces(path_dict["trace_path"])
	name2sta, cat2sta = return_stat(traces)
	time_line_dag = nx.DiGraph()













