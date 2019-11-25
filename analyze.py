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

def gen_dag_from_traces(traces, rank=0):
	dag = nx.DiGraph()
	#! Add a output node with avg time 0.0
	#! \TODO, automatically generate an output/softmax node when generating traces
	dag.add_node("OUTPUT", time=0.0)
	name2sta, cat2sta = return_stat(traces)
	for event in traces:
		name = event["name"]
		event_args = event["args"]
		if name != event_args["name"]:
			# -- ignore nodes which are sub-tasks of communication nodes
			continue
		node_name = "rank%d_"%rank + name
		avg_time = name2sta[name]["avg"]
		if node_name not in dag.nodes: 
			dag.add_node(node_name, time=avg_time)

		if "BW" in name and len(event_args.items()) == 1:
			#! this is the first BW node
			#  add edge:(FW. --> OUTPUT) first
			value = "FW." + name.split("BW.")[1]
			value_name = "rank%d_"%rank + value
			value_avg_time = name2sta[value]["avg"] if value in name2sta else 0.0
			if value_name not in dag.nodes:
				dag.add_node(value_name, time=value_avg_time)
			if (value_name, "OUTPUT") not in dag.edges:
				dag.add_edge(value_name, "OUTPUT", weight=value_avg_time)

			#!  add edge:(OUTPUT, BW) then
			if ("OUTPUT", node_name) not in dag.edges:
				dag.add_edge("OUTPUT", node_name, weight=0.0)
			continue

		#! for other normal nodes, create edges for each `input0`
		for key, value in event_args.items():
			#! \TODO, delete arg
			if "input" not in key and "arg" not in key:
				continue
			value_name = "rank%d_"%rank + value
			#! \TODO: why some vulues don't exist.
			value_avg_time = name2sta[value]["avg"] if value in name2sta else 0.0
			if value_name not in dag.nodes:
				dag.add_node(value_name, time=value_avg_time)
			
			#! If this edge has exist, ignore it
			if "Comm." in value:
				if (value_name, 'Sync') not in dag.edges:
					# -- for the edge from Comm to FW., assume there is a Sync node for all GPU cards.
					dag.add_edge(value_name, 'Sync', weight=value_avg_time)
					#~ Delete edges from Sync to FW. nodes or this graph will not be an directed *acyclic* graph
					# dag.add_edge('Sync', node_name)
			elif (value_name, node_name) not in dag.edges:
				dag.add_edge(value_name, node_name, weight=value_avg_time)
	return dag

def gen_dag_from_gml_and_traces(trace_path, gml_path, rank=0):
	traces = read_traces(trace_path)
	name2sta, cat2sta = return_stat(traces)
	mygraph = nx.read_gml(gml_path)
	dag = nx.DiGraph()

	def _read_stat(node_name, _assert=False):
		#! delete rank
		node_name = '.'.join(node_name.split('.')[1:])
		return name2sta[node_name]["avg"] if node_name in name2sta else 0.0
	def add_fwd_prefix(name):
		return "rank%d.FW."%rank + name
	def add_bw_prefix(name):
		return "rank%d.BW."%rank + name
	def add_rank(name):
		return "rank%d."%rank + name

	#! Add edges from BW nodes to Comm nodes.
	def handel_comm_node(name):
		if "var" in mygraph.nodes[name] and mygraph.nodes[name]["var"] != '[]':
			for comm_node in mygraph.nodes[name]["var"]:
				comm_node = add_rank(comm_node)
				if args.del_queue == True:
					#! Add all the sub-tasks to the dag.
					prev_node = None
					for suffix in QueueType[:-1]:
						cur_node = comm_node + '.' + suffix
						if _read_stat(cur_node) == 0:
							continue
						if prev_node:
							dag.add_edge(prev_node, cur_node, weight=_read_stat(prev_node, _assert=False))
						else:
							dag.add_edge(add_bw_prefix(name), cur_node, weight=_read_stat(add_bw_prefix(name)))
						prev_node = cur_node
					if prev_node is None:
						#! There is no subtask events, by default, take all the subtasks as one event.
						dag.add_edge(add_bw_prefix(name), comm_node, weight=_read_stat(add_bw_prefix(name)))
						dag.add_edge(comm_node, "Sync", weight=_read_stat(comm_node))
					else:
						dag.add_edge(prev_node, "Sync", weight=_read_stat(prev_node, _assert=False))
				else:
					#! Take all the subtasks as one event.
					dag.add_edge(add_bw_prefix(name), comm_node, weight=_read_stat(add_bw_prefix(name)))
					dag.add_edge(comm_node, "Sync", weight=_read_stat(comm_node))

	#! Add FW, BW nodes, add BW-->Comm edges (no need to add Comm nodes, auto add)
	for _node in mygraph.nodes:
		dag.add_node(add_fwd_prefix(_node))
		dag.add_node(add_bw_prefix(_node))
		handel_comm_node(_node)

	#! Add all edges with weight
	for (src, dest) in mygraph.edges:
		dag.add_edge(add_fwd_prefix(src), add_fwd_prefix(dest), weight=_read_stat(add_fwd_prefix(src)))
		dag.add_edge(add_bw_prefix(dest), add_bw_prefix(src), weight=_read_stat(add_bw_prefix(dest)))

	#! Add a output node with avg time 0.0
	#! \TODO, automatically generate an output/softmax node when generating traces
	dag.add_node(add_rank("OUTPUT"))
	for _node in dag.nodes:
		if len(list(dag.successors(_node))) == 0 and "FW." in _node:
			_bw_node = '.'.join(_node.split('.')[2:])
			dag.add_edge(_node, add_rank("OUTPUT"), weight=_read_stat(_node))
			dag.add_edge(add_rank("OUTPUT"), add_bw_prefix(_bw_node), weight=0.0)
	
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
		# traces = read_traces(path_dict["trace_path"])
		# graphs.append(gen_dag_from_traces(traces, local_rank))
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













