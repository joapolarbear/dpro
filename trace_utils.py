import os
import json
import xlsxwriter
import traceback

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
	'''
	Return: a list of traces
	'''
	with open(traces_path, 'r') as fp:
		_traces = json.load(fp)
	if isinstance(_traces, dict):
		traces = _traces.get("traceEvents")
	elif isinstance(_traces, list):
		traces = _traces
	else:
		raise ValueError("The output file not follow the stardard chrome tracing format!: " + traces_path)
	return traces

def _comm_is_subtask(comm_name):
	return comm_name.split(".")[-1] in QueueType

def return_stat(traces):
	""" Basic Statistic """
	name2sta = {}
	cat2sta = {}
	for event in traces:
		name = event["name"]
		if "Comm" in name and _comm_is_subtask(name):
			#! sub-task comm nodes, add partition key to the name
			main_task_name = ".".join(name.split(".")[:-1])
			name += "." + event["tid"]
			#! record the partition keys in the main-task node
			#	for the ease of looking up partition keys
			if main_task_name in name2sta:
				if "key" in name2sta[main_task_name]:
					name2sta[main_task_name]["key"].add(event["tid"])
				else:
					name2sta[main_task_name]["key"] = {event["tid"]}
			else:
				name2sta[main_task_name] = {"key" : {event["tid"]}}
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
		if "Comm" in name and _comm_is_subtask(name):
			name += "." + event["tid"]
		name2sta[name]["var"] += pow(event["dur"] / 1000.0 - name2sta[name]["avg"], 2)

	for name, statistic in name2sta.items():
		statistic["var"] = statistic["var"] / float(statistic["cnt"])
	return name2sta, cat2sta

def export2xlsx(_stats, _dir, _order=False, filename=None):
	''' Export the statitic results to an XLSX file

	Parameters
	----------
	_stats: list
		A list of statitic results
	_dir: str
		The directory to store the XLSX file
	_order: bool
		TODO(huhanpeng): delete
	'''
	workbook = xlsxwriter.Workbook(os.path.join(_dir, 'statistic.xlsx' if filename is None else filename + ".xlsx"))
	for _stat in _stats:
		worksheet = workbook.add_worksheet()
		row = 0
		header = []
		for name, statistic in _stat.items():
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

def split_name(_name):
	try:
		name_split = _name.split(".")
		_local_rank = int(name_split[0].split("rank")[1])
		raw_name = ".".join(name_split[1:])
	except:
		raise ValueError("split_name error: " + _name)
	return _local_rank, raw_name

def lookup_stat(_all_name2sta, _name, _field="avg"):
	''' look up data from the entire worker stat info
	'''
	if "rank" not in _name:
		return _all_name2sta[_name][_field]
	_local_rank, _raw_name = split_name(_name)
	return _all_name2sta["traces"][_local_rank][_raw_name][_field]

def _del_prefix(name):
	#! delete the prefix rank0.
	return ".".join(name.split(".")[1:])

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

def combine_add_traces(_traces, _local_rank, _tmp_traces, _comm_filter=None):
	for event in _traces:
		if event["cat"] == "Comm" and _comm_filter is not None and event["args"]["name"] not in _comm_filter:
			#! Only show the communication nodes belonging to comm_filter if comm_filter is set
			continue
		event['pid'] = "rank%d."%_local_rank + str(event['pid'])
		event['name'] = "rank%d."%_local_rank + str(event['name'])
		_tmp_traces.append(event)

def combine_process_one_path(_path, _comm_filter=None):
	tmp_traces = []
	_path = os.path.abspath(_path)
	if os.path.isdir(_path):
		#! If its a directory of a worker, read all traces of all GPUs
		root, dirs, _ = list(os.walk(_path))[0]
		#! avoid that directory is like `worker/0/`
		if len(dirs) == 0:
			raise ValueError("Given path should be the root directory of a worker traces"
				" or the path of one trace TXT file")
		dirs = sorted(dirs)		
		for _dir in dirs:
			path_dict = return_path_dict(os.path.join(root, _dir))
			local_rank = path_dict["local_rank"]
			traces = read_traces(path_dict["trace_path"])
			combine_add_traces(traces, local_rank, tmp_traces, _comm_filter=_comm_filter)
	else:
		#! Or, read just one trace file
		traces = read_traces(_path)
		local_rank = _path.split('/')[-2]		
		combine_add_traces(traces, local_rank, tmp_traces, _comm_filter=_comm_filter)
	return tmp_traces


