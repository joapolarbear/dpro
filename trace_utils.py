import json
import xlsxwriter

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