import os 
import json
import argparse
import networkx as nx
import traceback
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logger_utils
from progress_utils import progressBar

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--option", type=str, default="gpu_trace", 
					choices=["gpu_trace"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
parser.add_argument("--path", type=str, required=True, help="The paths of traces you want to analyze, support multiple paths seperated with comma.")
parser.add_argument("--logging_level", type=int, default="20", help="Logging level")
parser.add_argument("--clean", action="store_true", help="Flush the log file")
parser.add_argument("--progress", action="store_true", help="Show the progress bar if it is set, disable the std output")
args = parser.parse_args()

logger = logger_utils.get_logger(args)
logger.info(args)

def printIter(_iter, prefix=''):
	for _cmp in _iter:
		logger.info(prefix + _cmp)

def handle(path, platform):
	with open(path, 'r') as fp:
		s = fp.readlines()
	i = 0
	sta = {}
	while i < len(s):
		if "Device   Context    Stream" in s[i]:
			i += 1
			break
		i += 1
	while i < len(s):
		if len(s[i]) < 162:
			break
		try:
			stream_id = int(s[i][162:168])
		except:
			logger.info(len(s[i]), s[i-1])
			raise
		#! delete the index of each kernel, reduce the duplication number of each kernal
		#! only focus on the name of each kernal
		name = s[i][170:].split(" [")[0].split("<")[0]
		if stream_id not in sta:
			sta[stream_id] = {"cmp": set(), "mem": set()}
		if "memcpy" in name or "memset" in name:
			sta[stream_id]["mem"].add(name)
		else:
			sta[stream_id]["cmp"].add(name)
		i += 1
	for k, v in sta.items():
		logger.info("Stream ID: %-2d => cmp: %-10d : mem %-10d %s" % (k, len(v["cmp"]), len(v["mem"]), '' if len(v["mem"]) <= 2 else str(v["mem"])))
	#! Used for debug
	sta1 = sta2 = None
	if platform == 'pytorch':
		sta1 = sta[7]
		sta2 = sta[21]
	elif platform == "tensorflow":
		sta1 = sta[182]
		sta2 = sta[214]
	if sta1 is not None and sta2 is not None:
		logger.info("platform: %s" % (platform))
		logger.info("   intersection: ")
		printIter(sta1["cmp"].intersection(sta2["cmp"]), prefix="\t ")
		logger.info("   minor set: ")
		printIter(sta2["cmp"], prefix="\t ")
		logger.info("   major set: ")
		printIter(sta1["cmp"], prefix="\t ")

if __name__ == "__main__":
	if args.option == "gpu_trace":
		cur_dir = os.path.abspath(args.path)
		root, dirs, files = list(os.walk(cur_dir, topdown=True))[0]
		for file in files:
			#! file name must follow the following format <date>_<id>_<platform>_<others>.txt
			#! e.g., 20191217_04_pytorch_mnist.txt, and must in lowercase.
			if "txt" in file and "log" not in file:
				#! Get the platform name, e.g. mxnet, tensorflow or pytorch
				platform = file.split("_")[2]
				cur_path = os.path.join(root, file)
				logger.info(cur_path)
				handle(cur_path, platform)
	else:
		raise NotImplementedError()

