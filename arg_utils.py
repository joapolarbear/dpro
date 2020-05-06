import argparse
from base import Singleton

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, 
					choices=["statistic", "graph", "combine", "compare", "critical", "timeline", "replay", "topo_sort", "collect", "3dcompare"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
parser.add_argument("--sub_option", type=str, default=None, help="Sub options for each option")
parser.add_argument("--path", type=str, required=True, help="The paths of traces you want to analyze, support multiple paths seperated with comma.")

parser.add_argument("--sort", action="store_true", help="Sorted in descending order")
parser.add_argument("--head", type=int, default=None, help="Print the first few lines")
parser.add_argument("--xlsx", action="store_true", help="Output XLSX file of the statistic results")
parser.add_argument("--del_queue", action="store_true", help="If set True, delete the queue time in communication traces. ")
parser.add_argument("--logging_level", type=int, default="20", help="Logging level")
parser.add_argument("--clean", action="store_true", help="Flush the log file")

parser.add_argument("--pretty", action="store_true", help="Output necessary info if set")
parser.add_argument("--filter", type=str, default=None, help="Used to show part of communication operations, seperated with comma.")
parser.add_argument("--progress", action="store_true", help="Show the progress bar if it is set, disable the std output")
parser.add_argument("--debug_traces", action="store_true", help="If set, output traces profiled for the analysis process")

### collect
parser.add_argument("--nccl_algo", type=str, default=None, help="NCCL algorithm")
parser.add_argument("--trace_level", type=str, choices=["debug", "info"], default="info", help="if set to debug, show some trival traces")
parser.add_argument("--disable_revise", action="store_true", help="By default, revise traecs according to SEND-RECV dependency, set to disable this argument to disable")

### replay
parser.add_argument("--step_num", type=int, default="1", help="Default step numbers to replay.")
parser.add_argument("--delay_ratio", type=float, default=1.1, help="delay ratio")

args = parser.parse_args()


@Singleton
class SingleArg:
	def __init__(self):
		self.args = args