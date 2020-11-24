import argparse
from base import Singleton

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, 
					choices=["statistic", "graph", "combine", "mapping", "compare", "critical", "timeline", "replay", "topo_sort", "collect", "3dcompare", "optimize"],
					help="The type of analysis to process. including:\n" + 
						"* statistic: show the statistic results\n" + 
						"* graph: show the dependency graph\n")
parser.add_argument("--sub_option", type=str, default=None, help="Sub options for each option")
parser.add_argument("--path", type=str, required=True, help="The paths of traces you want to analyze, support multiple paths seperated with comma.")

parser.add_argument("--sort", action="store_true", help="Sorted in descending order")
parser.add_argument("--head", type=int, default=None, help="Print the first few lines")
parser.add_argument("--xlsx", action="store_true", help="Output XLSX file of the statistic results")
parser.add_argument("--del_queue", action="store_true", help="If set True, delete the queue time in communication traces. ")
parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level")
parser.add_argument("--clean", action="store_true", help="Flush the log file")

parser.add_argument("--pretty", action="store_true", help="Output necessary info if set")
parser.add_argument("--filter", type=str, default=None, help="Used to show part of communication operations, seperated with comma.")
parser.add_argument("--progress", action="store_true", help="Show the progress bar if it is set, disable the std output")
parser.add_argument("--debug_traces", action="store_true", help="If set, output traces profiled for the analysis process")

### collect
parser.add_argument("--comm_backend", type=str, default="NCCL", choices=["NCCL", "BYTEPS"], help="Communication backend")
parser.add_argument("--platform", type=str, default="TENSORFLOW", choices=["TENSORFLOW", "MXNET"], help="Platform used to run the model")
parser.add_argument("--pcap_file_path", type=str, default=None, help="Path to the directory containing BytePS communication pcap files.")
parser.add_argument("--server_log_path", type=str, default=None, help="Path to the directory containing BytePS server log files.")
parser.add_argument("--profile_start_step", type=int, default=None, help="The start step of computation profiling. Used for truncating BytePS comm trace.")
parser.add_argument("--profile_duration", type=int, default=None, help="The duration (in steps) of computation profiling. Used for truncating BytePS comm trace.")
parser.add_argument("--nccl_algo", type=str, default=None, help="NCCL algorithm, Tree or Ring")
parser.add_argument("--trace_level", type=str, choices=["debug", "info"], default="info", help="if set to debug, show some trival traces")
parser.add_argument("--disable_revise", action="store_true", help="By default, revise traecs according to SEND-RECV dependency, set to disable this argument to disable")
parser.add_argument("--force", action="store_true", help="Force to re-generate traces, graphs")

### replay
parser.add_argument("--update_barrier", type=bool, default=False, help="If true, add a barrier before all UPDATE ops.")
parser.add_argument("--step_num", type=int, default="1", help="Default step numbers to replay.")
parser.add_argument("--delay_ratio", type=float, default=1.1, help="delay ratio")
parser.add_argument("--full_trace", action="store_true", help="If this arg is set, simulate traces with detailed dependency info.")
parser.add_argument("--show_queue", action="store_true", help="If this arg is set, record the queue status of each device during replaying.")

### Optimize
parser.add_argument("--optimizer", type=str, default="MCTS", choices=["MCTS", "MCMC"], help="The algorithm used to search the optimal optimzation strategy")
parser.add_argument("--ucb_type", type=str, default="AVG", choices=["MAX", "AVG"], help="The type of quanlity value used in the UCB euqation")
parser.add_argument("--no_mutation", action="store_true", help="If this arg is set, the default policy of MCTS will not rollout")
parser.add_argument("--ucb_gamma", type=float, default=0.1, help="Hyper Parameter used in UCB to control the exploration rate.")
parser.add_argument("--ucb_visual", action="store_true", help="If this arg is set, visualize the MCTS search process")
parser.add_argument("--mcmc_beta", type=float, default=100, help="Hyper Parameter used in MCMC/SA to control the exploration rate")
parser.add_argument("--cost_model_tmp_dir", type=str, default="./", help="Tmp directory for cost model to store intermediate files.")
parser.add_argument("--heat_window_size", type=int, default=5, help="Window size for the heat based search heuristic.")

args = parser.parse_args()


@Singleton
class SingleArg:
	def __init__(self):
		self.args = args