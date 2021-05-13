import argparse
from base import Singleton

parser = argparse.ArgumentParser(description="dPRO Arguments",
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
group_clct = parser.add_argument_group('Trace Collection')
group_clct.add_argument("--comm_backend", type=str, default="NCCL", choices=["NCCL", "BYTEPS"], help="Communication backend")
group_clct.add_argument("--platform", type=str, default="TENSORFLOW", choices=["TENSORFLOW", "MXNET"], help="Platform used to run the model")
group_clct.add_argument("--pcap_file_path", type=str, default=None, help="Path to the directory containing BytePS communication pcap files.")
group_clct.add_argument("--zmq_log_path", type=str, default=None, help="Path to the directory containing BytePS communication zmq log files.")
group_clct.add_argument("--server_log_path", type=str, default=None, help="Path to the directory containing BytePS server log files.")
group_clct.add_argument("--profile_start_step", type=int, default=None, help="The start step of computation profiling. Used for truncating BytePS comm trace.")
group_clct.add_argument("--profile_duration", type=int, default=None, help="The duration (in steps) of computation profiling. Used for truncating BytePS comm trace.")
group_clct.add_argument("--nccl_algo", type=str, default=None, help="NCCL algorithm, Tree or Ring")
group_clct.add_argument("--van_type", type=str, choices=["ZMQ", "RDMA"], default=None, help="Type of protocol used in BytePS.")
group_clct.add_argument("--trace_level", type=str, choices=["debug", "info"], default="info", help="if set to debug, show some trival traces")
group_clct.add_argument("--disable_revise", action="store_true", help="By default, revise traces according to SEND-RECV dependency, set to disable this argument to disable")
group_clct.add_argument("--force", action="store_true", help="Force to re-generate traces, graphs")
group_clct.add_argument("--metadata_path", type=str, default=None,
                    help="Paths to Model metadata")

### replay
group_replay = parser.add_argument_group('Replayer')
group_replay.add_argument("--update_barrier", action="store_true", default=False, help="If true, add a barrier before all UPDATE ops.")
group_replay.add_argument("--update_infi_para", action="store_true", help="If true, UPDATE nodes will be replayed in parallel.")
group_replay.add_argument("--update_clip_overlapping", action="store_true", help="If true, clip overlapping UPDATE nodes in the timeline.")
group_replay.add_argument("--step_num", type=int, default="1", help="Default step numbers to replay.")
group_replay.add_argument("--delay_ratio", type=float, default=1.1, help="delay ratio")
group_replay.add_argument("--full_trace", action="store_true", help="If this arg is set, simulate traces with detailed dependency info.")
group_replay.add_argument("--show_queue", action="store_true", help="If this arg is set, record the queue status of each device during replaying.")

### Optimize
group_opt = parser.add_argument_group('Optimal Strategies Search')
group_opt.add_argument("--optimizer", type=str, default="MCMC", choices=["MCTS", "MCMC"], help="The algorithm used to search the optimal optimzation strategy")
group_opt.add_argument("--ucb_type", type=str, default="AVG", choices=["MAX", "AVG"], help="The type of quanlity value used in the UCB euqation")
group_opt.add_argument("--no_mutation", action="store_true", help="If this arg is set, the default policy of MCTS will not rollout")
group_opt.add_argument("--ucb_gamma", type=float, default=0.1, help="Hyper Parameter used in UCB to control the exploration rate.")
group_opt.add_argument("--ucb_visual", action="store_true", help="If this arg is set, visualize the MCTS search process")
group_opt.add_argument("--no_crit", action="store_true", help="If this arg is set, relax the critical path constaint")

group_opt.add_argument("--mcmc_beta", type=float, default=10, help="Hyper Parameter used in MCMC/SA to control the exploration rate")
group_opt.add_argument("--step_size", type=int, default=1, help="Step size used in MCMC optimizer.")

group_opt.add_argument("--heat_window_size", type=int, default=5, help="Window size for the heat based search heuristic.")
group_opt.add_argument("--relabel", action="store_true", help="If this arg is set, relabel the dag with indexes.")
group_opt.add_argument("--ckpt", action="store_true", help="If this arg is set, start search from checkpoint")
group_opt.add_argument("--workspace", type=str, default=None, help="Workerspace of the optimizer")
group_opt.add_argument("--memory_budget", type=float, default=16, help="GPU Memory budget")

### Operator fusion
group_xla = parser.add_argument_group('Operator Fusion')
group_xla.add_argument("--simulate", action="store_true", help="If this arg is set, simulate the XLA cost model,"
						" but still use its rule to determine which operators to fuse.")
group_xla.add_argument("--xla_candidate_path", type=str, default=None, help="XLA candidate path")
group_xla.add_argument("--layer_num_limit", type=str, default=None, help="Sample some operator fusion strategies, "
                       "where BW operators are fused layer by layer."
                       "This argument specifies the maximum number of layers that can be fused."
                       "Test multiple values by separating them with commas")
group_xla.add_argument("--layer_by_layer", action="store_true", help="Fuse operators layer by layer, if set ture")
group_xla.add_argument("--fusion_once", action="store_true",
                       help="If set, one op can be fused only once")

args = parser.parse_args()


@Singleton
class SingleArg:
	def __init__(self):
		self.args = args
