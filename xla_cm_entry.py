import os
import argparse
import logger_utils
from cost_model._xla.xla_module_cost_model import XLAModuleCostModel
from cost_model._xla.gen_dataset_utils import XlaKernelDataset

try:
    import byteps.tensorflow as bps
except:
    pass

try:
    import horovod.tensorflow as hvd
except:
    pass

parser = argparse.ArgumentParser(description="Script to launch the kernel dataset generator and train the XLA module cost model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", type=int, default=0,
					help="Different actions with different mode:\n"
                    "   0: generate training data and train the cost model\n"
                    "   1: only generate training data\n"
                    "   2: only traing the cost model\n"
                    "   3: only test the cost model")

parser.add_argument("--trace_dir", type=str, help="Path to the directory containing trace files for a GPU.")
parser.add_argument("--output_dir", type=str, help="Directory where the generated dataset files will be dumped to.")
parser.add_argument("--num_samples", type=int, help="Number of random samples to generate.")
parser.add_argument("--max_cluster_samples", type=int, default=0, help="Number of max cluster samples to generate.")
parser.add_argument("--min_cluster_size", type=int, default=4, help="Minimum subgraph size.")
parser.add_argument("--max_cluster_size", type=int, default=800, help="Maximum subgraph size.")

parser.add_argument("--batch_size", type=int, default=256,
                    help="Directory where the generated cost model files will be dumped to.")

parser.add_argument("--dataset_dir", type=str,
					help="Path to the directory containing generated dataset files.")

args = parser.parse_args()

logger = logger_utils.SingleLogger(args.output_dir, "xla_cm", "INFO")
logger.info(args)

if args.mode == 0 or args.mode == 1:
    logger_utils.SingleLogger().info("Generate Kernel dataset ...")
    print("""Using configuation: 
    \t Trace Dir: {}\n\t Output Dir: {}\n\t # Random Samples: {}
    \t # Max Cluster Samples: {}\n\t Min Cluster Size: {}\n\t Max Cluster Size: {}""".format(
    args.trace_dir, args.output_dir, args.num_samples, 
    args.max_cluster_samples, args.min_cluster_size, args.max_cluster_size
    ))

    ### Generate Kernel dataset
    XlaKernelDataset.construct_kernel_dataset(args.trace_dir , os.path.join(args.output_dir, "kernel_dataset"), 
                                                num_samples=args.num_samples, 
                                                num_max_cluster_samples=args.max_cluster_samples, 
                                                min_subgraph_level=args.min_cluster_size, 
                                                max_subgraph_level=args.max_cluster_size)

if args.mode == 0 or args.mode == 2:
    logger_utils.SingleLogger().info("Train the cost model ...")
    ### Train the cost model
    assert os.path.exists(os.path.join(args.output_dir, "kernel_dataset"))
    XLAModuleCostModel.train_on_dataset(
        os.path.join(args.output_dir, "kernel_dataset"), 
        os.path.join(args.output_dir, "cost_model"), 
        args.batch_size)

if args.mode == 3:
    logger_utils.SingleLogger().info("Test the cost model ...")
    module_cost_model = XLAModuleCostModel(os.path.join(args.output_dir, "cost_model"))
    module_cost_model.test_on_dataset(args.dataset_dir)


