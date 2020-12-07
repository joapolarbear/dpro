import argparse

from cost_model_xla import XlaKernelDataset

parser = argparse.ArgumentParser(description="Script to launch the kernel dataset generator.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--trace_dir", type=str, required=True,
					help="Path to the directory containing trace files for a GPU.")
parser.add_argument("--output_dir", type=str, required=True, 
                    help="Directory where the generated dataset files will be dumped to.")
parser.add_argument("--num_samples", type=int, required=True,
                    help="Number of random samples to generate.")
parser.add_argument("--max_cluster_samples", type=int, default=0,
                    help="Number of max cluster samples to generate.")
parser.add_argument("--min_cluster_size", type=int, default=4,
                    help="Minimum subgraph size.")
parser.add_argument("--max_cluster_size", type=int, default=800,
                    help="Maximum subgraph size.")

args = parser.parse_args()

print("""Using configuation: 
\t Trace Dir: {}\n\t Output Dir: {}\n\t # Random Samples: {}
\t # Max Cluster Samples: {}\n\t Min Cluster Size: {}\n\t Max Cluster Size: {}""".format(
args.trace_dir, args.output_dir, args.num_samples, 
args.max_cluster_samples, args.min_cluster_size, args.max_cluster_size
))

while True:
    choice = input("Continue? [Y/n] ")
    choice = choice.lower()
    if choice in ["y", "n", ""]:
        if choice == "y" or choice == "":
            break
        else:
            print("Aborted.")
            exit(0)
    else:
        print("Please enter y or n only.")

XlaKernelDataset.construct_kernel_dataset(args.trace_dir , args.output_dir, 
                                            num_samples=args.num_samples, 
                                            num_max_cluster_sample=args.max_cluster_samples, 
                                            min_subgraph_level=args.min_cluster_size, 
                                            max_subgraph_level=args.max_cluster_size)