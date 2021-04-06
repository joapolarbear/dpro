import argparse
from cost_model._xla.xla_module_cost_model import XLAModuleCostModel

parser = argparse.ArgumentParser(description="Script to train the XLA module cost model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset_dir", type=str, required=True,
					help="Path to the directory containing generated dataset files.")
parser.add_argument("--output_dir", type=str, required=True, 
                    help="Directory where the generated cost model files will be dumped to.")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Directory where the generated cost model files will be dumped to.")

args = parser.parse_args()

XLAModuleCostModel.train_on_dataset(args.dataset_dir, args.output_dir, args.batch_size)
