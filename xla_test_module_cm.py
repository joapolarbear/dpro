import argparse
from cost_model_xla.xla_module_cost_model import XLAModuleCostModel

parser = argparse.ArgumentParser(description="Script to train the XLA module cost model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset_dir", type=str, required=True,
					help="Path to the directory containing generated dataset files.")
parser.add_argument("--cost_model_dir", type=str, required=True, 
                    help="Path to the generated cost model directory.")

args = parser.parse_args()

module_cost_model = XLAModuleCostModel(args.cost_model_dir)
module_cost_model.test_on_dataset(args.dataset_dir)