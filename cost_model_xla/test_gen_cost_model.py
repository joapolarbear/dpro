from .cost_model import *

XlaDataset.construct_dataset("/root/capture_file_tf/run_0/traces_0/0", "/root/test_dataset", "/root/mixbench/build/mixbench-cuda-alt",num_samples=1, min_subgraph_level=3, max_subgraph_level=15)