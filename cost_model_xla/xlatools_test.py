import os
from xlatools import *

compile_to_hlo("/root/bert_subgraph/0.pbtxt", "/root/bert_subgraph/0_config.pbtxt", "/root/bert_subgraph/0_unopt.txt", "/root/bert_subgraph/0_opt.txt")
assert os.path.isfile("/root/bert_subgraph/0_unopt.txt")
assert os.path.isfile("/root/bert_subgraph/0_opt.txt")

print("Compile to HLO successful.")

gen_feature_vector("/root/bert_subgraph/0_unopt.txt", "/root/bert_subgraph/feature.txt", 1.2, 5.3)
assert os.path.isfile("/root/bert_subgraph/feature.txt")

print("gen_feature_vector successful.")

time = replay_hlo("/root/bert_subgraph/0_unopt.txt")
print("Average replay time: {}".format(time * 1e6))
