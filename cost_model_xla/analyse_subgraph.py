import os
import re
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Compile and execute subgraphs in XLA.")

parser.add_argument("graph_path", type=str, 
					help="Path to the graph folder.")

args_ = parser.parse_args()

with open(args_.graph_path, "r") as f:
    all_op_types = [raw_str.split("\"")[1] for raw_str in re.findall("op: \".*\"", f.read())]

stat_counter = defaultdict(int)

for op_type in all_op_types:
    stat_counter[op_type] += 1

stat_list = []

for key, val in stat_counter.items():
    stat_list.append((val, key))

stat_list = sorted(stat_list, reverse=True)
for val, key in stat_list:
    print("{}: {}".format(key, val))

