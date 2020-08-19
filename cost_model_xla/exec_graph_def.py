from cost_model_xla.gen_dataset_utils import *
import tensorflow as tf
from google.protobuf import text_format
import argparse

parser = argparse.ArgumentParser(description="Execute graph def")
parser.add_argument("--graph_path", type=str, required=True, help="Graph def path")
parser.add_argument("--config_path", type=str, required=True, help="Graph config path")
parser.add_argument("--result_dir", type=str, required=True, help="Profile result dir")

args = parser.parse_args()

class Name(object):
    def __init__(self, name):
        super().__init__()
        self.name = name

with tf.compat.v1.gfile.FastGFile(args.graph_path, "r") as f:
    print("Reading graph...")
    graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

with open(args.config_path, "r") as f:
    print("Reading config...")
    config_content = [line.rstrip('\n') for line in f]

feeds = []
fetches = []
index = 0
while index < len(config_content):
    line = config_content[index]
    if "feed" in line:
        next_line = config_content[index + 1]
        assert "id" in next_line
        name = next_line.split("\"")[1]
        feeds.append(Name(name))
        index += 2
    elif "fetch" in line:
        next_line = config_content[index + 1]
        assert "id" in next_line
        name = next_line.split("\"")[1]
        fetches.append(Name(name))
        index += 2
    else:
        index += 1

execute_graph_def(graph_def, feeds, fetches, args.result_dir, num_runs=800, trace_start=780, trace_end=790)