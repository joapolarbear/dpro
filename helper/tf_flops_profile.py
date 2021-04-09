
''' 
    Refer to: https://gist.github.com/shinseung428/752f284d1c065870d7f5a7e4208f0583
'''
import json
import os, sys
import tensorflow as tf
from google.protobuf.json_format import Parse as ParseJSON
from google.protobuf.text_format import Parse as ParseText
from google.protobuf.json_format import MessageToJson
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef

try:
    import horovod.tensorflow as hvd
except:
    pass


def profile_flops(graph_def_path, tmp_path):
    with open(graph_def_path, "r") as f:
        if graph_def_path.endswith("pbtxt"):
            pb = f.read()
            graph_def = ParseText(pb, GraphDef())
            json_string = MessageToJson(graph_def)
            graph_def_as_json = json.loads(json_string)
        else:
            graph_def_as_json = json.load(f)
        cleaned_graph_def_str = json.dumps(graph_def_as_json)
        graph_def = ParseJSON(cleaned_graph_def_str, GraphDef())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        with graph.as_default():
            opt = (tf.profiler.ProfileOptionBuilder(
                tf.profiler.ProfileOptionBuilder.float_operation())
                .with_file_output(tmp_path)
                .build())
            flops = tf.profiler.profile(graph, options=opt)
            total_flops = flops.total_float_ops
            print ("========================================================")
            print ('Total Flops : {}'.format(total_flops))

            # opt = tf.profiler.ProfileOptionBuilder.time_and_memory()
            # rst = tf.profiler.profile(graph, options=opt)
            # print(type(rst))

def parse_flops_dict(graph_def_path, tmp_path):
    profile_flops(graph_def_path, tmp_path)
    op_name2flops = {}
    with open(tmp_path, 'r') as fp:
        lines = fp.read().split("Profile:\n")[1].split("\n")[2:]
    for line in lines:
        line_split = line.split(" ")
        if len(line_split) < 5:
            continue
        # print(line_split)
        op_name = line_split[2]
        flops_str = line_split[3].split("/")[1]
        if flops_str[-1] == "k":
            flops = float(flops_str[:-1]) * 1e3
        elif flops_str[-1] == "m":
            flops = float(flops_str[:-1]) * 1e6
        elif flops_str[-1] == "b":
            flops = float(flops_str[:-1]) * 1e9
        elif flops_str[-1] == "p":
            flops = float(flops_str[:-1]) * 1e12
        else:
            flops = float(flops_str)
        op_name2flops[op_name] = flops
    return op_name2flops

if __name__ == "__main__":
    graph_def_path = sys.argv[1]
    parse_flops_dict(graph_def_path, "flops_log.txt")
