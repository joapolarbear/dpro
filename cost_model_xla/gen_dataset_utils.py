import shutil
import re
import os
import subprocess
import json
from pathlib import Path
import tensorflow as tf
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef
try:
    NodeDef = tf.NodeDef
except:
    NodeDef = tf.compat.v1.NodeDef
try:
    convert_variables_to_constants = tf.graph_util.convert_variables_to_constants
except:
    convert_variables_to_constants = tf.compat.v1.graph_util.convert_variables_to_constants
try:
    Session = tf.Session
except:
    Session = tf.compat.v1.Session
try:
    import_graph_def = tf.graph_util.import_graph_def
except:
    import_graph_def = tf.compat.v1.graph_util.import_graph_def
from google.protobuf.json_format import Parse
import byteps.tensorflow as bps
from execute_graph import *
from gen_samples import *
from process_trace import *
from xlatools import *
from tqdm import trange

def clean_up(profile_dir, xla_dir):
    shutil.rmtree(profile_dir)
    shutil.rmtree(xla_dir)
    os.makedirs(profile_dir)
    os.makedirs(xla_dir)

def clean_up_dir(dir_path):
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def clean_up_profile(profile_dir):
    shutil.rmtree(profile_dir)
    os.makedirs(profile_dir)

def log_op_names(graph_def, log_file="./op_names_in_def.txt"):
    with open(log_file, "w") as f:
        for node_def in graph_def.node:
            f.write(node_def.name)
            f.write("\n")

def profile_entire_graph(sess, sample_generator, profile_dir, num_runs_per_sample=20):
    graph_def, input_defs = sample_generator.get_original_graph_def()
    # log_op_names(graph_def)
    execute_graph_def(graph_def, input_defs, profile_dir, num_runs=num_runs_per_sample)
    # get trace
    trace_path = search_for_trace(profile_dir)
    if trace_path is None:
        clean_up_profile(profile_dir)
        return
    # parse trace
    op_time_dict = get_execution_time_for_whole_graph(trace_path)
    ops_to_remove = []
    for op_name, (time, count) in op_time_dict.items():
        if count != num_runs_per_sample:
            ops_to_remove.append(op_name)
    for op_name in ops_to_remove:
        op_time_dict.pop(op_name)
    clean_up_profile(profile_dir)
    return op_time_dict

def run_gpu_profile(profiler_exec):
    process = subprocess.run([profiler_exec], check=True, capture_output=True)
    output = process.stdout.decode("ascii")
    gflops = float(re.findall("Compute throughput: .* GFlops", output)[0].split()[2])
    gbps = float(re.findall("Memory bandwidth: .* GB/sec", output)[0].split()[2])
    return gflops, gbps

def get_time_as_sum_of_indivisual(op_time_dict, graph_def):
    total_time = 0
    for node_def in graph_def.node:
        if node_def.name in op_time_dict:
            time, _ = op_time_dict[node_def.name]
            total_time += time
    return total_time

def gen_sample_once(sample_id, sample_generator, dataset_dir, label_file_path, feature_dir, dataset_hlo_dir, 
                    profile_dir, xla_dir, gen_feature_exec, op_time_dict, gpu_gflops, gpu_gbps, debug=False, min_levels=1, max_levels=8, 
                    num_runs_per_sample=20):
    # generate one sample
    subgraph_def, input_defs, output_def, _ = sample_generator.gen_random_subgraph(min_levels=min_levels, max_levels=max_levels, debug_dir="./debug" if debug else None)
    # execute the graph
    execute_graph_def(subgraph_def, input_defs, profile_dir, num_runs=num_runs_per_sample)
    # get trace
    trace_path = search_for_trace(profile_dir)
    if trace_path is None:
        clean_up(profile_dir, xla_dir)
        return
    # parse_trace
    avg_time, count = get_execution_time_from_trace(trace_path)
    if count != num_runs_per_sample:
        print("[WARINNG] Expected {} items in trace but found {}.".format(num_runs_per_sample, count))
        clean_up(profile_dir, xla_dir)
        return
    # get hlo
    hlo_path = search_for_hlo(xla_dir)
    if hlo_path is None:
        clean_up(profile_dir, xla_dir)
        return
    # generate feature vector
    feature_path = os.path.join(feature_dir, "{}.txt".format(str(sample_id)))
    subprocess.run([gen_feature_exec, hlo_path, feature_path, str(gpu_gflops), str(gpu_gbps)], check=True)
    # inject sum of running time into feature vector
    op_time_as_sum = get_time_as_sum_of_indivisual(op_time_dict, subgraph_def)
    print("[INFO] sum of running time is {}.".format(op_time_as_sum))
    with open(feature_path, "r") as f:
        content = f.read()
    content = str(op_time_as_sum) + "\n" + content
    with open(feature_path, "w") as f:
        f.write(content)
    # write running time to file
    with open(label_file_path, "a") as f:
        f.write("{}: {}\n".format(sample_id, avg_time))
    # copy and rename hlo file
    shutil.copyfile(hlo_path, os.path.join(dataset_hlo_dir, "hlo_{}.txt".format(sample_id)))
    clean_up(profile_dir, xla_dir)
    return

def gen_sample_once_using_replay(sample_generator, dataset_dir, label_file_path, feature_dir, dataset_hlo_dir, 
                    profile_dir, op_time_dict, gpu_gflops, gpu_gbps, min_levels=1, max_levels=8, debug_dir = None):
    # generate one sample
    raw_subgraph_dir = os.path.join(dataset_dir, "generated_subgraph")
    subgraph_def, input_defs, output_def, sample_id = sample_generator.gen_random_subgraph(choose_root_from_ops=list(op_time_dict.keys()), min_levels=min_levels, max_levels=max_levels, debug_dir=raw_subgraph_dir)
    # replay hlo
    def_path = os.path.join(raw_subgraph_dir, "{}.pbtxt".format(sample_id))
    config_path = os.path.join(raw_subgraph_dir, "{}_config.pbtxt".format(sample_id))
    unopt_path = os.path.join(profile_dir, "{}_unopt_hlo.txt".format(sample_id))
    opt_path = os.path.join(profile_dir, "{}_opt_hlo.txt".format(sample_id))
    try:
        compile_to_hlo(def_path, config_path, unopt_path, opt_path)
        avg_time = replay_hlo(unopt_path) * 1e6
    except:
        if debug_dir:
            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)
            def_name = Path(def_path).name
            shutil.copyfile(def_path, os.path.join(debug_dir, def_name))
            config_name = Path(config_path).name
            shutil.copyfile(config_path, os.path.join(debug_dir, config_name))
        print("[WARNING] Failed to compile & replay HLO code.")
        clean_up_dir(profile_dir)
        clean_up_dir(raw_subgraph_dir)
        return
    # generate feature vector
    feature_path = os.path.join(feature_dir, "{}.txt".format(str(sample_id)))
    gen_feature_vector(opt_path, feature_path, gpu_gflops, gpu_gbps)
    # inject sum of running time into feature vector
    op_time_as_sum = get_time_as_sum_of_indivisual(op_time_dict, subgraph_def)
    print("[INFO] sum of running time is {}.".format(op_time_as_sum))
    try:
        with open(feature_path, "r") as f:
            content = f.read()
        content = str(op_time_as_sum) + "\n" + content
        with open(feature_path, "w") as f:
            f.write(content)
        # write running time to file
        with open(label_file_path, "a") as f:
            f.write("{}: {}\n".format(sample_id, avg_time))
    except:
        print("[WARNING] Cannot find generated feature vector of sample {}".format(sample_id))
    # copy and rename hlo file
    shutil.copyfile(opt_path, os.path.join(dataset_hlo_dir, "hlo_{}.txt".format(sample_id)))
    clean_up_dir(profile_dir)
    clean_up_dir(raw_subgraph_dir)
    return

# def gen_dataset(graph_def, op_time_dict, gpu_benchmark_cmd, result_dir, num_samples=2000, 
#                 min_subgraph_level=5, max_subgraph_level=15):
#     if not os.path.isdir(result_dir):
#         os.makedirs(result_dir)
#         print("Result dir not exist. Created result dir at {}.".format(result_dir))
#     dataset_dir = os.path.join(result_dir, "dataset")
#     label_file_path = os.path.join(result_dir, "running_time.txt")
#     if os.path.isfile(label_file_path):
#         raise RuntimeError("Label file already exists.")
#     feature_dir = os.path.join(dataset_dir, "feature_vecs")
#     if not os.path.isdir(feature_dir):
#         os.makedirs(feature_dir)
#     dataset_hlo_dir = os.path.join(dataset_dir, "hlos")
#     if not os.path.isdir(dataset_hlo_dir):
#         os.makedirs(dataset_hlo_dir)
#     profile_dir = os.path.join(result_dir, "xla_profile")
#     if not os.path.isdir(profile_dir):
#         os.makedirs(profile_dir)
#     if isinstance(graph_def, str):
#         sample_generator = SampleGenerator(freezed_graph_path=graph_def)
#     else:
#         sample_generator = SampleGenerator(freezed_graph_def=graph_def)
#     print("Benchmarking GPU stats.")
#     gflops, gbps = run_gpu_profile(gpu_benchmark_cmd)
#     print("Start generation.")
#     for i in trange(num_samples):
#         gen_sample_once_using_replay(sample_generator, dataset_dir, label_file_path, feature_dir,
#                         dataset_hlo_dir, profile_dir, op_time_dict, gflops, gbps,
#                         min_levels=min_subgraph_level, max_levels=max_subgraph_level)
#     if os.path.isdir(profile_dir):
#         os.rmdir(profile_dir)
#     print("Dataset generation complete.")
def gen_np_tensor(shape, dtype):
    res = np.random.rand(*shape)
    if isinstance(res, float):
        if "int" in dtype:
            return int(res)
        else:
            return res
    else:
        return res.astype(dtype)


# Restore the variable values
def restore_var_values(sess, var_shapes):
    # Find the variable initialization operations
    assign_ops = []
    feed_dict = {}
    for v, meta in var_shapes.items():
        try:
            assign_op = sess.graph.get_operation_by_name(v + '/Assign')
        except:
            assign_op = sess.graph.get_operation_by_name("import/" + v + '/Assign')
        assign_ops.append(assign_op)
        shape = meta["shape"]
        dtype = meta["dtype"]
        feed_dict[assign_op.inputs[1]] = gen_np_tensor(shape, dtype)
    # Run the initialization operations with the given variable values
    sess.run(assign_ops, feed_dict=feed_dict)

def shape_as_list_to_pb_json(shape):
    shape_dict = {"shape": {"dim":[]}}
    for dim in shape:
        shape_dict["shape"]["dim"].append({"size": str(dim)})
    return shape_dict

def gen_dataset(trace_dir, op_time_dict, gpu_benchmark_cmd, result_dir, num_samples=2000, 
                min_subgraph_level=5, max_subgraph_level=15):
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        print("Result dir not exist. Created result dir at {}.".format(result_dir))
    dataset_dir = os.path.join(result_dir, "dataset")
    debug_dir = os.path.join(result_dir, "debug")
    label_file_path = os.path.join(result_dir, "running_time.txt")
    if os.path.isfile(label_file_path):
        raise RuntimeError("Label file already exists.")
    feature_dir = os.path.join(dataset_dir, "feature_vecs")
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    dataset_hlo_dir = os.path.join(dataset_dir, "hlos")
    if not os.path.isdir(dataset_hlo_dir):
        os.makedirs(dataset_hlo_dir)
    profile_dir = os.path.join(result_dir, "xla_profile")
    if not os.path.isdir(profile_dir):
        os.makedirs(profile_dir)
    ## load graphdef
    # load shape dict
    with open(os.path.join(trace_dir, "tensor_shapes.json"), "r") as f:
        shape_dict = json.load(f)
    with open(os.path.join(trace_dir, "graph.json"), "r") as f:
        graph_def_as_json = json.load(f)
        for node in graph_def_as_json["node"]:
            if "input" in node:
                cleaned_inputs = []
                for input_tensor in node["input"]:
                    if not input_tensor.startswith("^"):
                        cleaned_inputs.append(input_tensor)
                if cleaned_inputs:
                    node["input"] = cleaned_inputs
                else:
                    del node["input"]
            if "attr" in node:
                if "_class" in node["attr"]:
                    del node["attr"]["_class"]
                if "container" in node["attr"]:
                    del node["attr"]["container"]
                if "shared_name" in node["attr"]:
                    del node["attr"]["shared_name"]
                if "dtype" in node["attr"]:
                    # TODO: remove _ref dtypes
                    pass
                if "shape" in node["attr"]:
                    # print("====================")
                    # print(node["attr"]["shape"])
                    # print(shape_dict[node["name"]+":0"])
                    # print(shape_as_list_to_pb_json(shape_dict[node["name"]+":0"]))
                    # exit(0)
                    node["attr"]["shape"] = shape_as_list_to_pb_json(shape_dict[node["name"]+":0"])
        cleaned_graph_def_str = json.dumps(graph_def_as_json)
        graph_def = Parse(cleaned_graph_def_str, GraphDef())
    # sess = Session()
    # import_graph_def(graph_def)
    # with open(os.path.join(trace_dir, "variables_meta.json"), "r") as f:
    #     var_shapes = json.load(f)
    # with open(os.path.join(trace_dir, "run_meta.json"), "r") as f:
    #     run_meta = json.load(f)
    # feed_meta = run_meta["feed_dict"]
    # fetches = run_meta["fetches"]
    # restore_var_values(sess, var_shapes)

    # run_fetches = []
    # for tn in fetches:
    #     try:
    #         try:
    #             fetch = sess.graph.get_tensor_by_name(tn).name
    #         except:
    #             fetch = sess.graph.get_tensor_by_name("import/" + tn).name
    #     except:
    #         try:
    #             fetch = sess.graph.get_operation_by_name(tn).name
    #         except:
    #             fetch = sess.graph.get_operation_by_name("import/" + tn).name
    # run_fetches.append(fetch)

    # clean up graphdef
    save_nodes = []
    for node in graph_def.node:
        if "save" in node.name:
            save_nodes.append(node.name)
    cleaned_graph_def = GraphDef()
    cleaned_graph_def.versions.CopyFrom(graph_def.versions)
    cleaned_graph_def.library.CopyFrom(graph_def.library)
    for node in graph_def.node:
        if "save" in node.name:
            continue
        node_def = NodeDef()

        cleaned_graph_def.node.append(node)

    sample_generator = SampleGenerator(graph_def=cleaned_graph_def, shape_dict=shape_dict)
    print("Benchmarking GPU stats.")
    gflops, gbps = run_gpu_profile(gpu_benchmark_cmd)
    print("Start generation.")
    for i in trange(num_samples):
        gen_sample_once_using_replay(sample_generator, dataset_dir, label_file_path, feature_dir,
                        dataset_hlo_dir, profile_dir, op_time_dict, gflops, gbps,
                        min_levels=min_subgraph_level, max_levels=max_subgraph_level, debug_dir=debug_dir)
    if os.path.isdir(profile_dir):
        os.rmdir(profile_dir)
    print("Dataset generation complete.")