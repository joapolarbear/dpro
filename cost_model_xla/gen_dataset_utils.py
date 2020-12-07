import shutil
import re
import os
import subprocess
import json
import traceback
from pathlib import Path
import copy
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

from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split

from .gen_samples import *
from .process_trace import *
from .xlatools import *

MAX_OP_DUR_IN_US = 10000000

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

"""
    Classes for fusion cost model
"""    
class XlaKernelDataset(object):
    def __init__(self, dataset_path, test_size=0.2):
        super().__init__()
        self._dataset_path = dataset_path
        self.test_size_ = test_size
        if not os.path.isdir(self._dataset_path):
            # not a dir, must be a pickled dataset
            print("Loaded dumped dataset.")
            self._load_dataset(dataset_path)
        else:
            self._validate_dataset()
            self._process_dataset()

    def _validate_dataset(self):
        files = os.listdir(self._dataset_path)
        for fname in ["features", "labels.txt"]:
            if fname not in files:
                raise RuntimeError("Cannot find {} in {}.".format(fname, self._dataset_path))
        self._label_file_path = os.path.join(self._dataset_path, "labels.txt")
        self._feature_dir_path = os.path.join(self._dataset_path, "features")
        if not os.path.isdir(self._feature_dir_path):
            raise NotADirectoryError("{} is not a directory.".format(self._feature_dir_path))
        feature_file_names = os.listdir(self._feature_dir_path)
        if not feature_file_names:
            raise RuntimeError("Cannot find any feature vectors.")
        fv_name_regex = re.compile("^[0-9]*.txt$")
        for feature_file_name in feature_file_names:
            if fv_name_regex.match(feature_file_name) is None:
                raise RuntimeError("Found unexpected feature vector file {} in dataset.".format(feature_file_name))
    
    def _parse_feature_file(self, f):
        first_line = True
        subop_info_dict = {}
        dependency_edges = []
        fusion_type = None
        computation_hash = None
        for line in f:
            if first_line:
                fusion_type, computation_hash = [int(value.strip()) for value in line.split(",")]
                first_line = False
            else:
                splitted_line = [value.strip() for value in line.split(",")]
                op_name = splitted_line[0]
                op_code_in_str = splitted_line[1]
                op_code = int(splitted_line[2])
                num_inputs = int(splitted_line[3])
                input_names = []
                input_shapes = []
                for i in range(num_inputs):
                    input_name = splitted_line[4+2*i]
                    input_shape = splitted_line[4+2*i+1]
                    input_dims = [int(value.strip()) for value in input_shape.split(":")]
                    input_names.append(input_name)
                    input_shapes.append(input_dims)
                    dependency_edges.append((input_name, op_name))
                output_shape = [int(value.strip()) for value in splitted_line[-2].split(":")]
                op_hash = int(splitted_line[-1])
                subop_info_dict[op_name] = (op_name, op_code_in_str, op_code, input_names, input_shapes, output_shape, op_hash)
        # topologically sort subop_infos
        # create dependency dags
        G = nx.DiGraph()
        G.add_edges_from(dependency_edges)
        sorted_node_names = list(nx.topological_sort(G))
        subop_infos = []
        for node_name in sorted_node_names:
            subop_infos.append(subop_info_dict[node_name])
        # check if there is any nodes left
        for node_name in subop_info_dict.keys():
            if node_name not in G:
                sorted_node_names.append(node_name)
                subop_infos.append(subop_info_dict[node_name])
                G.add_node(node_name)
        adj = nx.adjacency_matrix(G.to_undirected(), nodelist=sorted_node_names)
        return (adj, fusion_type, computation_hash, subop_infos)

    def _load_labels_and_features(self):
        # load true labels
        label_dict = {}
        with open(self._label_file_path, "r") as f:
            for line in f:
                splitted = line.split(":")
                sample_id = int(splitted[0].strip())
                time = float(splitted[1].strip())
                if time > MAX_OP_DUR_IN_US:
                    continue
                label_dict[sample_id] = time
        num_all_samples = len(label_dict)
        # load features
        raw_feature_dict = {}
        duplicate_hashes = {}
        feature_file_names = os.listdir(self._feature_dir_path)
        for fn in feature_file_names:
            sample_id = int(fn.split(".txt")[0].strip())
            with open(os.path.join(self._feature_dir_path, fn), "r") as f:
                (adj, fusion_type, computation_hash, subop_infos) = self._parse_feature_file(f)
                if computation_hash not in duplicate_hashes:
                    duplicate_hashes[computation_hash] = []
                duplicate_hashes[computation_hash].append(sample_id)
                raw_feature_dict[sample_id] = (adj, fusion_type, computation_hash, subop_infos)
        # deduplication
        psd = []
        dedupedsid2finalsid = {}
        for comp_hash, sample_ids in duplicate_hashes.items():
            if len(sample_ids) > 1:
                recorded_times = [label_dict[sid] for sid in sample_ids]
                psd.append(100* np.std(recorded_times) / np.average(recorded_times))
                # delete all later sample_ids and replace the label with average
                for sid in sample_ids[1:]:
                    raw_feature_dict.pop(sid)
                    label_dict.pop(sid)
                    dedupedsid2finalsid[sid] = sample_ids[0]
                label_dict[sample_ids[0]] = np.average(recorded_times)

        print("All samples: {}, deduplicated: {}".format(num_all_samples, len(raw_feature_dict)))
        print("Duplication mean RSD: {}%, max RSD: {}%, min RSD: {}%, median RSD: {}%".format(np.average(psd), max(psd), min(psd), np.median(psd)))
        self.label_dict = label_dict
        self.dedupedsid2finalsid = dedupedsid2finalsid
        self.raw_feature_dict = raw_feature_dict

    def _relabel_op_codes_and_hashes(self):
        # relabel opcodes and calculate shape statistics
        max_fusion_type = 0
        opcode2index = {}
        ophash2index = {}
        index2opcode = {}
        index2ophash = {}
        max_input_num = 0
        max_input_dims = []
        max_output_dim = 0
        op_codes = set()
        op_hashes = set()
        for sample_id, (_, fusion_type, _, subop_infos) in self.raw_feature_dict.items():
            if fusion_type + 1> max_fusion_type:
                max_fusion_type = fusion_type + 1
            for _, _, op_code, _, input_shapes, output_shape, op_hash in subop_infos:
                op_codes.add(op_code)
                op_hashes.add(op_hash)
                max_input_num = max(max_input_num, len(input_shapes))
                for index, input_dims in enumerate(input_shapes):
                    if index +1 > len(max_input_dims):
                        max_input_dims.append(len(input_dims))
                    else:
                        max_input_dims[index] = max(max_input_dims[index], len(input_dims))
                max_output_dim = max(max_output_dim, len(output_shape))
        sorted_op_codes = sorted(list(op_codes))
        sorted_op_hashes = sorted(list(op_hashes))
        for index, code in enumerate(sorted_op_codes):
            opcode2index[code] = index
            index2opcode[index] = code
        for index, op_hash in enumerate(sorted_op_hashes):
            ophash2index[op_hash] = index
            index2ophash[index] = op_hash
        
        print("num opcodes: {}, num different op hashes: {}".format(len(sorted_op_codes), len(sorted_op_hashes)))
        print("feature vector dim: {}".format(sum(max_input_dims) + max_output_dim))
        self.max_fusion_type = max_fusion_type
        self.opcode2index = opcode2index
        self.ophash2index = ophash2index
        self.index2opcode = index2opcode
        self.index2ophash = index2ophash
        self.max_input_dims = max_input_dims
        self.max_output_dim = max_output_dim
        self.feature_dim = sum(self.max_input_dims) + self.max_output_dim
    
    def _gen_feature_vector(self, input_shapes, output_shape):
        subop_vector = []
        # inputs
        for index, input_shape in enumerate(input_shapes):
            if index + 1 > len(self.max_input_dims):
                # number of inputs larger than max # inputs of 
                # ops in the training set, may occur in inference
                # in this case we truncate.
                break
            if len(input_shape) > self.max_input_dims[index]:
                # have more dimensions than ever seen, truncate
                input_shape = input_shape.copy()[:self.max_input_dims[index]]
            subop_vector += input_shape
            len_input_paddings = self.max_input_dims[index] - len(input_shape)
            subop_vector += [0] * len_input_paddings
        for index in range(len(input_shapes), len(self.max_input_dims)):
            subop_vector += [0] * self.max_input_dims[index]
        if len(subop_vector) != sum(self.max_input_dims):
            print("len vec: {}, should be: {}".format(len(subop_vector), sum(self.max_input_dims)))
            exit(-1)
        # output
        subop_vector += output_shape
        len_output_paddings = self.max_output_dim - len(output_shape)
        subop_vector += [0] * len_output_paddings
        return subop_vector
    
    def gen_representation_for_sample(self, fusion_type, subop_infos):
        fusion_type_one_hot = [0] * self.max_fusion_type
        fusion_type_one_hot[fusion_type] = 1
        op_codes_in_sample = []
        op_hashes_in_sample = []
        feature_vectors_in_sample = []
        for subop_index, (_, _, op_code, _, input_shapes, output_shape, op_hash) in enumerate(subop_infos):
            op_codes_in_sample.append(self.opcode2index[op_code] + 1)
            if op_hash in self.ophash2index:
                op_hashes_in_sample.append(self.ophash2index[op_hash] + 1)
            else:
                # print("[WARNING] Unseen Op hash {}".format(op_hash))
                op_hashes_in_sample.append(random.randint(1, len(self.index2ophash)))
            feature_vectors_in_sample.append(self._gen_feature_vector(input_shapes, output_shape))
        return fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample

    def _gen_samples_representation(self):
        # transform samples into lists of shape (num_sample, num_subop, vectors)
        fusion_types = []
        op_codes = []
        op_hashes = []
        feature_vectors = []
        labels = []
        for sample_id, (_, fusion_type, _, subop_infos) in self.raw_feature_dict.items():
            fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample = self.gen_representation_for_sample(fusion_type, subop_infos)
            fusion_types.append(fusion_type_one_hot)
            op_codes.append(op_codes_in_sample)
            op_hashes.append(op_hashes_in_sample)
            feature_vectors.append(feature_vectors_in_sample)
            labels.append(self.label_dict[sample_id])

        # split training and test sets
        (fusion_types_train, fusion_types_test,
        op_codes_train, op_codes_test, 
        op_hashes_train, op_hashes_test, 
        feature_vectors_train, feature_vectors_test, 
        labels_train, labels_test) = train_test_split(fusion_types, op_codes, op_hashes, feature_vectors, labels, test_size=self.test_size_)

        self.fusion_types_train = fusion_types_train
        self.op_codes_train = op_codes_train
        self.op_hashes_train = op_hashes_train
        self.feature_vectors_train = feature_vectors_train
        self.labels_train = labels_train

        self.fusion_types_test = fusion_types_test
        self.op_codes_test = op_codes_test
        self.op_hashes_test = op_hashes_test
        self.feature_vectors_test = feature_vectors_test
        self.labels_test = labels_test

    def _process_dataset(self):
        self._load_labels_and_features()
        self._relabel_op_codes_and_hashes()
        self._gen_samples_representation()

    def dump_dataset(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump([self.dedupedsid2finalsid, self.max_fusion_type, self.fusion_types_train, self.fusion_types_test, self.feature_dim,
                        self.op_codes_train, self.op_hashes_train, self.feature_vectors_train, self.labels_train, 
                        self.op_codes_test, self.op_hashes_test, self.feature_vectors_test, self.labels_test, 
                        self.label_dict, self.raw_feature_dict,
                        self.opcode2index, self.ophash2index, self.index2opcode, self.index2ophash, 
                        self.max_input_dims, self.max_output_dim], f)
    
    def get_training_set(self):
        return copy.deepcopy(self.fusion_types_train), copy.deepcopy(self.op_codes_train), copy.deepcopy(self.op_hashes_train), copy.deepcopy(self.feature_vectors_train), copy.deepcopy(self.labels_train)
    
    def get_test_set(self):
        return copy.deepcopy(self.fusion_types_test), copy.deepcopy(self.op_codes_test), copy.deepcopy(self.op_hashes_test), copy.deepcopy(self.feature_vectors_test), copy.deepcopy(self.labels_test)
    
    def train_size(self):
        return len(self.fusion_types_train)

    def test_size(self):
        return len(self.fusion_types_test)
    
    def _load_dataset(self, file_path):
        try:
            with open(file_path, "rb") as f:
                (self.dedupedsid2finalsid, self.max_fusion_type, self.fusion_types_train, self.fusion_types_test, self.feature_dim,
                self.op_codes_train, self.op_hashes_train, self.feature_vectors_train, self.labels_train, 
                self.op_codes_test, self.op_hashes_test, self.feature_vectors_test, self.labels_test,
                self.label_dict, self.raw_feature_dict,
                self.opcode2index, self.ophash2index, self.index2opcode, self.index2ophash, 
                self.max_input_dims, self.max_output_dim)  = pickle.load(f)
        except:
            with open(file_path, "rb") as f:
                (self.dedupedsid2finalsid, self.fusion_types_train, self.fusion_types_test, self.feature_dim,
                self.op_codes_train, self.op_hashes_train, self.feature_vectors_train, self.labels_train, 
                self.op_codes_test, self.op_hashes_test, self.feature_vectors_test, self.labels_test,
                self.label_dict, self.raw_feature_dict,
                self.opcode2index, self.ophash2index, self.index2opcode, self.index2ophash, 
                self.max_input_dims, self.max_output_dim)  = pickle.load(f)
            self.max_fusion_type = 2

    @classmethod
    def construct_kernel_dataset(cls, trace_dir, result_dir, num_samples=2000,
                                num_max_cluster_sample = 500, 
                                min_subgraph_level = None, max_subgraph_level = None):
        op_times_dict = get_execution_time_from_temp_trace(os.path.join(trace_dir, "temp.json"))
        op_times_dict_dump_path = os.path.join(result_dir, "op_running_times.pickle")
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        with open(op_times_dict_dump_path, "wb") as f:
            pickle.dump(op_times_dict, f)     
        gen_kernel_dataset(trace_dir, op_times_dict, result_dir, num_samples, min_subgraph_level=min_subgraph_level, max_subgraph_level=max_subgraph_level)
    
    @classmethod
    def construct_diverse_kernel_dataset(cls, trace_dir, result_dir, num_samples=10000, 
                          num_profiles=2000, min_subgraph_level = None, max_subgraph_level = None, 
                          op_times_dict = None, dispersion_algorithm="partitioned"):
        op_times_dict = get_execution_time_from_temp_trace(os.path.join(trace_dir, "temp.json"))
        op_times_dict_dump_path = os.path.join(result_dir, "op_running_times.pickle")
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        with open(op_times_dict_dump_path, "wb") as f:
            pickle.dump(op_times_dict, f)
        gen_diverse_kernel_dataset(trace_dir, op_times_dict, result_dir, num_samples, num_profiles,
                        min_subgraph_level=min_subgraph_level, max_subgraph_level=max_subgraph_level,
                        dispersion_algorithm=dispersion_algorithm)

class XlaModuleTestSet():
    def __init__(self, test_dataset_path, training_dataset):
        self._training_dataset = training_dataset
        self._dataset_path = test_dataset_path
        self._load_module_execution_times()
    
    def _load_module_execution_times(self):
        self._modules_dir_path = os.path.join(self._dataset_path, "modules")
        self._features_dir_path = os.path.join(self._dataset_path, "features")
        sample_id_set = set()
        for fn in os.listdir(self._modules_dir_path):
            sample_id = int(fn.split(".txt")[0].split("_")[0])
            sample_id_set.add(sample_id)

        def add_config_suffix(sid):
            return str(sid) + "_config.txt"

        def add_exec_suffix(sid):
            return str(sid) + "_exec.txt"

        module_infos_dict = {}
        labels_dict = {}
        for sample_id in sample_id_set:
            # read config.txt
            config_path = os.path.join(self._modules_dir_path, add_config_suffix(sample_id))
            elem_op_hashes = []
            fused_op_infos = []
            with open(config_path, "r") as f:
                for line in f:
                    is_fused_op = bool(int(line.split(",")[0]))
                    if is_fused_op:
                        fused_op_hash = int(line.split(",")[1])
                        kernel_path = os.path.join(self._features_dir_path, line.split(",")[-1].split("/")[-1].strip())
                        kernel_sid = int(line.split(",")[-1].split("/")[-1].split(".txt")[0])
                        with open(kernel_path, "r") as f_kernel:
                            (adj, fusion_type, computation_hash, subop_infos) = self._training_dataset._parse_feature_file(f_kernel)
                        if (computation_hash != fused_op_hash):
                            print("Inconsistent hashes for module SID: {}, kernel SID: {}".format(sample_id, kernel_sid))
                            assert False
                        # generate representations
                        fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample = self._training_dataset.gen_representation_for_sample(fusion_type, subop_infos)
                        fused_op_infos.append((fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample))
                    else:
                        _, elem_op_hash, op_code = [v.strip() for v in line.split(",")]
                        elem_op_hashes.append((int(elem_op_hash), op_code))
            # read exec.txt
            exec_path = os.path.join(self._modules_dir_path, add_exec_suffix(sample_id))
            exec_times = []
            with open(exec_path, "r") as f: 
                for line in f:
                    exec_times.append(float(line.strip()))
            if len(exec_times) > 100:
                avg_time = np.average(exec_times[-100:-10])
            else:
                avg_time = np.average(exec_times[10:-10])

            module_infos_dict[sample_id] = (elem_op_hashes, fused_op_infos)
            labels_dict[sample_id] = avg_time
        
        self.module_infos_dict = module_infos_dict
        self.labels_dict = labels_dict

    def get_module_test_sets(self):
        module_infos_as_list = []
        labels_as_list = []
        sample_ids = []
        for key, module_infos in self.module_infos_dict.items():
            module_infos_as_list.append(module_infos)
            labels_as_list.append(self.labels_dict[key])
            sample_ids.append(key)
        return module_infos_as_list, labels_as_list, sample_ids

def run_gpu_profile(profiler_exec):
    process = subprocess.run([profiler_exec], capture_output=True)
    output = process.stdout.decode("ascii")
    gflops = float(re.findall("Compute throughput: .* GFlops", output)[0].split()[2])
    gbps = float(re.findall("Memory bandwidth: .* GB/sec", output)[0].split()[2])
    return gflops, gbps

def get_time_as_sum_of_indivisual(op_time_dict, graph_def):
    total_time = 0
    for node_def in graph_def.node:
        node_name = node_def.name
        if node_name in op_time_dict:
            time, _ = op_time_dict[node_name]
            total_time += time
    return total_time

def get_next_sample_id(feature_dir):
    if not os.path.isdir(feature_dir):
        sample_id = 0
    else:
        sample_id = max([int(fname.split(".txt")[0]) for fname in os.listdir(feature_dir)]) + 1
    return sample_id

def gen_max_cluster_kernel_samples_using_replay(sample_generator, op_time_dict, dataset_dir, dataset_hlo_dir, 
                    profile_dir, min_cluster_size=4, debug_dir = None, fuse_all_fw=False):
    print("Sampling using max cluster...")
    # generate one sample
    raw_subgraph_dir = os.path.join(dataset_dir, "generated_subgraph")
    if not os.path.isdir(raw_subgraph_dir):
        os.mkdir(raw_subgraph_dir)
    # determine sample id from feature files
    feature_dir = os.path.join(dataset_dir, "features")
    
    if fuse_all_fw:
        forbidden_nodes = [node for node in sample_generator.nx_graph.nodes if node.startswith("gradients") or "Assign" in node]
        func_gen, num_clusters = sample_generator.gen_max_cluster(forbidden_nodes=forbidden_nodes, min_cluster_size=min_cluster_size)
    else:
        func_gen, num_clusters = sample_generator.gen_mxa_cluster(random_sample=True)
    total_fused_hashes = []
    sample_id = get_next_sample_id(feature_dir)
    for gen_config_fun in tqdm(func_gen, total=num_clusters):
        try:
            graph_def_path, graph_def_config_path, _ = gen_config_fun(raw_subgraph_dir, sample_id)
        except GSInternalErrors as e:
            continue

        # compile hlo
        def_path = os.path.join(raw_subgraph_dir, "{}.pbtxt".format(sample_id))
        config_path = os.path.join(raw_subgraph_dir, "{}_config.pbtxt".format(sample_id))
        unopt_path = os.path.join(dataset_hlo_dir, "{}_unopt_hlo.txt".format(sample_id))
        opt_path = os.path.join(dataset_hlo_dir, "{}_opt_hlo.txt".format(sample_id))
        try:
            compile_to_hlo(def_path, config_path, unopt_path, opt_path)
        except:
            clean_up_dir(profile_dir)
            clean_up_dir(raw_subgraph_dir)
            continue
        # copy the graph def and config for debugging purpose
        if debug_dir:
            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)
            def_name = Path(def_path).name
            shutil.copyfile(def_path, os.path.join(debug_dir, def_name))
            config_name = Path(config_path).name
            shutil.copyfile(config_path, os.path.join(debug_dir, config_name))
        # replay HLO code
        try:
            replay_and_generate_kernel_sample(sample_id, unopt_path, profile_dir, dataset_dir)
        except:
            clean_up_dir(profile_dir)
            clean_up_dir(raw_subgraph_dir)
            continue
        # copy and rename hlo file
        clean_up_dir(profile_dir)
        clean_up_dir(raw_subgraph_dir)
        # DEBUG: also return hashes of the fused kernels
        module_dir = os.path.join(dataset_dir, "modules")
        sample_config_p = os.path.join(module_dir, "{}_config.txt".format(sample_id))
        with open(sample_config_p, "r") as f:
            for line in f:
                splitted = line.split(",")
                if int(splitted[0].strip()) == 1:
                    hash_v = int(splitted[1].strip())
                    total_fused_hashes.append(hash_v)
        sample_id = get_next_sample_id(feature_dir)
    return total_fused_hashes

def gen_kernel_sample_once_using_replay(sample_generator, op_time_dict, dataset_dir, dataset_hlo_dir, 
                    profile_dir, min_levels=1, max_levels=8, debug_dir = None, ):
    # generate one sample
    raw_subgraph_dir = os.path.join(dataset_dir, "generated_subgraph")
    if not os.path.isdir(raw_subgraph_dir):
        os.mkdir(raw_subgraph_dir)
    # determine sample id from feature files
    feature_dir = os.path.join(dataset_dir, "features")
    sample_id = get_next_sample_id(feature_dir)
    # filter candidate ops
    try:
        graph_def_path, graph_def_config_path, _ = sample_generator.gen_random_subgraph(raw_subgraph_dir, sample_id, choose_root_from_ops=list(op_time_dict.keys()), min_levels=min_levels, max_levels=max_levels)
    except Exception as e:
        traceback.print_exc()
        return False, True, []
    # compile hlo
    def_path = os.path.join(raw_subgraph_dir, "{}.pbtxt".format(sample_id))
    config_path = os.path.join(raw_subgraph_dir, "{}_config.pbtxt".format(sample_id))
    unopt_path = os.path.join(dataset_hlo_dir, "{}_unopt_hlo.txt".format(sample_id))
    opt_path = os.path.join(dataset_hlo_dir, "{}_opt_hlo.txt".format(sample_id))
    try:
        compile_to_hlo(def_path, config_path, unopt_path, opt_path)
    except:
        print("[WARNING] Failed to compile to HLO code.")
        clean_up_dir(profile_dir)
        clean_up_dir(raw_subgraph_dir)
        return False, False, []
    # copy the graph def and config for debugging purpose
    if debug_dir:
        if not os.path.isdir(debug_dir):
            os.makedirs(debug_dir)
        def_name = Path(def_path).name
        shutil.copyfile(def_path, os.path.join(debug_dir, def_name))
        config_name = Path(config_path).name
        shutil.copyfile(config_path, os.path.join(debug_dir, config_name))
    # replay HLO code
    try:
        replay_and_generate_kernel_sample(sample_id, unopt_path, profile_dir, dataset_dir)
    except:
        print("[WARNING] Failed to replay HLO code and generate samples.")
        clean_up_dir(profile_dir)
        clean_up_dir(raw_subgraph_dir)
        return False, False, []
    # copy and rename hlo file
    clean_up_dir(profile_dir)
    clean_up_dir(raw_subgraph_dir)
    # DEBUG: also return hashes of the fused kernels
    fused_op_hashes = []
    module_dir = os.path.join(dataset_dir, "modules")
    sample_config_p = os.path.join(module_dir, "{}_config.txt".format(sample_id))
    with open(sample_config_p, "r") as f:
        for line in f:
            splitted = line.split(",")
            if int(splitted[0].strip()) == 1:
                hash_v = int(splitted[1].strip())
                fused_op_hashes.append(hash_v)
    return True, False, fused_op_hashes

def gen_diverse_kernel_samples_using_replay(graph_def, N, k, op_time_dict, dataset_dir, dataset_hlo_dir, 
                    profile_dir, min_levels=1, max_levels=8, debug_dir = None, dispersion_algorithm="partitioned", shape_dict_path=None):
    # generate one sample
    raw_subgraph_dir = os.path.join(dataset_dir, "generated_subgraph")
    if not os.path.isdir(raw_subgraph_dir):
        os.mkdir(raw_subgraph_dir)
    feature_dir = os.path.join(dataset_dir, "features")

    subgraph_selector = SubgraphSelector(graph_def, op_time_dict, N, raw_subgraph_dir, 
                        min_levels=min_levels, max_levels=max_levels, 
                        dispersion_algorithm=dispersion_algorithm, shape_dict_path=shape_dict_path)
    subgraph_selector.gen_samples()
    selected_sample_ids = subgraph_selector.get_subset_indices(k)
    fused_op_hashes = []
    status = []
    for subgraph_sample_id in tqdm(selected_sample_ids, leave=False):
        if not os.path.isdir(feature_dir):
            kernel_sample_id = 0
        else:
            kernel_sample_id = max([-1] + [int(fname.split(".txt")[0]) for fname in os.listdir(feature_dir)]) + 1
        # compile hlo
        def_path = os.path.join(raw_subgraph_dir, "{}.pbtxt".format(subgraph_sample_id))
        config_path = os.path.join(raw_subgraph_dir, "{}_config.pbtxt".format(subgraph_sample_id))
        unopt_path = os.path.join(dataset_hlo_dir, "{}_unopt_hlo.txt".format(kernel_sample_id))
        opt_path = os.path.join(dataset_hlo_dir, "{}_opt_hlo.txt".format(kernel_sample_id))
        try:
            compile_to_hlo(def_path, config_path, unopt_path, opt_path)
        except:
            print("[WARNING] Failed to compile to HLO code.")
            clean_up_dir(profile_dir)
            fused_op_hashes.append([])
            status.append(False)
            continue
        # copy the graph def and config for debugging purpose
        if debug_dir:
            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)
            def_name = Path(def_path).name
            shutil.copyfile(def_path, os.path.join(debug_dir, def_name))
            config_name = Path(config_path).name
            shutil.copyfile(config_path, os.path.join(debug_dir, config_name))
        # replay HLO code
        try:
            replay_and_generate_kernel_sample(kernel_sample_id, unopt_path, profile_dir, dataset_dir)
        except:
            print("[WARNING] Failed to replay HLO code and generate samples.")
            clean_up_dir(profile_dir)
            fused_op_hashes.append([])
            status.append(False)
            continue
        # copy and rename hlo file
        clean_up_dir(profile_dir)
        # DEBUG: also return hashes of the fused kernels
        fused_op_hashes_for_current_sample = []
        module_dir = os.path.join(dataset_dir, "modules")
        sample_config_p = os.path.join(module_dir, "{}_config.txt".format(kernel_sample_id))
        with open(sample_config_p, "r") as f:
            for line in f:
                splitted = line.split(",")
                if int(splitted[0].strip()) == 1:
                    hash_v = int(splitted[1].strip())
                    fused_op_hashes_for_current_sample.append(hash_v)
        fused_op_hashes.append(fused_op_hashes_for_current_sample)
        status.append(True)
    return fused_op_hashes, status

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

def gen_kernel_dataset(trace_dir, op_time_dict, result_dir, num_samples=2000, num_max_cluster_samples = 2000,
                min_subgraph_level=5, max_subgraph_level=15):
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        print("Result dir not exist. Created result dir at {}.".format(result_dir))
    dataset_dir = os.path.join(result_dir, "dataset")
    debug_dir = os.path.join(result_dir, "debug")
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
    with open(os.path.join(trace_dir, "final_graph.json"), "r") as f:
        graph_def_as_json = json.load(f)
        # clean up communication nodes
        removed_node = set()
        for node in graph_def_as_json["node"]:
            if node["op"] == "BytepsPushPull" or node["name"].lower().startswith("save") or node["name"]+":0" not in shape_dict:
                removed_node.add(node["name"])
        # while True:
        #     removed_sth = False
        #     for node in graph_def_as_json["node"]:
        #         if node["name"] in removed_node:
        #             continue
        #         if "input" in node:
        #             for input_node in node["input"]:
        #                 if input_node in removed_node and node["name"] not in removed_node:
        #                     removed_sth = True
        #                     removed_node.add(node["name"])
        #     if not removed_sth:
        #         break
        
        graph_nodes = graph_def_as_json["node"].copy()
        graph_def_as_json["node"] = []

        for node in graph_nodes:
            if node["name"] not in removed_node:
                graph_def_as_json["node"].append(node)

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
                    node["attr"]["shape"] = shape_as_list_to_pb_json(shape_dict[node["name"]+":0"])
        cleaned_graph_def_str = json.dumps(graph_def_as_json)
        with open(os.path.join(trace_dir, "cleaned_graph.json"), "w") as f_cleaned:
            json.dump(graph_def_as_json, f_cleaned, indent=4)
        graph_def = Parse(cleaned_graph_def_str, GraphDef())
    # sample_generator = SampleGenerator(graph_def=graph_def, shape_dict=shape_dict)
    sample_generator = SampleGenerator(graph_def=graph_def, shape_dict_path=os.path.join(trace_dir, "tensor_shapes.json"))
    print("Start generation.")
    completed_samples = 0
    op_hash_set = set()
    unique_op_history = []
    print("Entering max cluster sample iterations...")
    pbar = tqdm(total=num_max_cluster_samples + 1)
    # generate fuse all FW samples
    fused_op_hashes = gen_max_cluster_kernel_samples_using_replay(sample_generator, op_time_dict, dataset_dir, dataset_hlo_dir, profile_dir, debug_dir=debug_dir, fuse_all_fw=True)
    unique_ops_in_this_step = 0
    for hash_v in fused_op_hashes:
        if hash_v not in op_hash_set:
            op_hash_set.add(hash_v)
            unique_ops_in_this_step += 1
    unique_op_history.append(unique_ops_in_this_step)
    pbar.update(1)
    # generate max cluster samples
    while completed_samples < num_max_cluster_samples:
        fused_op_hashes = gen_max_cluster_kernel_samples_using_replay(sample_generator, op_time_dict, dataset_dir, dataset_hlo_dir, profile_dir, debug_dir=debug_dir, fuse_all_fw=False)
        completed_samples += 1
        pbar.update(1)
        unique_ops_in_this_step = 0
        for hash_v in fused_op_hashes:
            if hash_v not in op_hash_set:
                op_hash_set.add(hash_v)
                unique_ops_in_this_step += 1
        unique_op_history.append(unique_ops_in_this_step)
    pbar.close()
    print("Entering random sample iterations...")
    pbar = tqdm(total=num_samples)
    early_stop_counter = 0
    while completed_samples < num_samples:
        status, should_early_stop, fused_op_hashes = gen_kernel_sample_once_using_replay(sample_generator, op_time_dict, dataset_dir,
                        dataset_hlo_dir, profile_dir, min_levels=min_subgraph_level, max_levels=max_subgraph_level, debug_dir=debug_dir)
        if status:
            completed_samples += 1
            pbar.update(1)
            unique_ops_in_this_step = 0
            for hash_v in fused_op_hashes:
                if hash_v not in op_hash_set:
                    op_hash_set.add(hash_v)
                    unique_ops_in_this_step += 1
            unique_op_history.append(unique_ops_in_this_step)
        elif should_early_stop:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                print("Early stopping because no new subgraphs can be generated.")
                break
    pbar.close()
    if os.path.isdir(profile_dir):
        os.rmdir(profile_dir)
    with open(os.path.join(dataset_dir, "unique_op_history.txt"), "w") as f:
        for count in unique_op_history:
            f.write(str(count))
            f.write("\n")
    print("Dataset generation complete.")

def gen_diverse_kernel_dataset(trace_dir, op_time_dict, result_dir, num_samples=10000, 
                num_profiles=2000, min_subgraph_level=2, max_subgraph_level=100, dispersion_algorithm="partitioned"):
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        print("Result dir not exist. Created result dir at {}.".format(result_dir))
    dataset_dir = os.path.join(result_dir, "dataset")
    debug_dir = os.path.join(result_dir, "debug")
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
    if os.path.isfile(os.path.join(trace_dir, "cleaned_graph.json")):
        with open(os.path.join(trace_dir, "cleaned_graph.json"), "r") as f_cleaned:
            cleaned_graph_def_str = f_cleaned.read()
        graph_def = Parse(cleaned_graph_def_str, GraphDef())
    else:
        with open(os.path.join(trace_dir, "final_graph.json"), "r") as f:
            graph_def_as_json = json.load(f)
            # clean up communication nodes
            removed_node = set()
            for node in graph_def_as_json["node"]:
                if node["op"] == "BytepsPushPull" or node["name"].lower().startswith("save") or node["name"]+":0" not in shape_dict:
                    removed_node.add(node["name"])
            # while True:
            #     removed_sth = False
            #     for node in graph_def_as_json["node"]:
            #         if node["name"] in removed_node:
            #             continue
            #         if "input" in node:
            #             for input_node in node["input"]:
            #                 if input_node in removed_node and node["name"] not in removed_node:
            #                     removed_sth = True
            #                     removed_node.add(node["name"])
            #     if not removed_sth:
            #         break
            
            graph_nodes = graph_def_as_json["node"].copy()
            graph_def_as_json["node"] = []

            for node in graph_nodes:
                if node["name"] not in removed_node:
                    graph_def_as_json["node"].append(node)

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
                        node["attr"]["shape"] = shape_as_list_to_pb_json(shape_dict[node["name"]+":0"])
            cleaned_graph_def_str = json.dumps(graph_def_as_json)
            with open(os.path.join(trace_dir, "cleaned_graph.json"), "w") as f_cleaned:
                json.dump(graph_def_as_json, f_cleaned, indent=4)
            graph_def = Parse(cleaned_graph_def_str, GraphDef())
    print("Start generation.")
    gen_diverse_kernel_samples_using_replay(graph_def, num_samples, num_profiles, 
                                    op_time_dict, dataset_dir, dataset_hlo_dir, 
                                    profile_dir, min_levels=min_subgraph_level, 
                                    max_levels = max_subgraph_level, debug_dir=debug_dir,
                                    dispersion_algorithm=dispersion_algorithm, shape_dict_path=os.path.join(trace_dir, "tensor_shapes.json"))
    if os.path.isdir(profile_dir):
        os.rmdir(profile_dir)
    print("Dataset generation complete.")