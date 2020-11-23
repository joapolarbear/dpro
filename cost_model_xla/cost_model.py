import pickle
import os
import re
import shutil
import copy
import numpy as np
import networkx as nx
import xgboost as xgb
import random
from scipy import optimize as scipy_optimize
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import tensorflow as tf
from google.protobuf import text_format
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef
from google.protobuf.json_format import Parse
import json
from .gen_dataset_utils import gen_dataset, gen_kernel_dataset, profile_entire_graph, run_gpu_profile, extract_kernel_features_from_hlo, gen_diverse_kernel_dataset
from .gen_samples import GraphDefUtil
from .process_trace import get_execution_time_from_temp_trace
from .xlatools import gen_feature_vector, compile_to_hlo, replay_hlo

MAX_OP_DUR_IN_US = 10000000


"""
    Classes for fusion cost model
"""

class XlaDataset(object):
    def __init__(self, dataset_path, test_):
        super().__init__()
        self._dataset_path = dataset_path
        if not os.path.isdir(self._dataset_path):
            # not a dir, must be a pickled dataset
            self._load_dataset(dataset_path)
        else:
            self._validate_dataset()
            self._process_dataset()

    def _validate_dataset(self):
        files = os.listdir(self._dataset_path)
        for fname in ["running_time.txt", "op_running_times.pickle", "dataset"]:
            if fname not in files:
                raise RuntimeError("Cannot find {} in {}.".format(fname, self._dataset_path))
        self._label_file_path = os.path.join(self._dataset_path, "running_time.txt")
        self._op_time_dict_path = os.path.join(self._dataset_path, "op_running_times.pickle")
        self._feature_dir_path = os.path.join(self._dataset_path, "dataset", "feature_vecs")
        self._gpu_stats_path = os.path.join(self._dataset_path, "gpu_stats.txt")
        if not os.path.isdir(self._feature_dir_path):
            raise NotADirectoryError("{} is not a directory.".format(self._feature_dir_path))
        feature_vec_names = os.listdir(self._feature_dir_path)
        if not feature_vec_names:
            raise RuntimeError("Cannot find any feature vectors.")
        fv_name_regex = re.compile("^[0-9]*.txt$")
        for feature_vec_name in feature_vec_names:
            if fv_name_regex.match(feature_vec_name) is None:
                raise RuntimeError("Found unexpected feature vector file {} in dataset.".format(feature_vec_name))

    def _process_dataset(self):
        # load op_time_dict
        with open(self._op_time_dict_path, "rb") as f:
            self.op_time_dict = pickle.load(f)
            if not isinstance(self.op_time_dict, dict):
                raise TypeError("Failed to load op time dict.")
        # load gpu stats
        with open(self._gpu_stats_path, "r") as f:
            gflops, gbps = f.read().split(":")
            gflops = float(gflops.strip())
            gbps = float(gbps.strip())
        self.gflops = gflops
        self.gbps = gbps
        self.sum_op_times = []
        # load true labels
        label_dict = {}
        times = []
        with open(self._label_file_path, "r") as f:
            for line in f:
                splitted = line.split(":")
                sample_id = int(splitted[0].strip())
                time = float(splitted[1].strip())
                if time > MAX_OP_DUR_IN_US:
                    continue
                label_dict[sample_id] = time
                times.append(time)
        # load feature vecs
        feature_dict = {}
        feature_file_names = os.listdir(self._feature_dir_path)
        for fn in feature_file_names:
            vec = []
            sample_id = int(fn.split(".txt")[0].strip())
            with open(os.path.join(self._feature_dir_path, fn), "r") as f:
                for line in f:
                    value = float(line)
                    if value == -1:
                        value = 0
                    vec.append(value)
            feature_dict[sample_id] = np.array(vec)
            self.sum_op_times.append(vec[0])
        self.sum_op_times = np.array(self.sum_op_times)
        # format features and labels
        X_list = []
        y_list = []
        sample_id_list = []

        for sample_id in feature_dict.keys():
            if sample_id in label_dict:
                X_list.append(feature_dict[sample_id])
                y_list.append(label_dict[sample_id])
                sample_id_list.append(sample_id)

        X = np.array(X_list)
        # remove all zero columns
        self.feature_mask = np.all(X == 0, axis=0)
        X = X[:, ~self.feature_mask]
        # normalize X
        # X, norm = normalize(X, axis=0, norm='max', return_norm=True)

        y = np.array(y_list)

        self.X = X
        self.y = y
        # self.norm = norm
        self.sample_ids = sample_id_list

    def dump_dataset(self, file_path):
        with open(file_path, "wb") as f:
            # pickle.dump([self.X, self.y, self.feature_mask, self.op_time_dict, self.norm, self.gflops, self.gbps, self.sum_op_times, self.sample_ids], f)
            pickle.dump([self.X, self.y, self.feature_mask, self.op_time_dict, self.gflops, self.gbps, self.sum_op_times, self.sample_ids], f)
    
    def _load_dataset(self, file_path):
        with open(file_path, "rb") as f:
            # X, y, feature_mask, op_time_dict, norm, gflops, gbps, sum_op_times, sample_ids= pickle.load(f)
            X, y, feature_mask, op_time_dict, gflops, gbps, sum_op_times, sample_ids= pickle.load(f)
            self.X = X
            self.y = y
            self.feature_mask = feature_mask
            self.op_time_dict = op_time_dict
            # self.norm = norm
            self.gflops = gflops
            self.gbps = gbps
            self.sum_op_times = sum_op_times
            self.sample_ids = sample_ids

    @classmethod
    def construct_dataset(cls, trace_dir, result_dir, gpu_benchmark_cmd, num_samples=2000, 
                          min_subgraph_level = None, max_subgraph_level = None, 
                          op_times_dict = None):
        op_times_dict = get_execution_time_from_temp_trace(os.path.join(trace_dir, "temp.json"))
        op_times_dict_dump_path = os.path.join(result_dir, "op_running_times.pickle")
        gpu_stats_path = os.path.join(result_dir, "gpu_stats.txt")
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        with open(op_times_dict_dump_path, "wb") as f:
            pickle.dump(op_times_dict, f)     
        print("Benchmarking GPU stats.")
        gflops, gbps = run_gpu_profile(gpu_benchmark_cmd)
        with open(gpu_stats_path, "w") as f:
            f.write(str(gflops))
            f.write(":")
            f.write(str(gbps))
        gen_dataset(trace_dir, op_times_dict, gflops, gbps, result_dir, num_samples, min_subgraph_level=min_subgraph_level, max_subgraph_level=max_subgraph_level)
    
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
                          min_subgraph_level = None, max_subgraph_level = None, 
                          op_times_dict = None):
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
    
class ElementaryOpCache():
    def __init__(self, dataset_path=None, load_from=None):
        if load_from is not None:
            self.load(load_from)
        else:
            if dataset_path is None:
                raise RuntimeError("At least one of dataset_path and load_from must be set.")
            self._dataset_path = dataset_path
            self._cache_path = os.path.join(dataset_path, "elementary_ops.txt")
            self._load_cache()
    
    def _load_cache(self):
        total_counter = 0
        hash_dict = defaultdict(list)
        op_code_dict = defaultdict(list)
        with open(self._cache_path, "r") as f:
            for line in f:
                hash_value, op_code, time = [v.strip() for v in line.split(",")]
                hash_value = int(hash_value)
                time = float(time)
                hash_dict[hash_value].append(time)
                op_code_dict[op_code].append(time)
                total_counter += 1
        self.hash_dict = hash_dict
        self.op_code_dict = op_code_dict
        psd = []
        lens = []
        counter = 0
        for key, times in hash_dict.items():
            lens.append(len(times))
            if np.max(times) != 0:
                counter += 1
            if len(times) >= 1 and np.average(times) != 0:
                psd.append(np.std(times) / np.average(times))

        print("[OP Cache INFO] Total recorded elementary ops: {}, deduplicated size of elementary ops: {}, non_zero ops: {}".format(total_counter ,len(hash_dict), counter))
        print("[OP Cache INFO] Mean PSD: {}, Max PSD: {}, Min PSD: {}, \n max collected len: {}, min collected len: {}, mean collected len: {}".format(np.average(psd), np.max(psd), np.min(psd), np.max(lens), np.min(lens), np.average(lens)))

        elemophash2index = {}
        index2elemophash = {}
        sorted_elem_op_hashes = sorted(list(self.hash_dict.keys()))
        for index, op_hash in enumerate(sorted_elem_op_hashes):
            elemophash2index[op_hash] = index
            index2elemophash[index] = op_hash
        
        self.elemophash2index = elemophash2index
        self.index2elemophash = index2elemophash
    
    def query(self, hash_v, op_code=None):
        if hash_v not in self.hash_dict:
            if op_code is not None and op_code in self.op_code_dict:
                return np.average(self.op_code_dict[op_code]), False
            else:
                return 0, False
        else:
            return np.average(self.hash_dict[hash_v]), True

    def dump(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump([self._dataset_path, self._cache_path, 
                        self.hash_dict, self.op_code_dict,
                        self.elemophash2index, self.index2elemophash], f)
    
    def load(self, save_path):
        with open(save_path, "rb") as f:
            (self._dataset_path, self._cache_path, 
            self.hash_dict, self.op_code_dict,
            self.elemophash2index, self.index2elemophash) = pickle.load(f)

class XLAModuleOverheadModel():
    def __init__(self, dataset_path = None, dataset=None, elem_op_cache=None, load_from=None, use_dual_model=True, split_threshold=10, regularization_lambda = 0.1):
        if load_from is not None:
            self.load(load_from)
        else:
            self.regularization_lambda = regularization_lambda
            self.use_dual_model = use_dual_model
            self.split_threshold = split_threshold
            if dataset_path is None or dataset is None or elem_op_cache is None:
                raise RuntimeError("Must set all parameters if not loading from cache.")
            self.dataset = dataset
            self.elem_op_cache = elem_op_cache
            self._dataset_path = dataset_path
            self._read_execution_times()
            self._preprocess_data()

    def _read_execution_times(self):
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

        module_time_dict = {}
        module_details_dict = {}
        max_dim = len(self.elem_op_cache.index2elemophash)
        abnormal_count = 0
        for sample_id in sample_id_set:
            if sample_id not in module_details_dict:
                module_details_dict[sample_id] = []
            # read config.txt
            config_path = os.path.join(self._modules_dir_path, add_config_suffix(sample_id))
            subop_exec_time = 0
            with open(config_path, "r") as f:
                for line in f:
                    is_fused_op = bool(int(line.split(",")[0]))
                    if is_fused_op:
                        fused_op_hash = int(line.split(",")[1])
                        kernel_path = os.path.join(self._features_dir_path, line.split(",")[-1].split("/")[-1].strip())
                        op_count = -1
                        with open(kernel_path, "r") as f_kernel:
                            for kernel_line in f_kernel:
                                op_count += 1
                        kernel_sid = int(line.split(",")[-1].split("/")[-1].split(".txt")[0])
                        # get fusion execution time
                        if kernel_sid in self.dataset.label_dict:
                            exec_time = self.dataset.label_dict[kernel_sid]
                        else:
                            exec_time = self.dataset.label_dict[self.dataset.dedupedsid2finalsid[kernel_sid]]
                        module_details_dict[sample_id].append(len(self.elem_op_cache.index2elemophash) -1 + op_count)
                        max_dim = max(max_dim, len(self.elem_op_cache.index2elemophash) + op_count)
                    else:
                        _, elem_op_hash, op_code = [v.strip() for v in line.split(",")]
                        elem_op_hash = int(elem_op_hash)
                        exec_time, _ = self.elem_op_cache.query(elem_op_hash, op_code=op_code)
                        module_details_dict[sample_id].append(self.elem_op_cache.elemophash2index[elem_op_hash])
                    subop_exec_time += exec_time
            # read exec.txt
            exec_path = os.path.join(self._modules_dir_path, add_exec_suffix(sample_id))
            exec_times = []
            with open(exec_path, "r") as f: 
                for line in f:
                    exec_times.append(float(line.strip()))
            if len(exec_times) > 100:
                module_avg_time = np.average(exec_times[-100:-10])
            else:
                module_avg_time = np.average(exec_times[10:-10])
            if module_avg_time > subop_exec_time:
                module_time_dict[sample_id] = (module_avg_time, subop_exec_time)
            else:
                abnormal_count += 1
                module_details_dict.pop(sample_id)
        print("Processed {} samples, with {} have module time <= subop exec time.".format(len(sample_id_set), abnormal_count))
        
        self.module_details_dict = module_details_dict
        self.module_time_dict = module_time_dict
        self.max_dim = max_dim
        self.fusion_op_offset = len(self.elem_op_cache.index2elemophash) - 1
    
    def _preprocess_data(self):
        # generate the matrix A, and vector b
        self.row_dim = len(self.module_details_dict)
        self.column_dim = self.max_dim + 1
        A = np.zeros((self.row_dim, self.column_dim))
        b = np.zeros((self.row_dim,))
        for row_index, key in enumerate(self.module_details_dict.keys()):
            # fill A
            A[row_index, 0] = 1
            for column_index in self.module_details_dict[key]:
                A[row_index, column_index + 1] += 1
            # fill B
            module_avg_time, subop_exec_time = self.module_time_dict[key]
            b[row_index] = module_avg_time - subop_exec_time

        A_normed = A.copy()
        A_normed = (A_normed.T / b).T

        if self.use_dual_model:
            small_mask = np.sum(A, axis=1) < self.split_threshold
            A_normed_small = A_normed[small_mask]
            b_normed_small = np.ones((A_normed_small.shape[0], ))
            A_normed_large = A_normed[~small_mask]
            b_normed_large = np.ones((A_normed_large.shape[0], ))
        else:
            A_normed_small = None
            b_normed_small = None
            A_normed_large = None
            b_normed_large = None

        self.A = A
        self.A_normed = A_normed
        self.b = b
        self.b_normed = np.ones((self.row_dim, ))
        self.A_normed_small = A_normed_small
        self.b_normed_small = b_normed_small
        self.A_normed_large = A_normed_large
        self.b_normed_large = b_normed_large

    def fit(self):
        if self.use_dual_model:
            reg_A = self.regularization_lambda * np.eye(self.A_normed_large.shape[1])
            reg_b = np.zeros(self.A_normed_large.shape[1])
            coeffs = None
            residual = None
            coeffs_large, residual_large = scipy_optimize.nnls(np.concatenate((self.A_normed_large, reg_A), axis=0), np.concatenate((self.b_normed_large, reg_b), axis=0))
            coeffs_small, residual_small = scipy_optimize.nnls(np.concatenate((self.A_normed_small, reg_A), axis=0), np.concatenate((self.b_normed_small, reg_b), axis=0))
            self.avg_elem_ovhd_large = np.average(coeffs_large[1:-1])
            self.avg_elem_ovhd_small = np.average(coeffs_small[1:-1])
            self.avg_elem_ovhd = None
        else:
            reg_A = self.regularization_lambda * np.eye(self.A_normed.shape[1])
            reg_b = np.zeros(self.A_normed.shape[1])
            coeffs, residual = scipy_optimize.nnls(np.concatenate((self.A_normed, reg_A), axis=0), np.concatenate((self.b_normed, reg_b), axis=0))
            self.avg_elem_ovhd_large = None
            self.avg_elem_ovhd_small = None
            self.avg_elem_ovhd = np.average(coeffs[1:-1])
            coeffs_large = residual_large = coeffs_small = residual_small = None
        # print("Fitted model. Residual: {}".format(residual))
        self.coeffs = coeffs
        self.coeffs_large = coeffs_large
        self.coeffs_small = coeffs_small

    def refit(self, use_dual_model=True, split_threshold=5, regularization_lambda = 0.1):
        self.use_dual_model = use_dual_model
        self.split_threshold = split_threshold
        self.regularization_lambda = regularization_lambda
        self._preprocess_data()
        self.fit()
    
    def reinit(self, use_dual_model=True, split_threshold=5, regularization_lambda = 0.1):
        self._read_execution_times()
        self.refit(use_dual_model, split_threshold, regularization_lambda)
    
    # def get_overhead(self, elem_op_hashes, num_fused_ops):
    def get_overhead(self, elem_op_hashes, subop_lengths):
        sum_ovhds = 0
        num_ops = len(elem_op_hashes) + len(subop_lengths)
        for (hash_v, op_code) in elem_op_hashes:
            if hash_v in self.elem_op_cache.elemophash2index:
                if self.use_dual_model:
                    if num_ops < self.split_threshold:
                        # use small model
                        sum_ovhds += self.coeffs_small[self.elem_op_cache.elemophash2index[hash_v]]
                    else:
                        sum_ovhds += self.coeffs_large[self.elem_op_cache.elemophash2index[hash_v]]
                else:
                    sum_ovhds += self.coeffs[self.elem_op_cache.elemophash2index[hash_v]]
            else:
                if self.use_dual_model:
                    if num_ops < self.split_threshold:
                        # use small model
                        sum_ovhds += self.avg_elem_ovhd_small
                    else:
                        sum_ovhds += self.avg_elem_ovhd_large
                else:
                    sum_ovhds += self.avg_elem_ovhd
        if self.use_dual_model:
            if num_ops < self.split_threshold:
                # use small model
                sum_ovhds += self.coeffs_small[0]
                # sum_ovhds += self.coeffs_small[-1] * num_fused_ops
                for length in subop_lengths:
                    sum_ovhds += self.coeffs_small[self.fusion_op_offset + length]
            else:
                sum_ovhds += self.coeffs_large[0]
                # sum_ovhds += self.coeffs_large[-1] * num_fused_ops
                for length in subop_lengths:
                    sum_ovhds += self.coeffs_large[self.fusion_op_offset + length]
        else:
            sum_ovhds += self.coeffs[0]
            # sum_ovhds += self.coeffs[-1] * num_fused_ops
            for length in subop_lengths:
                sum_ovhds += self.coeffs[self.fusion_op_offset + length]
        return sum_ovhds
    
    def dump(self, dump_path):
        with open(dump_path, "wb") as f:
            pickle.dump([self.max_dim, self.fusion_op_offset, 
                        self.regularization_lambda, self.use_dual_model, self.split_threshold,
                        self.coeffs, self.coeffs_large, self.coeffs_small, 
                        self.avg_elem_ovhd, self.avg_elem_ovhd_large, self.avg_elem_ovhd_small, 
                        self.A, self.A_normed, self.A_normed_large, self.A_normed_small, 
                        self.b, self.b_normed, self.b_normed_large, self.b_normed_small, 
                        self.elem_op_cache, self.dataset, self._dataset_path, 
                        self.module_details_dict, self.module_time_dict, 
                        self.row_dim, self.column_dim], f)
    
    def load(self, file_path):
        try:
            with open(file_path, "rb") as f:
                (self.max_dim, self.fusion_op_offset, self.regularization_lambda, self.use_dual_model, self.split_threshold,
                self.coeffs, self.coeffs_large, self.coeffs_small, 
                self.avg_elem_ovhd, self.avg_elem_ovhd_large, self.avg_elem_ovhd_small, 
                self.A, self.A_normed, self.A_normed_large, self.A_normed_small, 
                self.b, self.b_normed, self.b_normed_large, self.b_normed_small, 
                self.elem_op_cache, self.dataset, self._dataset_path, 
                self.module_details_dict, self.module_time_dict, 
                self.row_dim, self.column_dim) = pickle.load(f)
        except:
            self.use_dual_model = False
            self.split_threshold = -1
            self.regularization_lambda = 0.1
            with open(file_path, "rb") as f:
                (self.coeffs, self.avg_elem_ovhd, self.A, self.A_normed, self.b, self.b_normed, self.avg_elem_ovhd, self.elem_op_cache,
                self.dataset, self._dataset_path, self.module_details_dict, self.module_time_dict, self.row_dim, self.column_dim) = pickle.load(f)

class XlaKernelGCNDataset(XlaKernelDataset):
    def __init__(self, dataset_path, test_size=0.2):
        super().__init__(dataset_path, test_size)
    
    def _gen_samples_representation(self):
        # transform samples into lists of shape (num_sample, num_subop, vectors)
        fusion_types = []
        op_codes = []
        op_hashes = []
        feature_vectors = []
        labels = []
        adjs = []
        for sample_id, (adj, fusion_type, _, subop_infos) in self.raw_feature_dict.items():
            fusion_type_one_hot = [0] * self.max_fusion_type
            fusion_type_one_hot[fusion_type] = 1
            op_codes_in_sample = []
            op_hashes_in_sample = []
            feature_vectors_in_sample = []
            for subop_index, (_, _, op_code, _, input_shapes, output_shape, op_hash) in enumerate(subop_infos):
                op_codes_in_sample.append(self.opcode2index[op_code] + 1)
                op_hashes_in_sample.append(self.ophash2index[op_hash] + 1)
                feature_vectors_in_sample.append(self._gen_feature_vector(input_shapes, output_shape))
            fusion_types.append(fusion_type_one_hot)
            op_codes.append(op_codes_in_sample)
            op_hashes.append(op_hashes_in_sample)
            feature_vectors.append(feature_vectors_in_sample)
            labels.append(self.label_dict[sample_id])
            adjs.append(adj)

        # split training and test sets
        (adjs_train, adjs_test, fusion_types_train, fusion_types_test,
        op_codes_train, op_codes_test, 
        op_hashes_train, op_hashes_test, 
        feature_vectors_train, feature_vectors_test, 
        labels_train, labels_test) = train_test_split(adjs, fusion_types, op_codes, op_hashes, feature_vectors, labels, test_size=self.test_size_)

        self.adjs_train = adjs_train
        self.fusion_types_train = fusion_types_train
        self.op_codes_train = op_codes_train
        self.op_hashes_train = op_hashes_train
        self.feature_vectors_train = feature_vectors_train
        self.labels_train = labels_train

        self.adjs_test = adjs_test
        self.fusion_types_test = fusion_types_test
        self.op_codes_test = op_codes_test
        self.op_hashes_test = op_hashes_test
        self.feature_vectors_test = feature_vectors_test
        self.labels_test = labels_test
    
    def dump_dataset(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump([self.adjs_train, self.adjs_test,
                        self.fusion_types_train, self.fusion_types_test, 
                        self.op_codes_train, self.op_hashes_train, self.feature_vectors_train, self.labels_train, 
                        self.op_codes_test, self.op_hashes_test, self.feature_vectors_test, self.labels_test, 
                        self.label_dict, self.raw_feature_dict,
                        self.opcode2index, self.ophash2index, self.index2opcode, self.index2ophash, 
                        self.max_input_dims, self.max_output_dim], f)

    def _load_dataset(self, file_path):
        with open(file_path, "rb") as f:
            (self.adjs_train, self.adjs_test, 
            self.fusion_types_train, self.fusion_types_test, 
            self.op_codes_train, self.op_hashes_train, self.feature_vectors_train, self.labels_train, 
            self.op_codes_test, self.op_hashes_test, self.feature_vectors_test, self.labels_test,
            self.label_dict, self.raw_feature_dict, 
            self.opcode2index, self.ophash2index, self.index2opcode, self.index2ophash, 
            self.max_input_dims, self.max_output_dim)  = pickle.load(f)

    def get_training_set(self):
        return copy.deepcopy(self.adjs_train), copy.deepcopy(self.fusion_types_train), copy.deepcopy(self.op_codes_train), copy.deepcopy(self.op_hashes_train), copy.deepcopy(self.feature_vectors_train), copy.deepcopy(self.labels_train)
    
    def get_test_set(self):
        return copy.deepcopy(self.adjs_test), copy.deepcopy(self.fusion_types_test), copy.deepcopy(self.op_codes_test), copy.deepcopy(self.op_hashes_test), copy.deepcopy(self.feature_vectors_test), copy.deepcopy(self.labels_test)

class FusionCostModel(object):
    def __init__(self, tmp_dir = "./cost_model_tmp", shape_dict_path=None):
        super().__init__()
        self.op_time_dict = None
        self.feature_mask = None
        self.model = None
        self.graph_def_util = None
        self.graph_def = None
        self.gflops = None
        self.gbps = None
        # self.norm = None
        self.predict_acceleration = None
        self._tmp_dir = tmp_dir
        if not os.path.isdir(self._tmp_dir):
            os.makedirs(self._tmp_dir)
        self._model_ready = False
        self.shape_dict_path = shape_dict_path

    def load(self, file_path):
        try:
            with open(file_path, "rb") as f:
                # self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.norm, self.predict_acceleration = pickle.load(f)
                self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.predict_acceleration = pickle.load(f)
        except:
            with open(file_path, "rb") as f:
                # self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.norm = pickle.load(f)
                self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps = pickle.load(f)
        self.graph_def_util = GraphDefUtil(self.graph_def, shape_dict_path=self.shape_dict_path)

    def dump(self, file_path):
        self._check_model_ready()
        with open(file_path, "wb") as f:
            # pickle.dump([self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.norm, self.predict_acceleration], f)
            pickle.dump([self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.predict_acceleration], f)

    def _check_model_ready(self):
        # attrs = [self.op_time_dict, self.feature_mask, self.model, self.graph_def_util, self.gflops, self.gbps, self.norm, self.predict_acceleration]
        attrs = [self.op_time_dict, self.feature_mask, self.model, self.graph_def_util, self.gflops, self.gbps, self.predict_acceleration]
        # attr_names = ["Op time dict", "feature mask", "trained model", "graph def", "gflops", "gbps", "dataset norm", "option predict_acceleration"]
        attr_names = ["Op time dict", "feature mask", "trained model", "graph def", "gflops", "gbps", "option predict_acceleration"]
        not_ready_attr_names = [name for _, name in zip(attrs, attr_names)]
        if not not_ready_attr_names:
            raise RuntimeError("Cost model incomplete. Missing {}.".format(not_ready_attr_names))

    def set_feature_mask(self, feature_mask):
        self.feature_mask = feature_mask
    
    def set_op_time_dict(self, op_time_dict):
        if isinstance(op_time_dict, str):
            with open(op_time_dict, "rb") as f:
                self.op_time_dict = pickle.load(op_time_dict)
        else:
            assert isinstance(op_time_dict, dict)
            self.op_time_dict = op_time_dict
        if not isinstance(self.op_time_dict, dict):
            raise TypeError("Unrecognized op_time_dict.")
    
    def set_graph_def(self, graph_def, shape_dict_path=None):
        self.graph_def = graph_def
        self.graph_def_util = GraphDefUtil(graph_def, shape_dict_path=shape_dict_path)
    
    def set_graph_def_from_json(self, graph_def_json_path, shape_dict_path=None):
        with open(graph_def_json_path, "r") as f:
            cleaned_graph_def_str = f.read()
        self.graph_def = Parse(cleaned_graph_def_str, GraphDef())
        self.graph_def_util = GraphDefUtil(self.graph_def, shape_dict_path=shape_dict_path)
    
    def set_gflops(self, gflops):
        self.gflops = gflops

    def set_gbps(self, gbps):
        self.gbps = gbps
    
    # def set_norm(self, norm):
    #     self.norm = norm
    
    def init_from_dataset(self, dataset, graph_def):
        if not isinstance(dataset, XlaDataset):
            raise RuntimeError("{} is not an XlaDataset instance.".format(dataset))
        self.feature_mask = dataset.feature_mask
        self.op_time_dict = dataset.op_time_dict
        self.gflops = dataset.gflops
        self.gbps = dataset.gbps
        # self.norm = dataset.norm
        if isinstance(graph_def, str):
            self.set_graph_def_from_json(graph_def)
        else:
            self.set_graph_def(graph_def)
    
    def train(self, X, y, predict_acceleration=True, sum_op_times = None, use_log_train_obj=False, use_mape_cv=False, n_jobs=16, train_two_models=False):
        if predict_acceleration and sum_op_times is None:
            raise RuntimeError("sum_op_times cannot be None when predict_acceleration is set True.")

        objective_str = "reg:squaredlogerror" if use_log_train_obj else "reg:squarederror"

        params = {
            'learning_rate': [0.1],
            'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
            'colsample_bynode': [0.2, 0.4, 0.6, 0.8, 1.0],
            'reg_alpha': [0.8, 1, 1.5, 2],
            'reg_lambda': [1,2,3,4,5],
            'max_depth': [40],
            'n_estimators': [200],
            'objective': [objective_str],
        }

        def mape_loss(y_true, y_pred):
            return np.average(np.abs(y_true - y_pred) / y_true)

        mape_scorer = make_scorer(mape_loss, greater_is_better=False)

        if train_two_models:
            self.train_two_models = True
            small_mask = sum_op_times < 200
        else:
            self.train_two_models = False
            xg_reg = xgb.XGBRegressor()
            skf = KFold(n_splits=5, shuffle = True, random_state = 1001)
            grid_search = GridSearchCV(xg_reg, params, scoring= mape_scorer if use_mape_cv else 'neg_mean_squared_error', n_jobs=n_jobs, refit=True, cv=skf, verbose=2, return_train_score=True)
            if predict_acceleration:
                self.predict_acceleration = True
                y_accel = y / sum_op_times
                grid_search.fit(X, y_accel)
            else:
                self.predict_acceleration = False
                grid_search.fit(X, y)

            print('Best neg mean squared error score: ')
            print(grid_search.best_score_)
            print('Best hyperparameters:')
            print(grid_search.best_params_)

            xg_reg_best = grid_search.best_estimator_
            self.model = xg_reg_best

    def predict(self, node_names):
        self._check_model_ready()
        graph_def_path, config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(node_names, self._tmp_dir, 0)
        if graph_def_path is None or config_path is None:
            print("[Cost Model] Failed to predict the running time of ops: {}".format(node_names))
            return -1
        unopt_hlo_path = os.path.join(self._tmp_dir, "unopt.txt")
        opt_hlo_path = os.path.join(self._tmp_dir, "opt.txt")
        feature_vec_path = os.path.join(self._tmp_dir, "feature.txt")
        try:
            compile_to_hlo(graph_def_path, config_path, unopt_hlo_path, opt_hlo_path)
            gen_feature_vector(opt_hlo_path, feature_vec_path, self.gflops, self.gbps)
            # read feature vector
            feature_vector = []
            with open(feature_vec_path, "r") as f:
                for line in f:
                    value = float(line)
                    if value == -1:
                        value = 0
                    feature_vector.append(value)
            # inject sum of running time into feature vector
            sum_op_time = 0
            for node_name in node_names:
                if node_name in self.op_time_dict:
                    time, _ = self.op_time_dict[node_name]
                    sum_op_time += time
            feature_vector.insert(0, sum_op_time)
            x = np.expand_dims(np.array(feature_vector)[~self.feature_mask], 0)

            if self.predict_acceleration:
                predicted_time = self.model.predict(x)[0] * sum_op_time
            else:
                predicted_time = self.model.predict(x)[0]
        except:
            print("[Cost Model] Failed to predict the running time of ops: {}".format(node_names))
            return -1
        # clean up
        shutil.rmtree(self._tmp_dir)
        os.makedirs(self._tmp_dir)
        return predicted_time

    def execute(self, node_names):
        self._check_model_ready()
        graph_def_path, config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(node_names, self._tmp_dir, 0)
        if graph_def_path is None or config_path is None:
            print("[Cost Model] Failed to predict the running time of ops: {}".format(node_names))
            return -1
        unopt_hlo_path = os.path.join(self._tmp_dir, "unopt.txt")
        opt_hlo_path = os.path.join(self._tmp_dir, "opt.txt")
        feature_vec_path = os.path.join(self._tmp_dir, "feature.txt")
        try:
            compile_to_hlo(graph_def_path, config_path, unopt_hlo_path, opt_hlo_path)
            predicted_time = replay_hlo(unopt_hlo_path)
        except:
            print("[Cost Model] Failed to predict the running time of ops: {}".format(node_names))
            return -1
        # clean up
        shutil.rmtree(self._tmp_dir)
        os.makedirs(self._tmp_dir)
        return predicted_time * 1e6
    
    # def predict_from_feature(self, features, sum_op_times=None, normalized=True):
    #     self._check_model_ready()
    #     if normalized:
    #         predict_features = features
    #     else:
    #         predict_features = features / self.norm
    #     if self.predict_acceleration:
    #         if sum_op_times is None:
    #             raise RuntimeError("sum_op_time cannot be none when predict_acceleration is True.")
    #         return self.model.predict(predict_features) * sum_op_times
    #     else:
    #         return self.model.predict(predict_features)
    def predict_from_feature(self, features, sum_op_times=None):
        self._check_model_ready()
        if self.predict_acceleration:
            if sum_op_times is None:
                raise RuntimeError("sum_op_time cannot be none when predict_acceleration is True.")
            return self.model.predict(features) * sum_op_times
        else:
            return self.model.predict(features)