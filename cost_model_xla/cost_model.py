import pickle
import os
import re
import shutil
import numpy as np
import xgboost as xgb
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
from .gen_dataset_utils import gen_dataset, profile_entire_graph, run_gpu_profile
from .gen_samples import *
from .process_trace import get_execution_time_from_temp_trace
from .xlatools import gen_feature_vector, compile_to_hlo, replay_hlo

MAX_OP_DUR_IN_US = 10000000


"""
    Classes for fusion cost model
"""

class XlaDataset(object):
    def __init__(self, dataset_path):
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
        graph_json_path = os.path.join(trace_dir, "graph.json")
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
    
    # @classmethod
    # def gen_op_times_dict(cls, graph, result_dir, profile_tmp_dir = None):
    #     if profile_tmp_dir is None:
    #         profile_tmp_dir = os.path.join(result_dir, "profile_tmp")
    #     sample_generator = SampleGenerator(graph=graph)
    #     print("Profiling entire graph.")
    #     op_time_dict = profile_entire_graph(sample_generator, profile_tmp_dir)
    #     if op_time_dict is None:
    #         raise RuntimeError("Failed to profile graph.")
    #     if not os.path.isdir(result_dir):
    #         os.makedirs(result_dir)
    #     output_fn = os.path.join(result_dir, "op_running_times.pickle")
    #     with open(output_fn, "wb") as f:
    #         pickle.dump(op_time_dict, f)
    #     if os.path.isdir(profile_tmp_dir):
    #         os.rmdir(profile_tmp_dir)
    #     print("Op time profiling complete.")

class GraphDefUtil(object):
    def __init__(self, graph_def):
        super().__init__()
        self.graph_def = graph_def
        self.original_graph = tf.Graph()
        with self.original_graph.as_default():
            tf.import_graph_def(graph_def, name="")
        self.operation_names = set([node.name for node in self.original_graph.get_operations()])

    def gen_shape_type_attr_value(self, shape, data_type):
        shape_proto = shape.as_proto()
        shape_att_value = tf.compat.v1.AttrValue()
        shape_att_value.shape.CopyFrom(shape_proto)
        type_att_value = tf.compat.v1.AttrValue()
        type_att_value.type = data_type.as_datatype_enum
        return shape_att_value, type_att_value

    def get_subgraph_def_config_from_nodes(self, node_names, output_dir, sample_index):
        # check ops are in the graph
        for node_name in node_names:
            if node_name not in self.operation_names:
                print("[Cost Model] {} not in graph ops.".format(node_name))
                return None, None
        subgraph_nodes = set([self.original_graph.get_operation_by_name(name) for name in node_names])
        subgraph_frontline_nodes = set()
        internal_subgraph_nodes = set()
        subgraph_input_nodes = set()
        subgraph_output_nodes = set()
        for node in subgraph_nodes:
            node_is_internal = True
            for input_tensor in node.inputs:
                    input_op = input_tensor.op
                    if input_op not in subgraph_nodes:
                        subgraph_frontline_nodes.add(node)
                        node_is_internal = False
                        break
            if node_is_internal:
                if node.type == 'Placeholder' or node.type == "VariableV2":
                    subgraph_input_nodes.add(node)
                else:
                    internal_subgraph_nodes.add(node)
        generated_subgraph_input_defs = []
        converted_frontline_node_defs = []
        gen_placeholder_counter = 0
        for idx, n in enumerate(subgraph_frontline_nodes):
            if n in internal_subgraph_nodes or n in subgraph_input_nodes:
                continue
            if n.type == 'Placeholder' or n.type == "VariableV2":
                subgraph_input_nodes.add(n)
            elif len(n.inputs) == 0:
                internal_subgraph_nodes.add(n)
            else:
                # op with inputs, replace all its inputs with placeholders
                op_inputs = n.inputs
                rewritten_input_nodes = []
                for op_input in op_inputs:
                    op_input_source = op_input.op
                    if op_input_source.type == 'Const':
                        node_def = op_input_source.node_def
                        internal_subgraph_nodes.add(op_input_source)
                    else:
                        shape = op_input.shape
                        dtype = op_input.dtype
                        shape_attv, dtype_attv = self.gen_shape_type_attr_value(shape, dtype)
                        original_node_def = op_input_source.node_def
                        node_def = tf.compat.v1.NodeDef()
                        node_def.name = "generated{}_".format(gen_placeholder_counter) + original_node_def.name
                        gen_placeholder_counter += 1
                        node_def.op = "Placeholder"
                        node_def.device = original_node_def.device
                        node_def.attr["dtype"].CopyFrom(dtype_attv)
                        node_def.attr["shape"].CopyFrom(shape_attv)
                        generated_subgraph_input_defs.append(node_def)
                    rewritten_input_nodes.append(node_def)
                rewritten_node_def = tf.compat.v1.NodeDef()
                orig_output_node_def = n.node_def
                rewritten_node_def.name = orig_output_node_def.name
                rewritten_node_def.op = orig_output_node_def.op
                rewritten_node_def.device = orig_output_node_def.device
                for key in orig_output_node_def.attr.keys():
                    rewritten_node_def.attr[key].CopyFrom(orig_output_node_def.attr[key])
                rewritten_node_def.input.extend([input_node.name for input_node in rewritten_input_nodes])
                converted_frontline_node_defs.append(rewritten_node_def)

        # add all the defs into the out graph def
        out_graph_defs = []
        for n in internal_subgraph_nodes:
            out_graph_defs.append(n.node_def)
        out_graph_defs += converted_frontline_node_defs
        out_input_defs = []
        for n in subgraph_input_nodes:
            out_input_defs.append(n.node_def)
        out_input_defs += generated_subgraph_input_defs
        out_graph_defs += out_input_defs

        out_graph_def_final = tf.compat.v1.GraphDef()
        out_graph_def_final.versions.CopyFrom(self.original_graph.as_graph_def().versions)
        out_graph_def_final.node.extend(out_graph_defs)

        out_graph = tf.Graph()
        with out_graph.as_default():
            tf.import_graph_def(out_graph_def_final, name="")
            input_nodes = []
        for node_def in out_input_defs:
            node = out_graph.get_operation_by_name(node_def.name)
            input_nodes.append(node)
        output_nodes = []
        for node_def in [op.node_def for op in subgraph_nodes]:
            node = out_graph.get_operation_by_name(node_def.name)
            output_nodes.append(node)
        tf2xla_config_path = os.path.join(output_dir, "{}_config.pbtxt".format(sample_index))
        feed_names = []
        feed_shapes = []
        for node in input_nodes:
            feed_names.append(node.name)
            shape_as_list = [int(value) for value in list(node.outputs[0].shape)]
            feed_shapes.append(shape_as_list)
        fetch_names = []
        for node in output_nodes:
            fetch_names.append(node.name)
        serialize_feed_fetch_to_tf2xla_config(feed_names, feed_shapes, fetch_names, tf2xla_config_path)
        tf.io.write_graph(out_graph_def_final, output_dir, "{}.pbtxt".format(sample_index))
        return os.path.join(output_dir, "{}.pbtxt".format(sample_index)), tf2xla_config_path

class FusionCostModel(object):
    def __init__(self, tmp_dir = "./cost_model_tmp"):
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

    def load(self, file_path):
        try:
            with open(file_path, "rb") as f:
                # self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.norm, self.predict_acceleration = pickle.load(f)
                self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.predict_acceleration = pickle.load(f)
        except:
            with open(file_path, "rb") as f:
                # self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps, self.norm = pickle.load(f)
                self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps = pickle.load(f)
        self.graph_def_util = GraphDefUtil(self.graph_def)

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
    
    def set_graph_def(self, graph_def):
        self.graph_def = graph_def
        self.graph_def_util = GraphDefUtil(graph_def)
    
    def set_graph_def_from_json(self, graph_def_json_path):
        with open(graph_def_json_path, "r") as f:
            cleaned_graph_def_str = f.read()
        self.graph_def = Parse(cleaned_graph_def_str, GraphDef())
        self.graph_def_util = GraphDefUtil(self.graph_def)
    
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