import pickle
import os
import re
import shutil
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import tensorflow as tf
from google.protobuf import text_format

from gen_dataset_utils import gen_dataset, profile_entire_graph
from gen_samples import *
from xlatools import gen_feature_vector, compile_to_hlo

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
        # format features and labels
        X_list = []
        y_list = []

        for sample_id in feature_dict.keys():
            if sample_id in label_dict:
                X_list.append(feature_dict[sample_id])
                y_list.append(label_dict[sample_id])

        X = np.array(X_list)
        X = normalize(X, axis=0, norm='max')
        y = np.array(y_list)
        # y_max = y.max()
        # y_average = np.average(y)
        
        # remove all zero columns
        self.feature_mask = np.all(X == 0, axis=0)
        X = X[:, ~self.feature_mask]
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X = X
        self.y = y

    def dump_dataset(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump([self.X, self.y, self.feature_mask, self.op_time_dict], f)
    
    def _load_dataset(self, file_path):
        with open(file_path, "rb") as f:
            X, y, feature_mask, op_time_dict = pickle.load(f)
            self.X = X
            self.y = y
            self.feature_mask = feature_mask
            self.op_time_dict = op_time_dict

    @classmethod
    def construct_dataset(cls, graph_def, result_dir, gpu_benchmark_cmd, num_samples=2000, 
                          min_subgraph_level = None, max_subgraph_level = None, 
                          op_times_dict = None):
        if op_times_dict is None:
            # need generation
            gen_op_times_dict(graph_def, result_dir)
            op_times_dict = os.path.join("op_running_times.pickle")
        if isinstance(op_times_dict, str):
            # load from file
            op_times_dict_fn = op_times_dict
            with open(op_times_dict_fn, "rb") as f:
                op_time_dict = pickle.load(f)
        else:
            # type check
            if not isinstance(op_times_dict, dict):
                raise TypeError("op_times_dict expects a string or dict but got {}.".format(type(op_times_dict)))
        gen_dataset(graph_def, op_times_dict, gpu_benchmark_cmd, result_dir, num_samples, min)
    
    @classmethod
    def gen_op_times_dict(cls, graph_def, result_dir, profile_tmp_dir = None):
        if profile_tmp_dir is None:
            profile_tmp_dir = os.path.join(result_dir, "profile_tmp")
        if isinstance(graph_def, str):
            sample_generator = SampleGenerator(freezed_graph_path=graph_def)
        else:
            sample_generator = SampleGenerator(freezed_graph_def=graph_def)
        print("Profiling entire graph.")
        op_time_dict = profile_entire_graph(sample_generator, profile_tmp_dir)
        if op_time_dict is None:
            raise RuntimeError("Failed to profile graph.")
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        output_fn = os.path.join(result_dir, "op_running_times.pickle")
        with open(output_fn, "wb") as f:
            pickle.dump(op_time_dict, f)
        if os.path.isdir(profile_tmp_dir):
            os.rmdir(profile_tmp_dir)
        print("Op time profiling complete.")

class GraphDefUtil(object):
    def __init__(self, graph_def):
        super().__init__()
        self.graph_def = graph_def
        self.original_graph = tf.Graph()
        with self.original_graph.as_default():
            tf.import_graph_def(graph_def, name="")

    def gen_shape_type_attr_value(self, shape, data_type):
        shape_proto = shape.as_proto()
        shape_att_value = tf.compat.v1.AttrValue()
        shape_att_value.shape.CopyFrom(shape_proto)
        type_att_value = tf.compat.v1.AttrValue()
        type_att_value.type = data_type.as_datatype_enum
        return shape_att_value, type_att_value

    def get_subgraph_def_config_from_nodes(self, node_names, output_dir, sample_index):
        subgraph_nodes = set([self.original_graph.get_operation_by_name(name) for name in node_names])
        subgraph_frontline_nodes = set()
        internal_subgraph_nodes = set()
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
                internal_subgraph_nodes.add(node)
        subgraph_input_nodes = set()
        generated_subgraph_input_defs = []
        converted_frontline_node_defs = []
        gen_placeholder_counter = 0
        for idx, n in enumerate(subgraph_frontline_nodes):
            if n in internal_subgraph_nodes or n in subgraph_input_nodes:
                continue
            if n.type == 'Placeholder':
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
        self._tmp_dir = tmp_dir
        if not os.path.isdir(self._tmp_dir):
            os.makedirs(self._tmp_dir)
        self._model_ready = False
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps = pickle.load(f)
        self.graph_def_util = GraphDefUtil(self.graph_def)

    def dump(self, file_path):
        self._check_model_ready()
        with open(file_path, "wb") as f:
            pickle.dump([self.op_time_dict, self.feature_mask, self.model, self.graph_def, self.gflops, self.gbps], f)

    def _check_model_ready(self):
        attrs = [self.op_time_dict, self.feature_mask, self.model, self.graph_def_util, self.gflops, self.gbps]
        attr_names = ["Op time dict", "feature mask", "trained model", "graph def", "gflops", "gbps"]
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
    
    def set_gflops(self, gflops):
        self.gflops = gflops

    def set_gbps(self, gbps):
        self.gbps = gbps
    
    def train(self, X, y):
        params = {
            'learning_rate': [0.1],
            'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
            'colsample_bynode': [0.2, 0.4, 0.6, 0.8, 1.0],
            'reg_alpha': [0.8, 1, 1.5, 2],
            'reg_lambda': [1,2,3,4,5],
            'max_depth': [50],
            'n_estimators': [200],
            'objective': ['reg:squarederror'],
        }

        xg_reg = xgb.XGBRegressor()
        skf = KFold(n_splits=5, shuffle = True, random_state = 1001)
        grid_search = GridSearchCV(xg_reg, params, scoring='neg_mean_squared_error', n_jobs=16, refit=True, cv=skf, verbose=2, return_train_score=True)
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
        unopt_hlo_path = os.path.join(self._tmp_dir, "unopt.txt")
        opt_hlo_path = os.path.join(self._tmp_dir, "opt.txt")
        feature_vec_path = os.path.join(self._tmp_dir, "feature.txt")
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
        predicted_time = self.model.predict(x)[0]
        # clean up
        shutil.rmtree(self._tmp_dir)
        os.makedirs(self._tmp_dir)
        return predicted_time
    
    def predict_from_feature(self, features):
        self._check_model_ready()
        return self.model.predict(features)