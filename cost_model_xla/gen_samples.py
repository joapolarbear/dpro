from collections import deque
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
import networkx as nx
import random
import traceback
import code
import pickle
import numpy as np
import pickle
import os
import hashlib
import json
import scipy
from scipy.special import softmax
from tqdm import trange, tqdm
from multiprocessing import Pool
from .pk_graph import PKGraph, PKGraphCycleError, contract_nodes_nx, \
                        defuse_nodes_inplace_nx, postorder_contract_nx, \
                        subgraph_partition_connected_nx
from .constant_utils import *
import networkx as nx

try:
    graph_hash = nx.algorithms.weisfeiler_lehman_graph_hash
except:
    import hashlib
    # https://stackoverflow.com/questions/20530455/isomorphic-comparison-of-networkx-graph-objects-instead-of-the-default-addres
    def graph_hash(G, node_attr=None):
        node_labels = dict()
        for n in G.nodes():
            if not node_attr:
                node_labels[n] = str(G.degree(n))
            else:
                node_labels[n] = str(G.nodes[n][node_attr]) + "_" + str(G.degree(n))
        _hash = hashlib.sha1(json.dumps(
            node_labels, sort_keys=True).encode()).hexdigest()
        return _hash

def format_feed(node_name, shape):
    feed_str = "feed {{\n\tid {{ node_name: \"{}\" }}\n\tshape {{\n".format(node_name)
    for dim in shape:
        dim_str = "\t\tdim {{ size: {} }}\n".format(int(dim))
        feed_str += dim_str
    feed_str += "\t}\n}\n"
    return feed_str

def format_fetch(node_name):
    fetch_str = "fetch {{\n\tid {{ node_name: \"{}\" }}\n}}\n".format(node_name)
    return fetch_str

def serialize_feed_fetch_to_tf2xla_config(feed_names, feed_shapes, fetch_names, config_path):
    with open(config_path, "w") as f:
        for index, feed_name in enumerate(feed_names):
            f.write(format_feed(feed_name, feed_shapes[index]))
            f.write("\n")
        for fetch_name in fetch_names:
            f.write(format_fetch(fetch_name))

def get_input_def_from_graph_(graph):
    input_op_defs = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            input_op_defs.append(op.node_def)
    return input_op_defs

class GSInternalErrors(Exception):
    pass

class GSNonFixedShapeError(GSInternalErrors):
    pass

class GSNotInGraphError(GSInternalErrors):
    pass

class GSSubgraphTooSmallError(GSInternalErrors):
    pass

class GSFailedToCompileSampleError(GSInternalErrors):
    pass

class GSDuplicateSubgraphError(GSInternalErrors):
    pass

class GSCannotDecideShapeError(GSInternalErrors):
    pass

class GSConstantSubgraphError(GSInternalErrors):
    pass

class GraphDefUtil(object):
    """ Utility module to process TensorFlow GraphDef. """
    def __init__(self, graph_def, shape_dict):
        """ graph_def: Tensorflow GraphDef
            shape_dict: python dict, mapping tensor name to its shape
        """
        # import byteps.tensorflow as bps # type: ignore
        super().__init__()
        self.graph_def = graph_def
        self.original_graph = tf.Graph()
        with self.original_graph.as_default():
            tf.import_graph_def(graph_def, name="")
        self.shape_dict = shape_dict

        for op in self.original_graph.get_operations():
            for output in op.outputs:
                need_to_set_shape = False
                if output.shape.rank is None:
                    need_to_set_shape = True
                else:
                    for dim in output.shape.as_list():
                        if dim == -1 or dim is None:
                            need_to_set_shape = True
                            break
                if need_to_set_shape:
                    if output.name in self.shape_dict:
                        op_shape_as_list = self.shape_dict[output.name]
                    elif "_" + output.name in self.shape_dict:
                        op_shape_as_list = self.shape_dict["_" + output.name]
                    else:
                        print("Tensor: {}, shape: {}".format(output.name, output.shape))
                        exit(-1)
                    output.set_shape(op_shape_as_list)
        self.operation_names = set([node.name for node in self.original_graph.get_operations()])
        self.original_graph_def = self.original_graph.as_graph_def(add_shapes=True)
        self.name2nodedef = {}
        for node_def in self.original_graph_def.node:
            self.name2nodedef[node_def.name] = node_def

    def gen_shape_type_attr_value(self, shape, data_type):
        shape_proto = shape.as_proto()
        shape_att_value = tf.compat.v1.AttrValue()
        shape_att_value.shape.CopyFrom(shape_proto)
        type_att_value = tf.compat.v1.AttrValue()
        type_att_value.type = data_type.as_datatype_enum
        if type_att_value.type > 100:
            type_att_value.type -= 100
        return shape_att_value, type_att_value
    
    def is_fixed_shape(self, shape):
        if shape:
            for s in shape:
                if s is None:
                    return False
        return True

    def get_subgraph_def_config_from_nodes(self, node_names, output_dir, sample_index):
        # check ops are in the graph
        all_node_names = set()
        for node_name in node_names:
            if node_name not in self.operation_names:
                raise GSNotInGraphError("[Cost Model] {} not in graph ops.".format(node_name))
            all_node_names.add(node_name)
        # constant analysis
        output_map = {}
        input_map = {}
        constant_nodes = set()
        for name in node_names:
            op = self.original_graph.get_operation_by_name(name)
            if op.type == "Const":
                constant_nodes.add(name)
            if name not in input_map:
                input_map[name] = []
            for input_tensor in op.inputs:
                input_op_name = input_tensor.op.name
                if input_tensor.op.type in ["Const", "Shape", "ShapeN"]:
                    constant_nodes.add(input_op_name)
                if input_op_name not in output_map:
                    output_map[input_op_name] = []
                output_map[input_op_name].append(name)
                input_map[name].append(input_op_name)

        constant_queue = deque(list(constant_nodes))
        while len(constant_queue) != 0:
            c_node = constant_queue.pop()
            if c_node in output_map:
                c_outputs = output_map[c_node]
                for successor_node in c_outputs:
                    all_constant = True
                    for node_input in input_map[successor_node]:
                        if node_input not in constant_nodes:
                            all_constant = False
                            break
                    if all_constant:
                        constant_nodes.add(successor_node)
                        constant_queue.appendleft(successor_node)

        subgraph_nodes = set()
        for name in node_names:
            potential_op = self.original_graph.get_operation_by_name(name)
            require_const_op_types = ["Transpose", "VariableV2", "Tile", "Sum", 
            "UnsortedSegmentSum", "Pad", "DynamicStitch", "Concat", "DynamicSlice", 
            "BatchToSpaceND", "BroadcastArgs","BroadcastGradientArgs", 
            "Concat", "ConcatV2", "ConcatOffset", "Empty", "FFT", "FFT2D", "FFT3D",
            "Fill", "Gather", "Scatter", "ResizeNearestNeighbor", "ResizeBilinear",
            "ResizeBilinearGrad", "ListDiff", "Slice", "InvertPermutation", 
            "Reshape", "Transpose", "ConjugateTranspose", "Range", "Assert"]
            if potential_op.type not in require_const_op_types or potential_op.name in constant_nodes:
                subgraph_nodes.add(potential_op)
        if len(subgraph_nodes) < 2:
            raise GSSubgraphTooSmallError("Subgraph too small (# effective ops < 2).")
        for node in subgraph_nodes:
            # check fixed shape
            for output_tensor in node.outputs:
                if not self.is_fixed_shape(output_tensor.shape):
                    raise GSNonFixedShapeError("{} has non-fixed shape at compile time. (Shape: {})".format(output_tensor.name, output_tensor.shape))
        subgraph_frontline_nodes = set()
        internal_subgraph_nodes = set()
        subgraph_input_nodes = set()

        for node in subgraph_nodes:
            node_is_internal = True
            for input_tensor in node.inputs:
                input_op = input_tensor.op
                if input_op not in subgraph_nodes or input_op.type == "VariableV2":
                    subgraph_frontline_nodes.add(node)
                    node_is_internal = False
                    break
            if node_is_internal:
                if node.type == 'Placeholder':
                    subgraph_input_nodes.add(node)
                elif node.type == "VariableV2":
                    # ignore this node
                    pass
                else:
                    internal_subgraph_nodes.add(node)
        subgraph_frontline_nodes = list(subgraph_frontline_nodes)
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
                op_input_tensors = n.inputs
                rewritten_input_nodes = []
                for op_input_tensor in op_input_tensors:
                    input_source_op = op_input_tensor.op
                    # check if shape fixed
                    if not self.is_fixed_shape(op_input_tensor.shape):
                        raise GSNonFixedShapeError("{} has non-fixed shape at compile time." \
                                                    .format(input_source_op.name))
                    if input_source_op in subgraph_input_nodes \
                            or input_source_op in internal_subgraph_nodes:
                        node_def = self.name2nodedef[input_source_op.name]
                    elif input_source_op.type == 'Const':
                        node_def = self.name2nodedef[input_source_op.name]
                        internal_subgraph_nodes.add(input_source_op)
                    elif input_source_op.type in ["Shape", "ShapeN"]:
                        # convert to constant nodes
                        if input_source_op.type == "ShapeN":
                            input_index = int(op_input_tensor.name.split(":")[-1])
                        else:
                            input_index = 0
                        target_name = input_source_op.inputs[input_index].name
                        if target_name in self.shape_dict:
                            node_def = tf.constant(self.shape_dict[target_name], 
                                                dtype=op_input_tensor.dtype).op.node_def
                            node_def.name = "generated{}_".format(gen_placeholder_counter) + \
                                            input_source_op.name
                            generated_subgraph_input_defs.append(node_def)
                        else:
                            raise GSCannotDecideShapeError("Failed to infer value for required constant {}." \
                                                            .format(op_input_tensor))
                    else:
                        if input_source_op.type == "VarHandleOp":
                            shape = tf.TensorShape(self.shape_dict[op_input_tensor.name])
                        else:
                            shape = op_input_tensor.shape
                        dtype = op_input_tensor.dtype
                        shape_attv, dtype_attv = self.gen_shape_type_attr_value(shape, dtype)
                        original_node_def = self.name2nodedef[input_source_op.name]
                        node_def = tf.compat.v1.NodeDef()
                        node_def.name = "generated{}_".format(gen_placeholder_counter) + \
                                        original_node_def.name
                        gen_placeholder_counter += 1
                        node_def.op = "Placeholder"
                        node_def.device = original_node_def.device
                        node_def.attr["dtype"].CopyFrom(dtype_attv)
                        node_def.attr["shape"].CopyFrom(shape_attv)
                        generated_subgraph_input_defs.append(node_def)
                    rewritten_input_nodes.append(node_def)
                rewritten_node_def = tf.compat.v1.NodeDef()
                orig_output_node_def = self.name2nodedef[n.name]
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
            out_graph_defs.append(self.name2nodedef[n.name])
        out_graph_defs += converted_frontline_node_defs
        out_input_defs = []
        for n in subgraph_input_nodes:
            out_input_defs.append(self.name2nodedef[n.name])
        out_input_defs += generated_subgraph_input_defs
        out_graph_defs += out_input_defs

        out_graph_def_final = tf.compat.v1.GraphDef()
        out_graph_def_final.versions.CopyFrom(self.original_graph.as_graph_def().versions)
        out_graph_def_final.node.extend(out_graph_defs)

        out_graph = tf.Graph()
        with out_graph.as_default():
            try:
                tf.import_graph_def(out_graph_def_final, name="")
            except Exception as e:
                debug_def = MessageToJson(out_graph_def_final)
                debug_def_json = json.loads(debug_def)
                with open("/root/debug_graph.json", "w") as f:
                    json.dump(debug_def_json, f, indent=4)
                traceback.print_exc()
                raise RuntimeError()
        input_nodes = []
        for node_def in out_input_defs:
            if node_def.op != "Const":
                node = out_graph.get_operation_by_name(node_def.name)
                input_nodes.append(node)
        output_nodes = []
        all_ops_used_as_input = set()
        for node_def in [op.node_def for op in subgraph_nodes]:
            node = out_graph.get_operation_by_name(node_def.name)
            for input_tensor in node.inputs:
                all_ops_used_as_input.add(input_tensor.op.name)
        for node_def in [op.node_def for op in subgraph_nodes]:
            if node_def.name not in all_ops_used_as_input and node_def.op != "Const" and \
                                                    node_def.name not in constant_nodes:
                node = out_graph.get_operation_by_name(node_def.name)
                output_nodes.append(node)
        
        if not output_nodes:
            raise GSConstantSubgraphError("[GraphDefUtil] Subgraph do not have non-constant output.")

        tf2xla_config_path = os.path.join(output_dir, "{}_config.pbtxt".format(sample_index))
        feed_names = []
        feed_shapes = []
        for node in input_nodes:
            feed_names.append(node.name)
            try:
                shape_as_list = [int(value) for value in list(node.outputs[0].shape)]
            except:
                print("[GraphDefUtil] Node {} has non-fixed shapes.".format(node.name))
                raise GSNonFixedShapeError("[GraphDefUtil] Node {} has non-fixed shapes.".format(node.name))
            feed_shapes.append(shape_as_list)
        fetch_names = []
        for node in output_nodes:
            fetch_names.append(node.name)
        serialize_feed_fetch_to_tf2xla_config(feed_names, feed_shapes, fetch_names, tf2xla_config_path)
        tf.io.write_graph(out_graph_def_final, output_dir, "{}.pbtxt".format(sample_index))
        return os.path.join(output_dir, "{}.pbtxt".format(sample_index)), tf2xla_config_path

class SampleGenerator():
    def __init__(self, graph_def, shape_dict, ignored_nodes=None):
        self.graph_def_util = GraphDefUtil(graph_def, shape_dict=shape_dict)
        self.nx_graph = nx.DiGraph()
        self.node_sample_weights = {}
        self.edge_sample_weights = {}
        self.edge_max_weight = 1
        self._preprocess_nx_graph(ignored_nodes)
        self.generated_graph_hashes = set()
        self.ignored_nodes = ignored_nodes

    def _preprocess_nx_graph(self, ignored_nodes=None):
        if ignored_nodes is not None:
            ignored_nodes = set(ignored_nodes)
        else:
            ignored_nodes = set()
        original_graph = self.graph_def_util.original_graph
        op_info_dict = {}
        unique_op_types = set()
        for op_name in self.graph_def_util.operation_names:
            if op_name in ignored_nodes:
                continue
            if op_name not in op_info_dict:
                op_info_dict[op_name] = {}
            op = original_graph.get_operation_by_name(op_name)
            op_type = op.type # string
            unique_op_types.add(op_type)
            op_info_dict[op_name]["type"] = op_type
            input_size = 0
            # op_shapes = []
            # for t in op.inputs:
            #     try:
            #         l = t.shape.as_list()
            #         op_shapes.append(l)
            #     except:
            #         print("Tensor: {}, shape: {}".format(t, t.shape))
            op_shapes = [t.shape.as_list() for t in op.inputs]
            op_hash_str = op_type
            for input_shape in op_shapes:
                for dim in input_shape:
                    op_hash_str += ",{}".format(dim)
                    if dim is not None:
                        input_size += dim
                op_hash_str += ":"
            op_info_dict[op_name]["input_size"] = input_size
            output_size = 0
            for output_tensor in op.outputs:
                shape = output_tensor.shape.as_list()
                for dim in shape:
                    if dim is not None:
                        output_size += dim
            op_info_dict[op_name]["output_size"] = output_size
            op_info_dict[op_name]["degree"] = len(op.inputs) + len(op.outputs)
            op_hash = hashlib.md5(op_hash_str.encode("utf-8")).hexdigest()
            self.nx_graph.add_node(op_name, name=op_name, hash_value=op_hash)
            self.node_sample_weights[op_hash] = 0
            for input_tensor in op.inputs:
                if input_tensor.op.name in ignored_nodes:
                    continue
                self.nx_graph.add_edge(input_tensor.op.name, op_name)
        # generate feature vector for each node
        len_type_one_hot = len(unique_op_types)
        sorted_op_types = sorted(list(unique_op_types))
        op_type_to_id = {}
        for index, op_type in enumerate(sorted_op_types):
            op_type_to_id[op_type] = index
        for node_name in self.nx_graph.nodes:
            vec = [0] * len_type_one_hot
            vec[op_type_to_id[op_info_dict[node_name]["type"]]] = 1
            vec.append(op_info_dict[node_name]["input_size"])
            vec.append(op_info_dict[node_name]["output_size"])
            vec.append(op_info_dict[node_name]["degree"])
            self.nx_graph.nodes[node_name]["feature_vec"] = vec
        for (u, v) in self.nx_graph.edges:
            edge_hash = hashlib.md5(
                        (self.nx_graph.nodes[u]["hash_value"] + \
                        self.nx_graph.nodes[v]["hash_value"]).encode("utf-8")).hexdigest()
            self.nx_graph.add_edge(u, v, hash_value = edge_hash)
            self.edge_sample_weights[edge_hash] = 0
    
    def _forest_fire_sampler(self, root, grow_prob, max_size):
        subgraph_nodes = set([root])
        # expand
        frontier = list(self.nx_graph.successors(root)) + list(self.nx_graph.predecessors(root))
        while frontier and len(subgraph_nodes) < max_size:
            new_frontier = set()
            for node in frontier:
                subgraph_nodes.add(node)
                if len(subgraph_nodes) >= max_size:
                    break
                # expand with probability grow_prob
                neighbour_edges = list([(node, succ) for succ in self.nx_graph.successors(node)]) + \
                                    list([(pred, node) for pred in self.nx_graph.predecessors(node)])
                edge_frequencies = np.array([self.edge_sample_weights[self.nx_graph.edges[edge]["hash_value"]] for edge in neighbour_edges])
                edge_prob = (np.e ** (- 2 * edge_frequencies / self.edge_max_weight)) * grow_prob

                for index, (u, v) in enumerate(neighbour_edges):
                    candidate = u if v == node else v
                    if random.random() < edge_prob[index]:
                        if candidate not in subgraph_nodes:
                            new_frontier.add(candidate)
                            self.edge_sample_weights[self.nx_graph.edges[(u,v)]["hash_value"]] += 1
                            self.edge_max_weight = max(self.edge_max_weight, 
                                    self.edge_sample_weights[self.nx_graph.edges[(u,v)]["hash_value"]])
            frontier = list(new_frontier)
            random.shuffle(frontier)
        return list(subgraph_nodes)

    def _random_walk_sampler(self, root, max_size):
        subgraph_nodes = set([root])
        current_node = root
        while len(subgraph_nodes) < max_size:
            successors = list(self.nx_graph.successors(current_node))
            filtered_successors = [n for n in successors if n not in subgraph_nodes]
            if not filtered_successors:
                all_successors = []
                succ_edges = []
                for n in subgraph_nodes:
                    new_successors = [succ for succ in self.nx_graph.successors(n) \
                                                    if succ not in subgraph_nodes]
                    all_successors += new_successors
                    for succ_n in new_successors:
                        succ_edges.append((n, succ_n))
                if not all_successors:
                    break
                filtered_successors = all_successors
            else:
                succ_edges = [(current_node, succ) for succ in filtered_successors]
            edge_frequencies = [-self.edge_sample_weights[self.nx_graph.edges[edge]["hash_value"]] \
                                                                            for edge in succ_edges]
            edge_prob = softmax(edge_frequencies)
            next_node_index = random.choices(range(len(filtered_successors)), weights=edge_prob, k=1)[0]
            next_node = filtered_successors[next_node_index]
            prev_node = succ_edges[next_node_index][0]
            self.edge_sample_weights[self.nx_graph.edges[(prev_node ,next_node)]["hash_value"]] += 1
            self.edge_max_weight = max(self.edge_max_weight, self.edge_sample_weights[
                                        self.nx_graph.edges[(prev_node ,next_node)]["hash_value"]
                                        ])
            subgraph_nodes.add(next_node)
            current_node = next_node
        return list(subgraph_nodes)
    
    def _max_cluster_sampler(self, forbidden_list, size_limit=800):
        G = self.nx_graph.copy()
        PKG = PKGraph(G)

        source_nodes = sorted(list(G.nodes), key=lambda x: G.in_degree(x))

        # Run post order traversal on G
        print("Finding maximal clusters in the graph... This may take a while...")
        visited_nodes = set()
        for source in tqdm(source_nodes, total=len(source_nodes)):
            if source not in visited_nodes and source in G.nodes:
                _, _, G = postorder_contract_nx(G, PKG, source, visited_nodes, 
                            forbidden_list=forbidden_list, size_limit=size_limit)
        
        clusters_formed = []
        for node_name in G.nodes():
            if "+" in node_name:
                cluster_nodes = node_name.split("+")
                clusters_formed.append(cluster_nodes)
        return clusters_formed

    def get_original_graph_def(self):
        return ( self.graph_def_util.original_graph.as_graph_def(), 
                get_input_def_from_graph_(self.graph_def_util.original_graph) )
    
    def _choose_node_with_weight(self, ops=None, op_names=None):
        if ops is None and op_names is None:
            raise TypeError("At least on of ops and op_names must be provided.")
        if op_names is None:
            op_names = [op.name for op in ops]
        op_frequencies = [-self.node_sample_weights[self.nx_graph.nodes[op_name]["hash_value"]] \
                                                                        for op_name in op_names]
        op_weights = softmax(op_frequencies)
        chosen_node = random.choices(op_names, weights=op_weights, k=1)[0]
        self.node_sample_weights[self.nx_graph.nodes[chosen_node]["hash_value"]] += 1
        return chosen_node
    
    # this method returns a generator of functions
    def gen_max_cluster(self, forbidden_nodes=None, random_sample=False, min_cluster_size=4, max_cluster_size=800, cache_dir=None, forbidden_ratio=0.2):
        clusters = []
        if cache_dir is not None:
            # check if cluster cache exists
            cache_path = os.path.join(cache_dir, CMPaths.MAX_CLUSTER_CACHE_FILE)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    clusters = pickle.load(f)
                compute_cluster = False
                print("Load max_cluster.pickle use cache under {}".format(cache_dir))
            else:
                compute_cluster = True
        else:
            compute_cluster = True       
        if compute_cluster:
            if forbidden_nodes is None or random_sample:
                forbidden_nodes = random.sample(self.nx_graph.nodes, 
                            k=int(len(self.nx_graph.nodes) * forbidden_ratio))
            clusters = self._max_cluster_sampler(forbidden_list=forbidden_nodes, 
                                                    size_limit=max_cluster_size)
            if cache_dir is not None:
                with open(os.path.join(cache_dir, CMPaths.MAX_CLUSTER_CACHE_FILE), "wb") as f:
                    pickle.dump(clusters, f)
        def ret_gen():
            for cluster_nodes in clusters:
                def try_generate_cluster_config(output_dir, sample_id, 
                                                _min_cluster_size = min_cluster_size, 
                                                _cluster_nodes = cluster_nodes):
                    return self.gen_subgraph_def_config(_cluster_nodes, output_dir, 
                                                        sample_id, _min_cluster_size)
                yield try_generate_cluster_config
        return ret_gen(), len(clusters)

    def gen_subgraph_def_config(self, cluster_nodes, output_dir, sample_id, min_cluster_size):
        # filter the selected nodes
        filtered_selected_node_names = []
        for node_name in cluster_nodes:
            if not "Assign" in node_name and not "Initializer" in node_name:
                filtered_selected_node_names.append(node_name)
        if len(filtered_selected_node_names) < min_cluster_size:
            raise GSSubgraphTooSmallError

        graph_def_path, graph_def_config_path = \
            self.graph_def_util.get_subgraph_def_config_from_nodes(
                        filtered_selected_node_names, output_dir, sample_id)

        sub_g = self.nx_graph.subgraph(filtered_selected_node_names)
        g_hash = graph_hash(sub_g, node_attr="hash_value")
        if g_hash in self.generated_graph_hashes:
            raise GSDuplicateSubgraphError
        else:
            self.generated_graph_hashes.add(g_hash)
            return graph_def_path, graph_def_config_path, sub_g
    
    def gen_random_subgraph(self, output_dir, sample_id, choose_root_from_ops=None, 
                            min_levels=4, max_levels=800, forest_fire_p=0.5):
        # op selection logic
        if choose_root_from_ops is None:
            filtered_op = [op for op in self.graph_def_util.original_graph.get_operations() \
                            if op.type != "Placeholder" and op.type != "Const" \
                                                        and op.type != "Identity" \
                                                        and op.name not in self.ignored_nodes]
        else:
            filtered_op = []
            for op_name in choose_root_from_ops:
                try:
                    op = self.graph_def_util.original_graph.get_operation_by_name(op_name)
                except Exception as e:
                    continue
                if op.type != "Placeholder" and op.type != "Const" \
                                            and op.type != "Identity" \
                                            and op.type != "VariableV2" \
                                            and op.name not in self.ignored_nodes:
                    filtered_op.append(op)
        if not filtered_op:
            raise RuntimeError("Empty filtered op!")
        duplicated_graph_count = 0
        while True:
            root_op = self._choose_node_with_weight(ops=filtered_op)
            num_levels = random.randint(min_levels, max_levels)
            selected_sample_method = random.choice(["forest_fire", "random_walk"])
            if selected_sample_method == "forest_fire":
                # forest fire sampling
                selected_node_names = self._forest_fire_sampler(root_op, forest_fire_p, num_levels)
            elif selected_sample_method == "random_walk":
                # random walk sampling
                selected_node_names = self._random_walk_sampler(root_op, num_levels)
            else:
                raise RuntimeError("This should not happen")
            try:
                graph_def_path, graph_def_config_path, sub_g = \
                    self.gen_subgraph_def_config(selected_node_names, 
                                                output_dir, 
                                                sample_id, min_levels+1)
            except GSInternalErrors as e:
                duplicated_graph_count += 1
                if duplicated_graph_count > 200:
                    raise RuntimeError("Cannot generate new subgraphs.")
                continue
            break
        return graph_def_path, graph_def_config_path, sub_g