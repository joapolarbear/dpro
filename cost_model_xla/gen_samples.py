import tensorflow as tf
from tensorflow.core.framework import types_pb2
from google.protobuf import text_format
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
from wwl import PairwiseWWL, PairwiseOverlap
import igraph
from scipy.special import softmax
from tqdm import trange, tqdm
from multiprocessing import Pool

from .p_dispersion import p_dispersion_local_search, p_dispersion_lp, parallel_p_dispersion_local_search
from .pk_graph import PKGraph, PKGraphCycleError, contract_nodes_nx, \
                        defuse_nodes_inplace_nx, postorder_contract_nx, subgraph_partition_connected_nx

def format_feed(node_name, shape):
    feed_str = "feed {{\n\tid {{ node_name: \"{}\" }}\n\tshape {{\n".format(node_name)
    for dim in shape:
        dim_str = "\t\tdim {{ size: {} }}\n".format(int(dim))
        feed_str += dim_str
    feed_str += "\t\n}\n}\n"
    return feed_str

def format_fetch(node_name):
    fetch_str = "fetch {{\n\tid {{ node_name: \"{}\" }}\n}}".format(node_name)
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

def worker_func(subgraphs, node_features, k, wid):
    pw_wwl = PairwiseWWL(subgraphs, node_features)
    selected_indices = p_dispersion_local_search(pw_wwl, k, sample_ratio=0.05, patience=10, tqdm_position=wid)
    return selected_indices

def worker_func_overlap(subgraphs, node_features, k, wid):
    pw_ovlp = PairwiseOverlap(subgraphs, node_features)
    selected_indices = p_dispersion_local_search(pw_ovlp, k, sample_ratio=0.05, patience=10, tqdm_position=wid)
    return selected_indices

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

class GraphDefUtil(object):
    def __init__(self, graph_def, shape_dict_path=None):
        super().__init__()
        self.graph_def = graph_def
        self.original_graph = tf.Graph()
        with self.original_graph.as_default():
            tf.import_graph_def(graph_def, name="")
        if shape_dict_path is not None:
            with open(shape_dict_path, "r") as f:
                shape_dict = json.load(f)
            for op in self.original_graph.get_operations():
                for output in op.outputs:
                    need_to_set_shape = False
                    for dim in output.shape:
                        if dim == -1 or dim is None:
                            need_to_set_shape = True
                            break
                    if need_to_set_shape:
                        op_shape_as_list = shape_dict[output.name]
                        output.set_shape(op_shape_as_list)
        # for op in self.original_graph.get_operations():
        #     if not self.is_fixed_shape(op.outputs[0].shape):
        #         print("{} has non fixed shapes".format(op.name))
        #         og = self.original_graph
        #         code.interact(local=locals())
        #         exit(0)
        self.operation_names = set([node.name for node in self.original_graph.get_operations()])

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
        for node_name in node_names:
            if node_name not in self.operation_names:
                raise GSNotInGraphError("[Cost Model] {} not in graph ops.".format(node_name))
        subgraph_nodes = set()
        for name in node_names:
            potential_op = self.original_graph.get_operation_by_name(name)
            forbidden_op_types = ["Transpose", "VariableV2", "Tile", "Sum", 
            "UnsortedSegmentSum", "Pad", "DynamicStitch", "Concat", "DynamicSlice", 
            "BatchToSpaceND", "BroadcastArgs","BroadcastGradientArgs", 
            "Concat", "ConcatV2", "ConcatOffset", "Empty", "FFT", "FFT2D", "FFT3D",
            "Fill", "Gather", "Scatter", "ResizeNearestNeighbor", "ResizeBilinear",
            "ResizeBilinearGrad", "ListDiff", "Slice", "InvertPermutation", 
            "Reshape", "Transpose", "ConjugateTranspose", "Range"]
            if potential_op.type not in forbidden_op_types:
                subgraph_nodes.add(potential_op)
        if len(subgraph_nodes) < 2:
            raise GSSubgraphTooSmallError("Subgraph too small (# effective ops < 2).")
        for node in subgraph_nodes:
            # check fixed shape
            if not self.is_fixed_shape(node.outputs[0].shape):
                # print("[Cost Model] {} has non-fixed shape at compile time.".format(node.name))
                raise GSNonFixedShapeError("{} has non-fixed shape at compile time.".format(node.name))
        subgraph_frontline_nodes = set()
        internal_subgraph_nodes = set()
        subgraph_input_nodes = set()
        subgraph_output_nodes = set()
        for node in subgraph_nodes:
            node_is_internal = True
            for input_tensor in node.inputs:
                input_op = input_tensor.op
                if input_op not in subgraph_nodes or input_op.type == "VariableV2":
                    subgraph_frontline_nodes.add(node)
                    node_is_internal = False
                    break
                elif input_op.type == "VariableV2":
                    pass
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
                op_inputs = n.inputs
                rewritten_input_nodes = []
                for op_input in op_inputs:
                    op_input_source = op_input.op
                    # check if shape fixed
                    if not self.is_fixed_shape(op_input_source.outputs[0].shape):
                        # print("[Cost Model] {} has non-fixed shape at compile time.".format(op_input_source.name))
                        raise GSNonFixedShapeError("{} has non-fixed shape at compile time.".format(op_input_source.name))
                    if "BroadcastGradientArgs" in op_input_source.name:
                        node_def = op_input_source.node_def
                        if not op_input_source in subgraph_input_nodes and not op_input_source in internal_subgraph_nodes and not op_input_source in subgraph_frontline_nodes:
                            subgraph_frontline_nodes.append(op_input_source)
                    if op_input_source in subgraph_input_nodes or op_input_source in internal_subgraph_nodes:
                        node_def = op_input_source.node_def
                    elif op_input_source.type == 'Const':
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
            try:
                tf.import_graph_def(out_graph_def_final, name="")
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError()
        input_nodes = []
        for node_def in out_input_defs:
            node = out_graph.get_operation_by_name(node_def.name)
            input_nodes.append(node)
        output_nodes = []
        all_ops_used_as_input = set()
        for node_def in [op.node_def for op in subgraph_nodes]:
            node = out_graph.get_operation_by_name(node_def.name)
            for input_tensor in node.inputs:
                all_ops_used_as_input.add(input_tensor.op.name)
        for node_def in [op.node_def for op in subgraph_nodes]:
            if node_def.name not in all_ops_used_as_input:
                node = out_graph.get_operation_by_name(node_def.name)
                output_nodes.append(node)
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
    def __init__(self, graph_def, shape_dict_path=None):
        self.graph_def_util = GraphDefUtil(graph_def, shape_dict_path=shape_dict_path)
        self.nx_graph = nx.DiGraph()
        self.node_sample_weights = {}
        self.edge_sample_weights = {}
        self.edge_max_weight = 1
        self._preprocess_nx_graph()
        self.generated_graph_hashes = set()

    def _preprocess_nx_graph(self):
        original_graph = self.graph_def_util.original_graph
        op_info_dict = {}
        unique_op_types = set()
        for op_name in self.graph_def_util.operation_names:
            if op_name not in op_info_dict:
                op_info_dict[op_name] = {}
            op = original_graph.get_operation_by_name(op_name)
            op_type = op.type # string
            unique_op_types.add(op_type)
            op_info_dict[op_name]["type"] = op_type
            input_size = 0
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
                # # expand with probability grow_prob
                # if random.random() < grow_prob:
                # neighbours = list(self.nx_graph.successors(node)) + list(self.nx_graph.predecessors(node))
                neighbour_edges = list([(node, succ) for succ in self.nx_graph.successors(node)]) + list([(pred, node) for pred in self.nx_graph.predecessors(node)])
                edge_frequencies = np.array([self.edge_sample_weights[self.nx_graph.edges[edge]["hash_value"]] for edge in neighbour_edges])
                edge_prob = (np.e ** (- 2 * edge_frequencies / self.edge_max_weight)) * grow_prob
                # print("[DEBUG] Edge name: {}".format(neighbour_edges))
                # print("[DEBUG] Edge freq: {}".format(edge_frequencies))
                # print("[DEBUG] Edge prob: {}".format(edge_prob))
                for index, (u, v) in enumerate(neighbour_edges):
                    candidate = u if v == node else v
                    if random.random() < edge_prob[index]:
                        if candidate not in subgraph_nodes:
                            new_frontier.add(candidate)
                            self.edge_sample_weights[self.nx_graph.edges[(u,v)]["hash_value"]] += 1
                            self.edge_max_weight = max(self.edge_max_weight, self.edge_sample_weights[self.nx_graph.edges[(u,v)]["hash_value"]])
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
                    new_successors = [succ for succ in self.nx_graph.successors(n) if succ not in subgraph_nodes]
                    all_successors += new_successors
                    for succ_n in new_successors:
                        succ_edges.append((n, succ_n))
                if not all_successors:
                    break
                filtered_successors = all_successors
            else:
                succ_edges = [(current_node, succ) for succ in filtered_successors]
            edge_frequencies = [-self.edge_sample_weights[self.nx_graph.edges[edge]["hash_value"]] for edge in succ_edges]
            edge_prob = softmax(edge_frequencies)
            next_node_index = random.choices(range(len(filtered_successors)), weights=edge_prob, k=1)[0]
            next_node = filtered_successors[next_node_index]
            prev_node = succ_edges[next_node_index][0]
            self.edge_sample_weights[self.nx_graph.edges[(prev_node ,next_node)]["hash_value"]] += 1
            self.edge_max_weight = max(self.edge_max_weight, self.edge_sample_weights[self.nx_graph.edges[(prev_node ,next_node)]["hash_value"]])
            subgraph_nodes.add(next_node)
            current_node = next_node
        return list(subgraph_nodes)
    
    def _max_cluster_sampler(self, forbidden_list):
        G: nx.DiGraph = self.nx_graph.copy()
        PKG = PKGraph(G)

        source_nodes = [node for node in G.nodes if len(G.in_edges(node)) == 0]

        # Run post order traversal on G
        for source in tqdm(source_nodes, total=len(source_nodes)):
            _, _, G = postorder_contract_nx(G, PKG, source, forbidden_list=forbidden_list, size_limit=800)
        
        clusters_formed = []
        for node_name in tqdm(G.nodes()):
            if "+" in node_name:
                cluster_nodes = node_name.split("+")
                clusters_formed.append(cluster_nodes)
        return clusters_formed

    def get_original_graph_def(self):
        return self.graph_def_util.original_graph.as_graph_def(), get_input_def_from_graph_(self.graph_def_util.original_graph)
    
    def _choose_node_with_weight(self, ops=None, op_names=None):
        if ops is None and op_names is None:
            raise TypeError("At least on of ops and op_names must be provided.")
        if op_names is None:
            op_names = [op.name for op in ops]
        op_frequencies = [-self.node_sample_weights[self.nx_graph.nodes[op_name]["hash_value"]] for op_name in op_names]
        op_weights = softmax(op_frequencies)
        chosen_node = random.choices(op_names, weights=op_weights, k=1)[0]
        self.node_sample_weights[self.nx_graph.nodes[chosen_node]["hash_value"]] += 1
        return chosen_node
    
    # this method returns a generator of functions
    def gen_max_cluster(self, forbidden_nodes=None, random_sample=False, min_cluster_size=4, forbidden_ratio=0.2):
        if forbidden_nodes is None or random_sample:
            forbidden_nodes = random.sample(self.nx_graph.nodes, k=int(len(self.nx_graph.nodes) * forbidden_ratio))
        clusters = self._max_cluster_sampler(forbidden_list=forbidden_nodes)
        def ret_gen():
            for cluster_nodes in clusters:
                def try_generate_cluster_config(output_dir, sample_id, 
                                                _min_cluster_size = min_cluster_size, 
                                                _cluster_nodes = cluster_nodes):
                    return self.gen_subgraph_def_config(_cluster_nodes, output_dir, sample_id, _min_cluster_size)
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

        graph_def_path, graph_def_config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(filtered_selected_node_names, output_dir, sample_id)

        sub_g = self.nx_graph.subgraph(filtered_selected_node_names)
        g_hash = nx.algorithms.weisfeiler_lehman_graph_hash(sub_g, node_attr="hash_value")
        if g_hash in self.generated_graph_hashes:
            raise GSDuplicateSubgraphError
        else:
            self.generated_graph_hashes.add(g_hash)
            return graph_def_path, graph_def_config_path, sub_g
    
    def gen_random_subgraph(self, output_dir, sample_id, choose_root_from_ops=None, min_levels=1, max_levels=10, forest_fire_p=0.5):
        # op selection logic
        if choose_root_from_ops is None:
            filtered_op = [op for op in self.graph_def_util.original_graph.get_operations() if op.type != "Placeholder" and op.type != "Const" and op.type != "Identity"]
        else:
            filtered_op = []
            for op_name in choose_root_from_ops:
                try:
                    op = self.graph_def_util.original_graph.get_operation_by_name(op_name)
                except Exception as e:
                    continue
                if op.type != "Placeholder" and op.type != "Const" and op.type != "Identity" and op.type != "VariableV2":
                    filtered_op.append(op)
        if not filtered_op:
            # print(choose_root_from_ops)
            raise RuntimeError("Empty filtered op!")
        duplicated_graph_count = 0
        while True:
            root_op = self._choose_node_with_weight(ops=filtered_op)
            num_levels = random.randint(min_levels, max_levels)
            # print("\033[94m [SID {}] Using max level of {}. \033[0m".format(sample_id, num_levels))
            selected_sample_method = random.choice(["forest_fire", "random_walk"])
            if selected_sample_method == "forest_fire":
                # forest fire sampling
                # print("\033[94m [SID {}] Using forest fire sampling. \033[0m".format(sample_id))
                selected_node_names = self._forest_fire_sampler(root_op, forest_fire_p, num_levels)
            elif selected_sample_method == "random_walk":
                # random walk sampling
                # print("\033[94m [SID {}] Using random walk sampling. \033[0m".format(sample_id))
                selected_node_names = self._random_walk_sampler(root_op, num_levels)
            else:
                raise RuntimeError("This should not happen")
            try:
                graph_def_path, graph_def_config_path, sub_g = self.gen_subgraph_def_config(selected_node_names, output_dir, sample_id, min_levels+1)
            except GSInternalErrors as e:
                duplicated_graph_count += 1
                if duplicated_graph_count > 200:
                    raise RuntimeError("Cannot generate new subgraphs.")
                continue
            break
        return graph_def_path, graph_def_config_path, sub_g

class SubgraphSelector():
    def __init__(self, graph_def, op_time_dict, num_samples, subgraph_dir, min_levels=2, max_levels=100, dispersion_algorithm="partitioned", shape_dict_path=None):
        self.sample_generator = SampleGenerator(graph_def, shape_dict_path=shape_dict_path)
        if not os.path.isdir(subgraph_dir):
            os.mkdir(subgraph_dir)
        self.num_samples = num_samples
        self.subgraph_dir = subgraph_dir
        self.op_time_dict = op_time_dict
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.dispersion_algorithm = dispersion_algorithm

    def _gen_subgraph_samples(self):
        sampled_subgraphs = []
        # generate subgraph samples
        for i in trange(self.num_samples):
            while True:
                try:
                    graph_def_path, graph_def_config_path, nx_subgraph = self.sample_generator.gen_random_subgraph(self.subgraph_dir, i, choose_root_from_ops=list(self.op_time_dict.keys()), min_levels=self.min_levels, max_levels=self.max_levels)
                    sampled_subgraphs.append(nx_subgraph)
                    break
                except KeyboardInterrupt as e:
                    raise RuntimeError()
                except NameError as e:
                    raise e
                except AttributeError as e:
                    raise e
                except Exception as e:
                    continue
        self.sampled_subgraphs = sampled_subgraphs
    
    def _compute_node_features(self):
        # convert all subgraphs into igraph format
        igraph_subgraphs = []
        node_features = []
        for subgraph in self.sampled_subgraphs:
            g = igraph.Graph.from_networkx(nx.DiGraph.to_undirected(subgraph))
            original_node_names = g.vs["_nx_name"]
            features = []
            for node_name in original_node_names:
                feature_vec = subgraph.nodes[node_name]["feature_vec"]
                features.append(feature_vec)
            igraph_subgraphs.append(g)
            node_features.append(np.array(features))
        # compute pair wise wwl distence
        self.igraph_subgraphs = igraph_subgraphs
        self.node_features = node_features
    
    def gen_samples(self):
        self._gen_subgraph_samples()
        self._compute_node_features()
    
    def _select_partition(self, l, i, p_size):
        return l[i*p_size:(i+1)*p_size]
    
    def get_subset_indices_partitioned(self, k, k_in_each_partition=200):
        num_partitions = int(np.ceil(k / k_in_each_partition))
        data_partition_size = int(np.ceil(len(self.igraph_subgraphs) / num_partitions))

        total_selected_indices = []
        
        map_iterable = []
        for i in range(num_partitions):
            k_in_this_partition = min(k - i*k_in_each_partition, k_in_each_partition)
            map_iterable.append( (
                self._select_partition(self.igraph_subgraphs, i, data_partition_size), 
                self._select_partition(self.node_features, i, data_partition_size),
                k_in_this_partition,
                i
                ) )
        with Pool(min(os.cpu_count(), num_partitions)) as p:
            selected_indices = p.starmap(worker_func, map_iterable)
        
        # flatten total selected_indices
        flattened_selected_indices = [idx for result in selected_indices for idx in result]

        # for i in range(num_partitions):
        #     k_in_this_partition = min(k - i*k_in_each_partition, k_in_each_partition)
        #     pw_wwl = PairwiseWWL(
        #         self._select_partition(self.igraph_subgraphs, i, data_partition_size),
        #         self._select_partition(self.node_features, i, data_partition_size),
        #         sinkhorn=True)
        #     selected_indices = calc_p_dispersion(pw_wwl, k_in_this_partition, )
        #     total_selected_indices += selected_indices
        return flattened_selected_indices

    def get_subset_indices_parallel(self, k):
        pw_wwl = PairwiseWWL(self.igraph_subgraphs, self.node_features)
        selected_indices = parallel_p_dispersion_local_search(pw_wwl, k, sample_ratio=0.05, patience=10)
        return selected_indices
    
    def get_subset_indices_overlap(self, k, k_in_each_partition=1000):
        num_partitions = int(np.ceil(k / k_in_each_partition))
        data_partition_size = int(np.ceil(len(self.igraph_subgraphs) / num_partitions))

        total_selected_indices = []
        
        map_iterable = []
        for i in range(num_partitions):
            k_in_this_partition = min(k - i*k_in_each_partition, k_in_each_partition)
            map_iterable.append( (
                self._select_partition(self.igraph_subgraphs, i, data_partition_size), 
                self._select_partition(self.node_features, i, data_partition_size),
                k_in_this_partition,
                i
                ) )
        with Pool(min(os.cpu_count(), num_partitions)) as p:
            selected_indices = p.starmap(worker_func_overlap, map_iterable)
        
        # flatten total selected_indices
        flattened_selected_indices = [idx for result in selected_indices for idx in result]
        return flattened_selected_indices
    
    def get_subset_indices(self, k):
        if self.dispersion_algorithm == "partitioned":
            return self.get_subset_indices_partitioned(k)
        elif self.dispersion_algorithm == "parallel":
            return self.get_subset_indices_parallel(k)
        elif self.dispersion_algorithm == "overlap":
            return self.get_subset_indices_overlap(k)
        else:
            raise NotImplementedError


# def gen_diverse_subgraphs(graph_def, op_time_dict, num_samples, num_profiles, subgraph_dir, min_levels=2, max_levels=100, p_dispersion_alg = "local_search"):
#     sample_generator = SampleGenerator(graph_def)
#     if not os.path.isdir(subgraph_dir):
#         os.mkdir(subgraph_dir)
#     sampled_subgraphs = []
#     # generate subgraph samples
#     for i in trange(num_samples):
#         while True:
#             try:
#                 graph_def_path, graph_def_config_path, nx_subgraph = sample_generator.gen_random_subgraph(subgraph_dir, i, choose_root_from_ops=list(op_time_dict.keys()), min_levels=min_levels, max_levels=max_levels)
#                 sampled_subgraphs.append(nx_subgraph)
#                 break
#             except KeyboardInterrupt as e:
#                 raise RuntimeError()
#             except Exception as e:
#                 continue
#     # convert all subgraphs into igraph format
#     igraph_subgraphs = []
#     node_features = []
#     for subgraph in sampled_subgraphs:
#         g = igraph.Graph.from_networkx(nx.DiGraph.to_undirected(subgraph))
#         original_node_names = g.vs["_nx_name"]
#         features = []
#         for node_name in original_node_names:
#             feature_vec = subgraph.nodes[node_name]["feature_vec"]
#             features.append(feature_vec)
#         igraph_subgraphs.append(g)
#         node_features.append(np.array(features))
#     # compute pair wise wwl distence
#     kernel_matrix = wwl(igraph_subgraphs, node_features, sinkhorn=True)
#     if p_dispersion_alg == "local_search":
#         calc_p_dispersion = p_dispersion_local_search
#     elif p_dispersion_alg == "lp":
#         calc_p_dispersion = p_dispersion_lp
#     else:
#         raise RuntimeError("Unsupported p dispersion algorithm {}".format(p_dispersion_alg))
#     selected_indices = calc_p_dispersion(kernel_matrix, num_profiles)

