import tensorflow as tf
from tensorflow.core.framework import types_pb2
from google.protobuf import text_format
import random
import code
import pickle
import numpy as np
import pickle
import os
import hashlib

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

class SampleGenerator(object):
    def __init__(self, freezed_graph_path=None, graph_def=None, shape_dict=None, cache_dir="."):
        super().__init__()
        self.shape_dict = shape_dict
        if freezed_graph_path is None and graph_def is None:
            raise RuntimeError("At least one of freezed_graph_path and graph_def must be filled.")
        if graph_def is not None:
            graph_def = graph_def
        else:
            self.freezed_graph_path_ = freezed_graph_path
            cache_name = hashlib.md5(freezed_graph_path.encode('utf-8')).hexdigest() + ".pickle"
            cache_file_path = os.path.join(cache_dir, cache_name)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            if os.path.isfile(cache_file_path):
                print("Found cached graph def.")
                with open(cache_file_path, "rb") as f:
                    graph_def = pickle.load(f)
            else:
                with tf.compat.v1.gfile.FastGFile(freezed_graph_path, "r") as f:
                    print("Reading graph...")
                    graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())
                # try to cache graph result
                with open(cache_file_path, "wb") as f:
                    pickle.dump(graph_def, f)
                print("Graph def cache saved to {}".format(cache_file_path))
        print("Parsing graph...")
        self.original_graph = tf.Graph()
        with self.original_graph.as_default():
            tf.import_graph_def(graph_def, name="")
        if self.shape_dict is not None:
            for op in self.original_graph.get_operations():
                for output in op.outputs:
                    if output.name in self.shape_dict:
                        # print("[INFO] {} set to shape {}".format(output.name, self.shape_dict[output.name]))
                        output.set_shape(self.shape_dict[output.name])
                    else:
                        print("[WARNING] {} not in shape dict.".format(output.name))
        else:
            print("[WARNING] Shape dict is not set in sample generator. Trying to read shape from graph def.")
        self.counter_ = 0

    def is_equivalent_op(self, op1, op2):
        if op1.type != op2.type or len(op1.inputs) != len(op2.inputs):
            return False
        num_inputs = len(op1.inputs)
        for i in range(num_inputs):
            input1 = op1.inputs[i]
            input2 = op2.inputs[i]
            if input1.shape != input2.shape or input1.dtype != input2.dtype:
                return False
        
    def gen_shape_type_attr_value(self, shape, data_type):
        shape_proto = shape.as_proto()
        shape_att_value = tf.compat.v1.AttrValue()
        shape_att_value.shape.CopyFrom(shape_proto)
        type_att_value = tf.compat.v1.AttrValue()
        if data_type.as_datatype_enum > 100:
            type_att_value.type =  data_type.as_datatype_enum - 100
        else:
            type_att_value.type = data_type.as_datatype_enum
        return shape_att_value, type_att_value

    def get_original_graph_def(self):
        return self.original_graph.as_graph_def(), get_input_def_from_graph_(self.original_graph)

    def gen_random_subgraph(self, choose_root_from_ops=None, min_levels=1, max_levels=10, 
                            debug_dir=None, p=0.3):
        if choose_root_from_ops is None:
            filtered_op = [op for op in self.original_graph.get_operations() if op.type != "Placeholder" and op.type != "Const" and op.type != "Identity"]
        else:
            filtered_op = []
            for op_name in choose_root_from_ops:
                try:
                    op = self.original_graph.get_operation_by_name(op_name)
                except:
                    continue
                if op.type != "Placeholder" and op.type != "Const" and op.type != "Identity" and op.type != "VariableV2":
                    filtered_op.append(op)
        if not filtered_op:
            print(choose_root_from_ops)
            raise RuntimeError("Empty filtered op!")
        op = random.choice(filtered_op)
        num_levels = random.randint(min_levels, max_levels)
        print("\033[94m Using max level of {}. \033[0m".format(num_levels))
        subgraph_nodes = set()
        subgraph_input_nodes = set()
        current_frontline_nodes = set([op])
        converted_subgraph_input_defs = []
        converted_subgraph_defs = []
        compute_nodes = 0
        # grow the graph upwards by num_levels
        while current_frontline_nodes and compute_nodes < num_levels:
            tmp_frontline_nodes = set()
            processed_frontline_nodes = set()
            for n in current_frontline_nodes:
                if n in tmp_frontline_nodes or n in processed_frontline_nodes or n in subgraph_nodes or n in subgraph_input_nodes:
                    continue
                processed_frontline_nodes.add(n)
                # with probrability p we do not grow this node at this layer
                should_grow = np.random.binomial(1,1 - p)
                if not should_grow:
                    tmp_frontline_nodes.add(n)
                    continue
                if n.type == 'Placeholder' or n.type == "VariableV2":
                    subgraph_input_nodes.add(n)
                else:
                    subgraph_nodes.add(n)
                if op.type != "Placeholder" and op.type != "Const" and op.type != "Identity" and op.type != "VariableV2":
                    compute_nodes += 1
                for input_tensor in n.inputs:
                    tmp_frontline_nodes.add(input_tensor.op)
            current_frontline_nodes = tmp_frontline_nodes
        # replace the remaining input nodes with fake inputs
        gen_placeholder_counter = 0
        processed_frontline_nodes = set()
        final_frontline_nodes = set()
        current_frontline_nodes = list(current_frontline_nodes)
        # preprocess frontline nodes
        for n in current_frontline_nodes:
            if n in processed_frontline_nodes or n in subgraph_nodes or n in subgraph_input_nodes:
                continue
            if n.type == "Reshape":
                subgraph_nodes.add(n)
                for input_tensor in n.inputs:
                    current_frontline_nodes.append(input_tensor.op)
            else:
                final_frontline_nodes.add(n)
            processed_frontline_nodes.add(n)
        processed_frontline_nodes = set()
        for idx, n in enumerate(final_frontline_nodes):
            if n in processed_frontline_nodes or n in subgraph_nodes or n in subgraph_input_nodes:
                continue
            processed_frontline_nodes.add(n)
            if n.type == 'Placeholder' or n.type == 'VariableV2':
                subgraph_input_nodes.add(n)
            elif len(n.inputs) == 0:
                subgraph_nodes.add(n)
            else:
                # op with inputs, replace all its inputs with placeholders
                op_inputs = n.inputs
                rewritten_input_nodes = []
                for op_input in op_inputs:
                    op_input_source = op_input.op
                    if op_input_source.type == 'Const':
                        node_def = op_input_source.node_def
                        subgraph_nodes.add(op_input_source)
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
                        converted_subgraph_input_defs.append(node_def)
                    rewritten_input_nodes.append(node_def)
                rewritten_node_def = tf.compat.v1.NodeDef()
                orig_output_node_def = n.node_def
                rewritten_node_def.name = orig_output_node_def.name
                rewritten_node_def.op = orig_output_node_def.op
                rewritten_node_def.device = orig_output_node_def.device
                for key in orig_output_node_def.attr.keys():
                    rewritten_node_def.attr[key].CopyFrom(orig_output_node_def.attr[key])
                rewritten_node_def.input.extend([input_node.name for input_node in rewritten_input_nodes])
                converted_subgraph_defs.append(rewritten_node_def)
        
        # add all the defs into the out graph def
        out_graph_defs = []
        for n in subgraph_nodes:
            out_graph_defs.append(n.node_def)
        out_graph_defs += converted_subgraph_defs
        out_input_defs = []
        for n in subgraph_input_nodes:
            out_input_defs.append(n.node_def)
        out_input_defs += converted_subgraph_input_defs
        out_graph_defs += out_input_defs

        out_graph_def_final = tf.compat.v1.GraphDef()
        out_graph_def_final.versions.CopyFrom(self.original_graph.as_graph_def().versions)
        out_graph_def_final.node.extend(out_graph_defs)

        if debug_dir is not None:
            # log_path = os.path.join(debug_dir, "last.pbtxt")
            # if os.path.exists(log_path):
            #     os.remove(log_path)
            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(out_graph_def_final, name="")
                input_nodes = []
            for node_def in out_input_defs:
                node = graph.get_operation_by_name(node_def.name)
                input_nodes.append(node)
            output_nodes = []
            for node_def in [op.node_def]:
                node = graph.get_operation_by_name(node_def.name)
                output_nodes.append(node)
            tf2xla_config_path = os.path.join(debug_dir, "{}_config.pbtxt".format(self.counter_))
            feed_names = []
            feed_shapes = []
            for node in input_nodes:
                feed_names.append(node.name)
                try:
                    shape_as_list = [int(value) for value in list(node.outputs[0].shape)]
                except:
                    print(node)
                    print(node.outputs[0])
                    print(node.outputs[0].shape)
                    exit(0)
                feed_shapes.append(shape_as_list)
            fetch_names = []
            for node in output_nodes:
                fetch_names.append(node.name)
            serialize_feed_fetch_to_tf2xla_config(feed_names, feed_shapes, fetch_names, tf2xla_config_path)
            tf.io.write_graph(out_graph_def_final, debug_dir, "{}.pbtxt".format(self.counter_))
            #input_defs_serialized = [idef.SerializeToString() for idef in out_input_defs]
            #output_def_serialized = [op.node_def.SerializeToString()]

            #with open("{}_io_nodes.pickle".format(self.counter_), "wb") as f:
            #    pickle.dump([input_defs_serialized, output_def_serialized], f)
            # tf.io.write_graph(out_graph_def_final, debug_dir, "last.pbtxt")
        
        self.counter_ += 1
        
        return out_graph_def_final, out_input_defs, op.node_def, self.counter_ - 1
