import numpy as np
import json
import os
import re
import networkx as nx
from ml_platform.tensorflow.amp_lists import whitelist, blacklist, greylist, clearlist
from logger_utils import Singleton, SingleLogger
from trace_utils import FileName, parse_op_name

OP_HYPER_PARAMETERS = {
    "Conv2D": ["H", "W", "C", "R", "S", "P", "Q", "K", "B", "use_bias"],
    "MatMul": ["C_in", "C_out", "B"],
    "CastToFp16": [],
    "CastToFp32": []
}
FULL_HEADERS = {"base": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei"]}
BASE_HEADER_LEN = len(FULL_HEADERS['base'])
for key in OP_HYPER_PARAMETERS:
    FULL_HEADERS[key] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS[key]


### TODO (urgent), replace this
GFLOPS_FP32 = 1
GFLOPS_FP16 = 2

def tf_relabel_func(_name, update_nodes_in_dag):
    for prefix in ["Comm.", "Comp.", "BW.", "FW.", "UPDATE_."]:
        if _name.startswith(prefix):
            return _name
    if _name.startswith("^"):
        _name = _name[1:]
    last_slash_pos = _name.rfind("/")
    if last_slash_pos != -1 and last_slash_pos < len(_name)-1 and _name[last_slash_pos+1] == "_":
        _name = _name[:last_slash_pos]
    if "BytePSPushPull" in _name and "tensor" not in _name:
        _name = "Comm." + _name
    elif "input_barrier" in _name:
        _name = "FW." + _name
    elif "allreduce" in _name.lower():
        if "." in _name:
            _, tensor_name = _name.split(".")
            if "_" in tensor_name:
                tensor_name = tensor_name.split("_")[0]
            if "Switch" in _name:
                _name = "BW." + tensor_name + "_Switch"
            else:
                _name = "Comm." + tensor_name
        elif "cond_" in _name and "Switch" in _name:
            _name = "BW." + _name
        else:
            _name = "UPDATE_." + _name
    else:
        if update_nodes_in_dag is not None and _name in update_nodes_in_dag:
            _name = "UPDATE_." + _name
        elif _name.startswith("gradients") or _name.startswith("gradient_tape"):
            _name = "BW." + _name
        else:
            _name = "FW." + _name
    return _name

def wrap_read_graphdef(graphdef_path):
    with open(graphdef_path, "r") as f:
        graph_def = json.load(f)
    graph = nx.DiGraph()
    for node in graph_def["node"]:
        if "input" in node:
            for input_tensor_name in node["input"]:
                input_node_name = input_tensor_name.split(":")[0]
                graph.add_edge(input_node_name, node["name"])
    update_nodes_in_dag = set()
    def recursive_add_succs(_node):
        for succ_ in graph.successors(_node):
            update_nodes_in_dag.add(succ_)
            recursive_add_succs(succ_)
    for node in graph.nodes:
        if "allreduce" in node.lower() or "bytepspushpull" in node.lower():
            recursive_add_succs(node)
    new_graph = nx.DiGraph()
    for u, v in graph.edges:
        new_graph.add_edge(tf_relabel_func(u, update_nodes_in_dag), tf_relabel_func(v, update_nodes_in_dag))
    return new_graph, update_nodes_in_dag

class MetaInfo:
    def __init__(self, meta_dir):
        with open(os.path.join(meta_dir, FileName.METADATA.value), 'r') as fp:
            self.tf_meta = json.load(fp)

        try:
            with open(os.path.join(meta_dir, "gradient_name2ID.json"), 'r') as fp:
                name2id = json.load(fp)
                self.gradient_name_list = [name.replace(
                    "HorovodAllreduce.", "") for name, _ in sorted(name2id.items(), key=lambda x: x[1])]
        except FileNotFoundError:
            SingleLogger().warn("{} is not found !".format(FileName.TENSOR_NAME.value))
            self.gradient_name_list = []

        ### Batch size used for this meta data file
        self.old_B = None
        self.cache_hyper_para = {}
        ### Tranverse all metadata in advance, cache some importand operaots' metadata
        ### And get self.old_B
        self.get_hyper_para()

        self.local_dfg = None
        self.update_nodes_in_dag = None
        ### Mapping from tensor names to weight variable names
        self.tensor2weight = {}
        self._wrap_read_dfg(os.path.join(meta_dir, FileName.DAG.value))

    def pick_opnames_by_op_type(self, op_type):
        return [_name for _name in self.cache_hyper_para.keys() if self.parse_op_type(_name) == op_type]

    def parse_op_type(self, op_name):
        ### Special cases
        if "CastToFp16" in op_name:
            return "CastToFp16"
        elif "CastToFp32" in op_name:
            return "CastToFp32"
        if "ConstantFolding" in op_name:
            return "unknown"

        if op_name not in self.tf_meta:
            raise KeyError("{} not in the metadata".format(op_name))
        if "op" not in self.tf_meta[op_name]:
            raise KeyError("op {} has not been given a type".format(op_name))
        return self.tf_meta[op_name]["op"]

    def ret_tensor_size(self, tensor_id):
        tensor_name = self.gradient_name_list[tensor_id]
        weight_name = self.tensor2weight[tensor_name]
        
        bw_ops = list(self.local_dfg.predecessors(
            "Comm.{}".format(tensor_id)))
        assert len(bw_ops) == 1, (tensor_id, bw_ops)
        assert bw_ops[0].startswith("BW"), (tensor_id, bw_ops)

        bw_op_name = parse_op_name(bw_ops[0])
        outputs = self.tf_meta[bw_op_name]["output"]
        for output in outputs:
            if output["name"] == weight_name:
                dtype_size = self.dtype2size(output["dtype"])
                return np.prod(output["shape"]) * dtype_size

        raise ValueError("Fail to find the weight variable {}".format(weight_name))
    
    def ret_output_size_inB(self, op_name):
        raise NotImplementedError("Distinguish weights and activations")
        outputs = self.tf_meta[op_name]["output"]
        dtype_size = self.dtype2size(outputs[0]["dtype"])
        return np.prod(outputs[0]["shape"]) * dtype_size

    def ret_output_shape(self, op_name):
        outputs = self.tf_meta[op_name]["output"]
        return outputs[0]["shape"]

    def ret_op_precision(self, op_name):
        if op_name not in self.tf_meta:
            return None
        outputs = self.tf_meta[op_name]["output"]
        if len(outputs) == 0:
            return None
        return outputs[0]["dtype"]

    def ret_metadata(self, op_name, batch_size=None):
        '''
        Return:
            S_mul, S_add, S_in, S_out, S_weight
        '''
        op_type = self.parse_op_type(op_name)
        wei = 1 if batch_size is None else batch_size / self.old_B
        inputs = self.tf_meta[op_name]["input"]
        outputs = self.tf_meta[op_name]["output"]
        if op_type == "Conv2D":
            H, W, C, R, S, P, Q, K, old_B, use_bias = self.cache_hyper_para[op_name]
            return wei*old_B*K*P*Q*C*R*S, wei*old_B*K*P*Q*(C*R*S-1), wei*old_B*H*W*C, wei*old_B*P*Q*K, R*S*C*K
        elif op_type == "MatMul":
            C_in, C_out, old_B = self.cache_hyper_para[op_name]
            if old_B is None:
                assert self.old_B is not None
                old_B = self.old_B
            return wei*old_B*C_in*C_out, wei*old_B*(C_in-1)*C_out, wei*old_B*C_in, wei*old_B*C_out, C_in*C_out
        elif op_type == "BW_MatMul":
            C_in, C_out, old_B = self.cache_hyper_para[op_name]
            return wei*old_B*C_in*C_out, (wei*old_B-1)*C_in*C_out, wei*old_B*C_in, C_in*C_out, wei*old_B*C_out
        elif op_type == "CastToFp16" or op_type == "CastToFp32":
            pre_node = op_name.split("-")[0]
            assert pre_node in self.tf_meta
            return 1, 1, self.ret_output_size_inB(pre_node), 1, 1
        elif op_type == "Conv2DBackpropFilter":
            S_out = wei * np.prod(outputs[0]["shape"])
            SingleLogger().debug("{}({}) not fully implemented".format(op_name, op_type))
            S_in = wei * np.prod(inputs[-1]["shape"])
            S_wei = wei * np.prod(inputs[0]["shape"])
            return 1, 1, S_in, S_out, S_wei
        elif op_type == "unknown":
            return 0, 0, 0, 0, 0
        elif op_type == "Const":
            S_out = wei * np.prod(outputs[0]["shape"])
            return 0, 0, 0, S_out, 0
        else:
            ### Not cached operators
            if len(outputs) > 1:
                SingleLogger().debug("{} has multiple outputs: {}".format(op_name, outputs)) 
            S_out = wei * np.prod(outputs[0]["shape"])
            SingleLogger().debug("{} has not been fine-defined input/weight: {}".format(op_name, inputs))
            S_in = wei * np.prod(inputs[0]["shape"])
            S_wei = wei * np.prod(inputs[1]["shape"]) if len(inputs) > 1 else 1
            return 0, 0, S_in, S_out, S_wei

    def ret_rawmeta(self, op_name):
        op_type = self.parse_op_type(op_name)
        if op_type == "CastToFp16" or op_type == "CastToFp32":
            return []
        assert op_name in self.tf_meta, "shape info for {} is not in the meta data".format(
            op_name)
        return self.cache_hyper_para[op_name]

    def get_hyper_para(self):
        for op_name in self.tf_meta.keys():  
            inputs = self.tf_meta[op_name]["input"]
            outputs = self.tf_meta[op_name]["output"]
            op_type = self.parse_op_type(op_name)
            if op_type == "Conv2D":
                assert len(outputs) == 1
                shape_ = outputs[0]["shape"]
                assert len(shape_) == 4, (outputs[0]["shape"], self.tf_meta[op_name])
                N = shape_[0]
                if N is None:
                    continue
                if self.old_B is None:
                    self.old_B = N
                # P = shape_[1]
                ### TODO (huhanpeng), assume the width=height
                P = Q = shape_[2]
                ### different layout
                K = shape_[3] if shape_[1] == P else shape_[1]

                assert len(inputs) == 2
                C = None         # input channel
                H = W = None     # Input height/weight
                R = S = None     # kernel size
                for input_ in inputs:
                    shape_ = input_["shape"]
                    assert len(shape_) == 4
                    if "kernel" in input_["name"] or "ReadVariableOp" in input_["name"]:
                        ### weight
                        R, S = shape_[0], shape_[1]
                        if C is None:
                            C = shape_[2]
                        else:
                            assert C == shape_[2]
                        assert K == shape_[3]
                    else:
                        ### Input
                        assert shape_[0] == N, self.tf_meta[op_name]
                        H = W = shape_[2]
                        if C is None:
                            C = shape_[3] if shape_[1] == H else shape_[1]
                        else:
                            assert C == shape_[3] if shape_[1] == H else shape_[1]
                self.cache_hyper_para[op_name] = [H, W, C, R, S, P, Q, K, N, 0]
                assert not None in self.cache_hyper_para[op_name], self.tf_meta[op_name]
                assert not self.old_B is None
            elif op_type == "MatMul":
                B = C_in = C_out = None
                assert len(inputs) == 2 and len(
                    inputs[0]["shape"]) == 2 and len(inputs[1]["shape"]) == 2
                
                found = False
                for i in range(2):
                    if "kernel" in inputs[i]["name"] or "ReadVariableOp" in inputs[i]["name"]:
                        ### i is weight, 1-i is input
                        B, C_in = inputs[1-i]["shape"]
                        if C_in == inputs[i]["shape"][0]:
                            C_out = inputs[i]["shape"][1]
                        else:
                            C_out = inputs[i]["shape"][0]
                            assert C_in == inputs[i]["shape"][1]
                        assert (outputs[0]["shape"][0] == B and outputs[0]["shape"][1] == C_out), self.tf_meta[op_name]
                        found = True
                        break
                if not found:
                    B, C_out = outputs[0]["shape"]
                    for _shape in inputs[0]["shape"]:
                        if _shape != B and _shape != C_out:
                            C_in = _shape
                            break

                self.cache_hyper_para[op_name] = [C_in, C_out, B]
            # elif op_type == "Cast":
            #     assert len(outputs) == 1
            #     assert len(inputs) == 1 
            #     dtype_in_size = self.dtype2size(inputs[0]["dtype"])
            #     dtype_out_size = self.dtype2size(outputs[0]["dtype"])
            #     self.cache_hyper_para[op_name] = [
            #         np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), dtype_in_size, dtype_out_size, batch_size]
            else:
                # SingleLogger().warn(
                #     "Metadata for {} is not implemented yet. {}".format(op_name, op_type))
                pass
    
    def dtype2size(self, _dtype):
        if _dtype == "float32":
            return 4
        elif _dtype == "int32":
            return 4
        elif _dtype == "float16":
            return 2
        elif _dtype == "int16":
            return 2
        elif _dtype == "int64":
            return 8
        elif _dtype == "float64":
            return 8
        elif _dtype == "bool":
            return np.size(bool)
        elif _dtype == "string":
            return 1
        else:
            raise ValueError("{} not defined".format(_dtype))

    def check_amp_lists(self, op_name):
        try:
            op_type = self.tf_meta[op_name]["op"]
        except KeyError:
            return

        if op_type in whitelist:
            return "white"
        elif op_type in blacklist:
            return "black"
        elif op_type in greylist:
            return "grey"
        elif op_type in clearlist:
            return "clear"
        else:
            return
        # TODO (huhanpeng): use a more complex rule, just like in AMP of TensorFlow.
    
    def in_metadata(self, op_name):
        return op_name in self.tf_meta

    def is_const(self, op_name):
        return self.parse_op_type(op_name) == "Const"
    
    def is_variable(self, op_name):
        return self.parse_op_type(op_name) in ["Variable", "VariableV2", "AutoReloadVariable",
            "VarHandleOp", "ReadVariableOp",
            "_VarHandlesOp", "_ReadVariablesOp"]
    
    def read_dfg_with_var(self):
        g = nx.DiGraph()
        edges_to_add = []
        for op in self.tf_meta.keys():
            if "input" not in self.tf_meta[op] or "output" not in self.tf_meta[op]:
                continue
            for _input in self.tf_meta[op]["input"]:
                ### _input is a dict
                var_name = _input["name"]
                edges_to_add.append((var_name, op))
            for _output in self.tf_meta[op]["output"]:
                ### _output is a dict
                var_name = _output["name"]
                edges_to_add.append((op, var_name))
        g.add_edges_from(edges_to_add)
        return g
    
    def wrap_read_dfg(self, gml_path):
        if self.local_dfg is None:
            self._wrap_read_dfg(gml_path)
        
        return self.local_dfg, self.update_nodes_in_dag

    def _wrap_read_dfg(self, gml_path):
        ''' Read the raw gml file, return local DFG
            * Relabel the node name to standard format
            * Handle the mapping from BW->Comm
        '''
        mygraph = nx.read_gml(gml_path)
        dfg_with_var = self.read_dfg_with_var()

        ### Find out Update Operators
        update_nodes_in_dag = set()
        def recursive_add_succs(_node):
            for succ_ in mygraph.successors(_node):
                update_nodes_in_dag.add(succ_)
                recursive_add_succs(succ_)
                if "^"+succ_ in mygraph.nodes:
                    recursive_add_succs("^"+succ_)
        # mygraph.add_edges_from([(n[1:], n) for n in mygraph.nodes if n.startswith("^")])
        for node in mygraph.nodes:
            if ("allreduce" in node.lower() or "bytepspushpull" in node.lower()) \
                    and "switch" not in node.lower() and "input_barrier" not in node.lower():
                ### node is the Comm node, add its downstream nodes as update operators
                raise ValueError("TF 2.4 GraphDef does NOT contain Comm operators")
                recursive_add_succs(node)
            elif node == "GradientDescent" or ("GradientDescent" in node and "update" in node) \
                or node.startswith("Assign_") or ("cond" in node and "Assign" in node):
                update_nodes_in_dag.add(node)
        
        ### TF 2.4, record the mapping from BW to Comm in the local DFG
        edges_to_add = []
        edges_to_rm = []
        for update_op in update_nodes_in_dag:
            for pred in mygraph.predecessors(update_op):
                if tf_relabel_func(pred, update_nodes_in_dag).startswith("BW"):
                    ### Only check those BW->Update edges
                    for variable in dfg_with_var.successors(pred):
                        if update_op in dfg_with_var.successors(variable):
                            ### There is an edge from pred->variable->update_op
                            ### and the variable is a weight variable
                            tensor = "Comm.{}".format(
                                self.tensor_name_to_tensor_id(self.weight2tensor(variable)))
                            edges_to_add.append((pred, tensor))
                            edges_to_add.append((tensor, update_op))
                            edges_to_rm.append((pred, update_op))
        mygraph.add_edges_from(edges_to_add)
        mygraph.remove_edges_from(edges_to_rm)

        ### Re-label Nodes
        new_graph = nx.DiGraph()
        for u, v in mygraph.edges:
            nu, nv = tf_relabel_func(u, update_nodes_in_dag), tf_relabel_func(v, update_nodes_in_dag)
            assert not (nu.startswith("Comm") and nv.startswith("Comm")), (u, v)
            new_graph.add_edge(nu, nv)
        
        self.local_dfg = new_graph
        self.update_nodes_in_dag = update_nodes_in_dag

    def standard_name(self, _name, update_nodes_in_dag=None):
        return tf_relabel_func(_name, update_nodes_in_dag)

    def tensor_id_to_tensor_name(self, _id):
        return self.gradient_name_list(_id)
    
    def tensor_name_to_tensor_id(self, name):
        return self.gradient_name_list.index(name)

    def tensor_id2update_id(self, tensor_id):
        raise NotImplementedError()
    
    def weight2tensor(self, weight_name):
        """Normalizes operation name to TensorFlow rules."""
        tensor_name = re.sub('[^a-zA-Z0-9_]', '_', weight_name)
        if tensor_name not in self.tensor2weight:
            self.tensor2weight[tensor_name] = weight_name
        return tensor_name
