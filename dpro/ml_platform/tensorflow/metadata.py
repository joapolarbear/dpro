import numpy as np
import json
import os
import re
import networkx as nx

from .amp_lists import whitelist, blacklist, greylist, clearlist
from ...logger_utils import SingleLogger
from ...trace_utils import FileName, parse_op_name, formal_dpro_rawname
from ...base import bcolors

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

class MetaInfo:
    def __init__(self, meta_dir):
        with open(os.path.join(meta_dir, FileName.METADATA.value), 'r') as fp:
            tmp_txt = fp.read().replace(".", "_")
            self.tf_meta = json.loads(tmp_txt)

        ### Read tensor/gradient names
        try:
            with open(os.path.join(meta_dir, FileName.TENSOR_NAME.value), 'r') as fp:
                info = json.load(fp)
                self.gradient_name_list = info["gradient_name_list"]
                self.model_name = info["model_name"]
                ### Horovod use the same as weight names for tensor names
                self.same_tensor_weight_name = True
        except FileNotFoundError:
            self.model_name = None
            try:
                with open(os.path.join(meta_dir, "gradient_name2ID.json"), 'r') as fp:
                    name2id = json.load(fp)
                    self.gradient_name_list = [name.replace(
                        "HorovodAllreduce.", "") for name, _ in sorted(name2id.items(), key=lambda x: x[1])]
                    ### Horovod does NOT use the same as weight names for tensor names
                    self.same_tensor_weight_name = False
            except FileNotFoundError:
                SingleLogger().error("{} is not found !".format(FileName.TENSOR_NAME.value))
                self.gradient_name_list = []

        ### Batch size used for this meta data file
        self.old_B = None
        self.cache_hyper_para = {}
        ### Tranverse all metadata in advance, cache some importand operaots' metadata
        ### And get self.old_B
        self.get_hyper_para()

        self.local_dfg = None
        self.update_nodes_in_dag = None
        self.bw_nodes_in_dag = None
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

    def find_effective_bw_of_comm(self, comm_op, tensor_name):
        to_process = set(self.local_dfg.predecessors(comm_op))
        second_bws = []
        while len(to_process) > 0:
            bw_op = to_process.pop()
            assert bw_op.startswith("BW"), (comm_op, bw_op)
            bw_op_name = parse_op_name(bw_op)
            if bw_op_name.startswith("gradient_tape") or bw_op_name.startswith("gradients"):
                return bw_op_name, second_bws
            elif bw_op_name.endswith("/tensor"):
                for pre in self.local_dfg.predecessors(bw_op):
                    pred_op_name = parse_op_name(pre)
                    if tensor_name.startswith(pred_op_name):
                        return pred_op_name, second_bws
                raise ValueError(comm_op, bw_op_name, list(self.local_dfg.predecessors(bw_op)))
            else:
                if bw_op_name.startswith("DistributedGradientTape_Allreduce/cond_"):
                    bw_op_name = "/".join(bw_op_name.split("/")[2:])
                    second_bws.append(bw_op_name)
                to_process = to_process.union(
                    set([pre for pre in self.local_dfg.predecessors(bw_op) if "BW" in pre]))
        raise ValueError("Fail to find corresponding bw op for {}".format(comm_op))

    def ret_tensor_size(self, tensor_id):
        tensor_name = self.gradient_name_list[tensor_id]
        weight_name = self.tensor_name2weight_name(tensor_name)
        
        bw_op_name, second_bws = self.find_effective_bw_of_comm("Comm.{}".format(tensor_id), tensor_name)

        for bw_op_name_ in [bw_op_name]+second_bws:
            if bw_op_name_ not in self.tf_meta:
                continue
            outputs = self.tf_meta[bw_op_name_]["output"]
            for output in outputs:
                if output["name"] == weight_name:
                    dtype_size = self.dtype2size(output["dtype"])
                    return np.prod(output["shape"]) * dtype_size

        raise ValueError("Fail to find the weight variable {}, bw_op_name: {} {}, tensor_id: {}".format(
            weight_name, bw_op_name, second_bws, tensor_id))
    
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
            SingleLogger().debug("{} has not been fine-defined input/weight: {}".format(op_name, inputs))
            if len(outputs) == 0:
                S_out = None
            else:
                S_out = wei * np.prod(outputs[0]["shape"])
            if len(inputs) == 0:
                S_in = None
                S_wei = None
            else:
                S_in = wei * np.prod(inputs[0]["shape"])
                S_wei = wei * np.prod(inputs[1]["shape"]) if len(inputs) > 1 else 1
            return 0, 0, S_in, S_out, S_wei

    def ret_rawmeta(self, op_name):
        op_type = self.parse_op_type(op_name)
        if op_type == "CastToFp16" or op_type == "CastToFp32":
            return []
        assert op_name in self.tf_meta, "shape info for {} is not in the meta data".format(
            op_name)
        return self.cache_hyper_para.get(op_name, None)

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
                var_name = "VAR." + _input["name"]
                edges_to_add.append((var_name, op))
            for _output in self.tf_meta[op]["output"]:
                ### _output is a dict
                var_name = "VAR." + _output["name"]
                edges_to_add.append((op, var_name))
        g.add_edges_from(edges_to_add)
        return g
    
    def remove_last_trival_slash(self, _name):
        last_slash_pos = _name.rfind("/")
        if last_slash_pos != -1 and last_slash_pos < len(_name)-1 and _name[last_slash_pos+1] == "_":
            _name = _name[:last_slash_pos]
        return _name

    def tf_relabel_func(self, _name):
        for prefix in ["Comm.", "Comp.", "BW.", "FW.", "UPDATE_."]:
            if _name.startswith(prefix):
                return _name
        if _name.startswith("^"):
            _name = _name[1:]

        _name = self.remove_last_trival_slash(_name)
        
        bps_tensor_match = re.search("BytePSPushPull[_.]([\d_]+)$", _name)
        if bps_tensor_match:
            _name = "Comm." + bps_tensor_match.group(1).replace("_", "+")
        elif "input_barrier" in _name:
            _name = "FW." + formal_dpro_rawname(_name)
        elif "allreduce" in _name.lower():
            if "." in _name:
                _, tensor_name = _name.split(".")
                if "_" in tensor_name:
                    tensor_name = tensor_name.split("_")[0]
                if "Switch" in _name:
                    _name = "BW." + tensor_name + "_Switch"
                else:
                    _name = "Comm." + tensor_name
            elif self.update_nodes_in_dag is not None and _name in self.update_nodes_in_dag:
                _name = "UPDATE_." + formal_dpro_rawname(_name)
            else:
                _name = "BW." + formal_dpro_rawname(_name)
            # elif "cond_" in _name and "Switch" in _name:
            #     _name = "BW." + _name
            # elif "Const" in _name or "cond_" in _name:
            #     _name = "BW." + _name
            # else:
            #     _name = "UPDATE_." + _name
        else:
            if self.update_nodes_in_dag is not None and _name in self.update_nodes_in_dag:
                _name = "UPDATE_." + _name
            elif self.bw_nodes_in_dag is not None and _name in self.bw_nodes_in_dag:
                _name = "BW." + formal_dpro_rawname(_name)
            elif _name.startswith("gradients") or \
                (_name.startswith("gradient_tape") and not _name.endswith("ShapeN") \
                    and not "LayoutOptimizer" in _name):
                _name = "BW." + formal_dpro_rawname(_name)
            else:
                _name = "FW." + formal_dpro_rawname(_name)
        return _name

    def wrap_read_dfg(self, gml_path):
        if self.local_dfg is None:
            self._wrap_read_dfg(gml_path)
        
        return self.local_dfg
    
    def delete_node_from_dfg(self, old_node, dfg,
            edges_to_add=None,
            edges_to_rm=None,
            nodes_to_rm = None,
            new_node=None,
            to_fitler_fn=None):
        ### Delete `old_node` from dfg, not inplace, only calculate edges to add or remove
        ### *NOTE*, if `new_node` is given, `new_node` will replace `old_node`
        
        nodes_to_rm.append(old_node)

        pred_ops = set()
        to_process = set(dfg.predecessors(old_node))
        edges_to_rm += [(pred, old_node) for pred in to_process]
        while len(to_process) > 0:
            pred = to_process.pop()
            if to_fitler_fn is None or not to_fitler_fn(pred):
                pred_ops.add(pred)
            else:
                nodes_to_rm.append(pred)
                for _pred in dfg.predecessors(pred):
                    to_process.add(_pred)
                    edges_to_rm.append((_pred, pred))
        
        succ_ops = set()
        to_process = set(dfg.successors(old_node))
        edges_to_rm += [(old_node, succ) for succ in to_process]
        while len(to_process) > 0:
            succ = to_process.pop()
            if to_fitler_fn is None or not to_fitler_fn(succ):
                succ_ops.add(succ)
            else:
                nodes_to_rm.append(succ)
                for _succ in dfg.successors(succ):
                    to_process.add(_succ)
                    edges_to_rm.append((succ, _succ))

        for pred in pred_ops:
            for succ in succ_ops:
                if new_node is None:
                    edges_to_add.append((pred, succ))
                else:
                    edges_to_add += [(pred, new_node), (new_node, succ)]

    def _wrap_read_dfg(self, gml_path):
        ''' Read the raw gml file, return local DFG
            * Relabel the node name to standard format
            * Handle the mapping from BW->Comm
        '''
        graphdef_dag_path = os.environ.get("DPRO_GRAPHDEF_DFG_PATH", None)
        if graphdef_dag_path:
            if not os.path.exists(graphdef_dag_path):
                dumped_hlo_graph_path = os.path.join(os.path.dirname(graphdef_dag_path), "before_mark_for_compilation.pbtxt")
                cur_dir, _ = os.path.split(os.path.realpath(__file__))
                cmd = "python3 {}/util.py {}".format(cur_dir, dumped_hlo_graph_path)
                os.system(cmd)
            mygraph = nx.read_gml(graphdef_dag_path)
            SingleLogger().info(bcolors.CGREEN + 
                "[TF Metadata] read DFG parsed from graphdef: {}".format(graphdef_dag_path) + bcolors.ENDC)
        else:
            # mygraph = nx.read_gml(gml_path)
            dfg_with_var = self.read_dfg_with_var()
            mygraph = dfg_with_var
            ### Mark comm operators
            edges_to_add = []
            edges_to_rm = []
            nodes_to_rm = []
            FORCE_DELETE_VAR = True
            for op_or_var in dfg_with_var.nodes:
                if not op_or_var.startswith("VAR."):
                    continue

                if FORCE_DELETE_VAR:
                    self.delete_node_from_dfg(op_or_var, dfg_with_var, edges_to_add, edges_to_rm, nodes_to_rm,
                        to_fitler_fn=lambda x: x.startswith("VAR."))
                    continue
                ### weights or activations
                var_name = op_or_var.strip("VAR.")
                if var_name in self.gradient_name_list:
                    ### Commnunication
                    keep_var = True
                    for succ in dfg_with_var.successors(op_or_var):
                        if self.tf_relabel_func(succ).startswith("Comm."):
                            keep_var = False
                            break
                    if keep_var:
                        ### bw_op --> bw_op:0 --> update_op
                        comm_op = "Comm.{}".format(self.tensor_name_to_tensor_id(var_name))
                        self.delete_node_from_dfg(
                            op_or_var, dfg_with_var, edges_to_add, edges_to_rm, nodes_to_rm,
                            new_node=comm_op)
                    else:
                        ### bw_op --> bw_op:0 --> comm_op
                        self.delete_node_from_dfg(op_or_var, dfg_with_var, edges_to_add, edges_to_rm, nodes_to_rm)
                else:
                    ### Activations
                    self.delete_node_from_dfg(op_or_var, dfg_with_var, edges_to_add, edges_to_rm, nodes_to_rm)
            mygraph.add_edges_from(edges_to_add)
            mygraph.remove_edges_from(edges_to_rm)
            mygraph.remove_nodes_from(nodes_to_rm)

            # for node in mygraph.nodes:
            #     if node.startswith("VAR."):
            #         print(node)
            #         print(list(mygraph.predecessors(node)))
            #         print(list(mygraph.successors(node)))
            #         raise
            
            SingleLogger().info(bcolors.CGREEN + 
                "[TF Metadata] read DFG: {}".format(gml_path) + bcolors.ENDC)

        ### Find out Update Operators
        update_nodes_in_dag = set()
        def recursive_add_succs(_node):
            for succ_ in mygraph.successors(_node):
                update_nodes_in_dag.add(self.remove_last_trival_slash(succ_))
                recursive_add_succs(succ_)
                if "^"+succ_ in mygraph.nodes:
                    recursive_add_succs("^"+succ_)
        # mygraph.add_edges_from([(n[1:], n) for n in mygraph.nodes if n.startswith("^")])
        for node in mygraph.nodes:
            # if ("allreduce" in node.lower() or "bytepspushpull" in node.lower()) \
            #         and "switch" not in node.lower() and "input_barrier" not in node.lower():
            #     ### node is the Comm node, add its downstream nodes as update operators
            #     # raise ValueError("TF 2.4 GraphDef does NOT contain Comm operators")
            #     if "." in node and node.split(".")[1].isdigit():
            if self.tf_relabel_func(node).startswith("Comm."):
                recursive_add_succs(node)
            elif node == "GradientDescent" or ("GradientDescent" in node and "update" in node) \
                    or node.startswith("Assign_") or ("cond" in node and "Assign" in node) \
                    or "SGD" in node or "Assign" in node:
                update_nodes_in_dag.add(self.remove_last_trival_slash(node))
        
        self.update_nodes_in_dag = update_nodes_in_dag

        ### Re-label Nodes
        new_graph = nx.DiGraph()
        for u, v in mygraph.edges:
            nu, nv = self.tf_relabel_func(u), self.tf_relabel_func(v)
            assert not (nu.startswith("Comm") and nv.startswith("Comm")), (
                u, v, list(mygraph.successors(u)), list(mygraph.successors(v)), 
                list(mygraph.predecessors(u)), list(mygraph.predecessors(v)))
            new_graph.add_edge(nu, nv)
        
        ### Relabel operators, all downstream operators of BW OPs should be BW OPs
        relabel_map = {}
        self.bw_nodes_in_dag = set()
        visited_nodes = set()
        def recur_update_fw2bw(bw_op):
            if bw_op in visited_nodes:
                return
            visited_nodes.add(bw_op)
            for succ in new_graph.successors(bw_op):
                if "FW" in succ:
                    # print(bw_op, succ)
                    relabel_map[succ] = succ.replace("FW", "BW")
                    self.bw_nodes_in_dag.add(parse_op_name(succ))
                    recur_update_fw2bw(succ)
                elif "Comm" in succ:
                    pass
                elif "UPDATE_" in succ:
                    pass
                else:
                    ### "BW" in succ
                    recur_update_fw2bw(succ)
        for _op in new_graph.nodes():
            if "BW" in _op:
                self.bw_nodes_in_dag.add(parse_op_name(_op))
                recur_update_fw2bw(_op)
        # if "bert" not in self.model_name:
        nx.relabel_nodes(new_graph, relabel_map, copy=False)

        self.local_dfg = new_graph
        nx.write_gml(new_graph, "tmp.gml")

    def standard_name(self, op_name):
        assert self.update_nodes_in_dag is not None and self.bw_nodes_in_dag is not None
        return self.tf_relabel_func(op_name)

    def tensor_id_to_tensor_name(self, _id):
        return self.gradient_name_list(_id)
    
    def tensor_name_to_tensor_id(self, name):
        return self.gradient_name_list.index(name)

    def tensor_id2update_id(self, tensor_id):
        raise NotImplementedError()
    
    def weight_name2tensor_name(self, weight_name):
        if self.same_tensor_weight_name:
            return weight_name
        """Normalizes operation name to TensorFlow rules."""
        tensor_name = re.sub('[^a-zA-Z0-9_]', '_', weight_name)
        if tensor_name not in self.tensor2weight:
            self.tensor2weight[tensor_name] = weight_name
        return tensor_name
    
    def tensor_name2weight_name(self, tensor_name):
        if self.same_tensor_weight_name:
            return tensor_name
        return self.tensor2weight[tensor_name]
    
    def parse_model_name(self):
        return self.model_name
