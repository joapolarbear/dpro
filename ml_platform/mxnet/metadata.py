import json
import numpy as np
import os, sys
import re
import networkx as nx

# (TODO): delete, 20201026
# FULL_HEADERS = {
#     "base": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei"],
#     "conv": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei", "H", "W", "C", "R", "S", "P", "Q", "K", "B", "use_bias"],
#     "dense": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei", "C_in", "C_out", "B"],
# }

OP_HYPER_PARAMETERS = {
    "conv": ["H", "W", "C", "R", "S", "P", "Q", "K", "B", "use_bias"],
    "dense": ["C_in", "C_out", "B"],
}

FULL_HEADERS = {"base": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei"]}
BASE_HEADER_LEN = len(FULL_HEADERS['base'])
for key in OP_HYPER_PARAMETERS:
    FULL_HEADERS[key] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS[key]

class GPUInfo:
    def __init__(self, flops32, flops16):
        self.GFLOPS_FP32 = flops32
        self.GFLOPS_FP16 = flops16

### TODO (huhanpeng), how to initialize these GPU info
GPUSINFO = {
    'V100': GPUInfo(1, 2),
    '1080Ti': GPUInfo(1, 2)
}

### TODO (urgent), replace this
GFLOPS_FP32 = 1
GFLOPS_FP16 = 2

class Parameter:
    def __init__(self, index, name, shape, dtype):
        self.index = index
        self.name = name
        self.shape = shape
        self.dtype = dtype

class MetaInfo:
    def __init__(self, meta_dir, gpu_name=None):
        self.meta_dir = meta_dir
        with open(os.path.join(meta_dir, "metadata.json"), "r") as fp:
            self.mx_meta = json.load(fp)
        all_output = self.mx_meta["outputs"]
        all_shape = self.mx_meta["out_shapes"]
        assert len(all_output) == len(all_shape)

        self.gpu_name = gpu_name
        self.gpu_info = GPUSINFO[self.gpu_name] if gpu_name is not None else None   

        ### dependency graph of this model
        self.dag = self.gen_dag()
        # nx.write_gml(self.dag, os.path.join(
        #     "/Users/bytedance/0/data/20201126_01_2W2G_rdma_hvd_mx_bert_hvd_input_data_shape", "dag.gml"), lambda x: str(x))

        ### init name2shape dict, convert name to std. name
        self.name2shape = {}
        for idx, out_ in enumerate(all_output):
            if "data" in out_:
                std_name = out_.replace("data", "I/O_")
            elif "_output" in out_:
                ### activations
                raw_name = out_.split("_output")[0]
                std_name = "FW." + (raw_name.split("_fwd")[0] if "_fwd" in raw_name else raw_name)   
            else:
                ### weights or variables
                std_name = "Comm." + out_
            if std_name in self.name2shape:
                print("[WARNING] {} has multiple outputs, assume to use the first output as the input of the next op".format(std_name))
            else:
                self.name2shape[std_name] = all_shape[idx]

        self.cache_hyper_para = {}
        self.get_hyper_para()
        self.gen_grad()

        ### Map tensor name to its update index
        if "opt_aggregate_num" in self.mx_meta:
            aggregate_num = self.mx_meta["opt_aggregate_num"]
        else:
            aggregate_num = 0
        self.tensor2update = {}
        self.map_tensors_to_update(aggregate_num)
        
    def map_tensors_to_update(self, aggregate_num=0):
        ''' Map each tensor id to its corresponding update operation
        For MXNet
        '''
        max_update_id = 0
        grad_cnt = len(self.gradient_name_list)
        for idx in range(grad_cnt):
            gra_idx = grad_cnt - 1 - idx
            self.tensor2update[gra_idx] = idx if aggregate_num == 0 else int(idx / aggregate_num)
            max_update_id = max(max_update_id, self.tensor2update[gra_idx])
        self.tensor2update["max"] = max_update_id

    def gen_grad(self):
        self.gradient_name_list = []
        self.parameters = []
        for idx, para_ in enumerate(self.mx_meta["gradient_name_list"]):
            ### e.g., bertencoder0_position_weight;shape=(512, 1024);dtype=float16
            if isinstance(para_, list):
                para_split = para_
            else:
                para_split = para_.split(";")
            name_ = para_split[0]
            if len(para_split) == 1:
                ### No shape and dtype info are provided
                shape_ = None
                dtype_ = None
            else:
                shape_ = [int(e) for e in re.findall(r"\d+", para_split[1])]
                dtype_ = para_split[2].split("dtype=")[1]
            self.gradient_name_list.append(name_)
            self.parameters.append(Parameter(idx, name_, shape_, dtype_))

    def is_ignore(self, node):
        if "Comm" in node:
            return True
        else:
            return False

    def get_wei_shape(self, node):
        ### weights
        bp_node = "BW.".join(node.split("FW.")) if "FW" in node else node 
        wei_node = bias_node = None
        for succ_ in self.dag.successors(bp_node):
            if "Comm." in succ_:
                if "wei" in succ_:
                    wei_node = succ_
                elif "bias" in succ_:
                    bias_node = succ_
                else:
                    raise ValueError(
                        "Conv2D node {} has undefined parameter {}".format(node, succ_))
        if wei_node is None:
            raise ValueError("No variable/weights for {}".format(node))
        return self.name2shape[wei_node], bias_node
    
    def get_hyper_para(self):
        for node in self.name2shape:
            if self.is_ignore(node):
                continue
            op_type = self.parse_op_type(node)
            output_shape = self.name2shape[node]
            if op_type == "conv":
                ### outputs
                assert len(output_shape) == 4, (node, output_shape)
                N = output_shape[0]
                ### TODO (huhanpeng): assume the width=height, the same for input shape
                P = Q = output_shape[2]
                ### different layout, NHWC --> shape[3] or NCHW --> shape[1]
                K = output_shape[3] if output_shape[1] == P else output_shape[1]

                ### inputs
                prevs = self.dag.in_edges(node)
                assert len(prevs) == 1, (node, prevs)
                prev_, _ = list(prevs)[0]
                input_shape = self.name2shape[prev_]
                assert input_shape[0] == N, (node, input_shape, output_shape)
                H = W = input_shape[2]
                C = input_shape[3] if input_shape[1] == H else input_shape[1]

                wei_shape, bias_node = self.get_wei_shape(node)
                # TODO (huhanpeng): still assume the kernel is a square
                if wei_shape[2] == wei_shape[3]:
                    R = S = wei_shape[2]
                else:
                    R = S = wei_shape[0]
                self.cache_hyper_para[node] = [H, W, C, R, S, P, Q, K, N, 0 if bias_node is None else 1]

            elif op_type == "dense":
                third_d = 1
                B = output_shape[0]
                C_out = output_shape[1]
                ### prevs
                prevs = self.dag.in_edges(node)
                if len(prevs) == 1:
                    prev_, _ = list(prevs)[0]
                    input_shape = self.name2shape[prev_]
                    if len(input_shape) == 2 or len(input_shape) == 4:
                        assert input_shape[0] == B, (node, input_shape, output_shape)
                        C_in = input_shape[1]   
                    elif len(input_shape) == 3:
                        weight_shape, _ = self.get_wei_shape(node)
                        C_in, C_out = weight_shape
                        if C_in == input_shape[0] * input_shape[2]:
                            assert input_shape[1] == B, (node, input_shape, output_shape)
                        elif C_in == input_shape[2]:
                            third_d = input_shape[1]
                            assert third_d == output_shape[1] and B == input_shape[0], (
                                node, input_shape, output_shape)
                    else:
                        raise ValueError(node, input_shape, output_shape)
                else:
                    input_shapes = [self.name2shape[prev_] for prev_, _ in prevs]
                    print(self.get_wei_shape(node), input_shapes)
                    raise
                self.cache_hyper_para[node] = [C_in, C_out, B, third_d] 
            elif op_type == 'lstm_param_concat':
                pass
            elif op_type == 'lstm_rnn':
                pass
            elif op_type == 'lstm_reshape':
                pass
            elif op_type == "cast":
                ### (TODO) 
                continue
                raise NotImplementedError()
        
                ### prevs
                prevs = self.dag.in_edges(node)
                assert len(prevs) == 1, prevs
                prev_, _ = list(prevs)[0]
                input_shape = self.name2shape[prev_]
                dtype_size = self.dtype2size(inputs[0]["dtype"])

                self.cache_meta[node] = (op_type, 0, 0, np.prod(inputs[0]["shape"])*dtype_size, np.prod(outputs[0]["shape"])*dtype_size, 0)

            elif op_type == "embedding":
                ### (TODO) 
                continue
                raise NotImplementedError()
                output_size = np.prod(output_shape)
                if len(output_shape) == 2:
                    ### no batch size
                    B = None          
                elif len(output_shape) == 3:
                    B = output_shape[0]
                    if batch_size is not None:
                        output_size = output_size * batch_size / B
                else:
                    raise

                ### prevs
                prevs = self.dag.in_edges(node)
                assert len(prevs) == 1, prevs
                prev_, _ = list(prevs)[0]
                input_shape = self.name2shape[prev_]
                input_size = np.prod(input_shape)
                if B is not None and batch_size is not None:
                    input_size = input_size * batch_size / B

                bp_node = "BW.".join(node.split("FW."))
                comm_node = []
                for succ_ in self.dag.successors(bp_node):
                    if "Comm." in succ_:
                        comm_node.append(succ_)
                if comm_node is None:
                    raise ValueError("No variable/weights for {}".format(node))
                wei_shape = [self.name2shape[e] for e in comm_node][0]
                wei_size = np.prod(wei_shape)

                # print(input_shape, output_shape, wei_shape)

                self.cache_meta[node] = (op_type, input_size*wei_size, 0, input_size, output_size, wei_size) 


            else:
                # raise NotImplementedError("Metadata for {} is not implemented yet.".format(node))
                ### (TODO) should raise error
                pass

    def ret_metadata(self, node, batch_size=None):
        '''
        node: node name in the dag
        '''
        assert node in self.name2shape, "shape info for {} is not in the meta data".format(node)

        if "FW" not in node:
            ### only consider FW node
            return
        op_type = self.parse_op_type(node)

        if op_type == "conv":
            H, W, C, R, S, P, Q, K, old_B, use_bias = self.cache_hyper_para[node]
            return batch_size*K*P*Q*C*R*S, batch_size*K*P*Q*(C*R*S-1), batch_size*H*W*C, batch_size*P*Q*K, R*S*C*K

        elif op_type == "dense":
            C_in, C_out, old_B, third_d = self.cache_hyper_para[node]
            return batch_size*C_in*C_out*third_d, batch_size*(C_in-1)*C_out*third_d, batch_size*C_in*third_d, batch_size*C_out*third_d, C_in*C_out
        else:
            raise NotImplementedError("Metadata for {} is not implemented yet.".format(node))

    def ret_rawmeta(self, node):
        assert node in self.name2shape, "shape info for {} is not in the meta data".format(node)
        if op_type == "dense":
            ### last dimension third_d, considering some special cases, 3-D matrix multiplication
            return self.cache_hyper_para[node][:-1]
        else:
            return self.cache_hyper_para[node]

    def parse_op_type(self, op_name):
        op_name = op_name.lower()
        if "conv" in op_name:
            return "conv"
        elif "_dense" in op_name and 'cast' not in op_name:
            return "dense"
        elif "cast" in op_name:
            return "cast"
        elif "embedding" in op_name:
            return "embedding"
        elif "lstm" in op_name:
            if "param_concat" in op_name:
                return "lstm_param_concat"
            elif "_rnn" in op_name:
                return "lstm_rnn"
            else:
                return "lstm_reshape"
        else:
            # raise ValueError("Undefined op type for {}".format(op_name))
            return "undefined"
    
    def gen_dag(self, _main=False):
        """Construct a DAG from the mxnet info

        Parameters:
        ----------
        s : str
            Must follow the standard chrome trace format and not None.
        """
        with open(os.path.join(self.meta_dir, "symbol_debug_str.txt"), "r") as fp:
            s = fp.read()
        _dag = nx.DiGraph()
        blocks = s.split("--------------------\n")
        
        #! 3. FW -> OUTPUT and 4. OUTPUT -> BW
        first_ls = blocks[0].split('\n')
        output_cnt = 0
        for i in range(len(first_ls)):
            if "Variable:" in first_ls[i]:
                break
            if "output[" in first_ls[i]:
                output_node = first_ls[i].split(']=')[1].split('(')[0]
                output_node = output_node.split("_fwd")[0] if "_fwd" in output_node else output_node
                _dag.add_edge("FW." + output_node, "OUTPUT%d"%output_cnt)
                _dag.add_edge("OUTPUT%d"%output_cnt, "BW." + output_node)
                output_cnt += 1
        all_tensor = set()
        for i in range(1, len(blocks)):
            prev_block = blocks[i-1]
            var = []
            prev_ls = prev_block.split('\n')
            for l in prev_ls:
                if "Variable" in l:
                    tensor_name = l.split('Variable:')[1]
                    var.append(tensor_name)
                    all_tensor.add(tensor_name)
            block = blocks[i]
            ls = block.split('\n')
            if 'Name' not in ls[0]:
                continue
            name = ls[0].split('Name=')[1]
            op = ls[0].split(',')[0].split("Op:")[1]
            args = []
            for l in ls:
                if "arg[" in l:
                    arg_name = l.split(']=')[1].split('(')[0]
                    if arg_name not in all_tensor:
                        args.append(arg_name)
                    elif arg_name not in var:
                        ### in cases on weight is used for multiple times
                        var.append(arg_name)
            if "_fwd" in name:
                name = name.split("_fwd")[0]

            #! --------- construct the graph ----
            _dag.add_node("FW." + name, op=op)
            _dag.add_node("BW." + name, op=op)
            for innode in args:
                innode = innode.split("_fwd")[0] if "_fwd" in innode else innode
                #! 2. FW -> FW and 5. BW -> BW
                _dag.add_edge("FW." + innode, "FW." + name)
                _dag.add_edge("BW." + name, "BW." + innode)
            for _var in var:
                if "data" in _var:
                    _dag.add_edge(_var.replace("data", "I/O_"), "FW." + name)
                    if _main:
                        #! 1. IO -> FW, 8. BW -> UPDATE -> FW                  
                        _dag.add_edge("BW." + name, "UPDATE")
                        _dag.add_edge("UPDATE", "FW." + name)
                else:
                    #! 7. Comm -> FW and 6. BW -> Comm
                    _dag.add_edge("Comm." + _var, "UPDATE")
                    _dag.add_edge("BW." + name, "Comm." + _var)
        return _dag
    
    def ret_tensor_size(self, tensor_id):
        tensor_name = self.gradient_name_list[tensor_id]
        shape_sum = np.prod(self.name2shape["Comm." + tensor_name])
        dtype_size = self.dtype2size(self.parameters[tensor_id].dtype)
        return shape_sum * dtype_size

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

    def wrap_read_dfg(self, gml_path):
        mygraph = nx.read_gml(gml_path)
        update_nodes_in_dag = None
        return mygraph, update_nodes_in_dag

    def standard_name(self, _name):
        #! add for mxnet-gluon case
        if "name=" in _name:
            _name = _name.split("name=")[1].split(";")[0]
        #! backward nodes or forward nodes
        _name = "BW." + _name.split("_backward")[0] if "_backward" in _name else "FW." + _name
        _name = _name.split("_fwd")[0] if "_fwd" in _name else _name
        return _name

    def tensor_id_to_tensor_name(self, _id):
        return self.gradient_name_list[_id]
    
    def tensor_name_to_tensor_id(self, name):
        return self.gradient_name_list.index(name)
    
    def tensor_id2update_id(self, tensor_id):
        '''tensor id may be 'max' to return the maximum update id '''
        return self.tensor2update[tensor_id]
