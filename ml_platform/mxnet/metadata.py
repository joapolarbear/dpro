import json
import numpy as np
import os, sys
import networkx as nx

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
FULL_HEADERS["conv"] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS['conv']
FULL_HEADERS["dense"] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS['dense']

class GPUInfo:
    def __init__(self, flops32, flops16):
        self.GFLOPS_FP32 = flops32
        self.GFLOPS_FP16 = flops16

### TODO (huhanpeng), how to initialize these GPU info
GPUSINFO = {
    'V100': GPUInfo(1, 2),
    '1080Ti': GPUInfo(1, 2)
}

class MetaInfo:
    def __init__(self, meta_dir, gpu_name=None):
        self.meta_dir = meta_dir
        with open(os.path.join(meta_dir, "metadata.json"), "r") as fp:
            self.mx_meta = json.load(fp)

        self.gpu_name = gpu_name
        self.gpu_info = GPUSINFO[self.gpu_name] if gpu_name is not None else None

        self.cache_hyper_para = {}

        ### dependency graph of this model
        self.dag = self.gen_dag()

        all_output = self.mx_meta["outputs"]
        all_shape = self.mx_meta["out_shapes"]
        assert len(all_output) == len(all_shape)

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
            self.name2shape[std_name] = all_shape[idx]

        self.get_hyper_para()

    def is_ignore(self, node):
        if "Comm" in node:
            return True
        else:
            return False

    def get_hyper_para(self):
        for node in self.name2shape:
            if self.is_ignore(node):
                continue
            op_type = self.parse_op_type(node)
            if op_type == "conv":
                ### outputs
                output_shape = self.name2shape[node]
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

                ### weights
                bp_node = "BW.".join(node.split("FW."))
                wei_node = bias_node = None
                for succ_ in self.dag.successors(bp_node):
                    if "Comm." in succ_:
                        if "wei" in succ_:
                            wei_node = succ_
                        elif "bias" in succ_:
                            bias_node = succ_
                        else:
                            raise ValueError("Conv2D node {} has undefined parameter {}".format(node, succ_))
                if wei_node is None:
                    raise ValueError("No variable/weights for {}".format(node))
                wei_shape = self.name2shape[wei_node]
                # TODO (huhanpeng): still assume the kernel is a square
                if wei_shape[2] == wei_shape[3]:
                    R = S = wei_shape[2]
                else:
                    R = S = wei_shape[0]
                self.cache_hyper_para[node] = [H, W, C, R, S, P, Q, K, N, 0 if bias_node is None else 1]

            elif op_type == "dense":
                ### nexts
                output_shape = self.name2shape[node]
                assert len(output_shape) == 2, (node, output_shape)
                B = output_shape[0]
                C_out = output_shape[1]

                ### prevs
                prevs = self.dag.in_edges(node)
                assert len(prevs) == 1, (node, prevs)
                prev_, _ = list(prevs)[0]
                input_shape = self.name2shape[prev_]
                assert input_shape[0] == B, (node, input_shape, output_shape)
                C_in = input_shape[1]

                ### weights
                ### No need to read weights
                self.cache_hyper_para[node] = [C_in, C_out, B] 

            elif op_type == "cast":
                ### (TODO) 
                continue
                raise NotImplementedError()
                ### nexts
                output_shape = self.name2shape[node]
        
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
                output_shape = self.name2shape[node]
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

    def ret_mx_metadata(self, node, batch_size=None):
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
            C_in, C_out, old_B = self.cache_hyper_para[node]
            return batch_size*C_in*C_out, batch_size*(C_in-1)*C_out, batch_size*C_in, batch_size*C_out, C_in*C_out
        else:
            raise NotImplementedError("Metadata for {} is not implemented yet.".format(node))

    def ret_mx_rawmeta(self, node, batch_size):
        assert node in self.name2shape, "shape info for {} is not in the meta data".format(node)
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

        for i in range(1, len(blocks)):
            prev_block = blocks[i-1]
            var = []
            prev_ls = prev_block.split('\n')
            for l in prev_ls:
                if "Variable" in l:
                    var.append(l.split('Variable:')[1])
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
                    if arg_name not in var:
                        args.append(arg_name)
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
        