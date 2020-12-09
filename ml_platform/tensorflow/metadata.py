import numpy as np
import json
import os
from ml_platform.tensorflow.amp_lists import whitelist, blacklist, greylist, clearlist
from logger_utils import Singleton, SingleLogger

OP_HYPER_PARAMETERS = {
    "Conv2D": ["H", "W", "C", "R", "S", "P", "Q", "K", "B", "use_bias"],
    "MatMul": ["C_in", "C_out", "B"],
}
FULL_HEADERS = {"base": ["avg", "G", "S_mul", "S_add", "S_in", "S_out", "S_wei"]}
BASE_HEADER_LEN = len(FULL_HEADERS['base'])
for key in OP_HYPER_PARAMETERS:
    FULL_HEADERS[key] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS[key]

class MetaInfo:
    def __init__(self, meta_dir):
        with open(os.path.join(meta_dir, "metadata.json"), 'r') as fp:
            self.tf_meta = json.load(fp)
        self.cache_hyper_para = {}
        self.get_hyper_para()

    def parse_op_type(self, op_name):
        if op_name not in self.tf_meta:
            raise KeyError("{} not in the metadata".format(op_name))
        if "op" not in self.tf_meta[op_name]:
            raise KeyError("op {} has not been given a type".format(op_name))

        ### Special cases
        if "Cast" == self.tf_meta[op_name]["op"] and len(self.tf_meta[op_name]["input"][0]["shape"]) == 0:
            return "vCast"
        if "control_dependency" in op_name:
            return "Gradient"
        return self.tf_meta[op_name]["op"]

    def ret_tf_metadata(self, op_name, batch_size=None):
        '''
        Return:
            S_mul, S_add, S_in, S_out, S_weight
        '''
        op_type = self.parse_op_type(op_name)
        if op_type == "Conv2D":
            H, W, C, R, S, P, Q, K, old_B, use_bias = self.cache_hyper_para[op_name]
            return batch_size*K*P*Q*C*R*S, batch_size*K*P*Q*(C*R*S-1), batch_size*H*W*C, batch_size*P*Q*K, R*S*C*K
        elif op_type == "MatMul":
            C_in, C_out, old_B = self.cache_hyper_para[op_name]
            return batch_size*C_in*C_out, batch_size*(C_in-1)*C_out, batch_size*C_in, batch_size*C_out, C_in*C_out
        elif op_type == "BW_MatMul":
            C_in, C_out, old_B = self.cache_hyper_para[op_name]
            return batch_size*C_in*C_out, (batch_size-1)*C_in*C_out, batch_size*C_in, C_in*C_out, batch_size*C_out
        elif op_type == "Cast":
            if batch_size is not None:
                inputs[0]["shape"][0] = batch_size
                outputs[0]["shape"][0] = batch_size

            input_size, output_size, dtype_in_size, dtype_out_size, old_B = self.cache_hyper_para[
                op_name]
            return 0, 0, batch_size * input_size / old_B, \
                batch_size * output_size / old_B, 0
        ### above is operator, below is activation, weight and tensors
        elif op_type == "Gradient":
            input_size, output_size, dtype_in_size, dtype_out_size = self.cache_hyper_para[
                op_name]
            return 0, 0, input_size * dtype_in_size, output_size * dtype_out_size, 0

    def ret_tf_rawmeta(self, op_name, batch_size):
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
            elif op_type == "MatMul":
                B = C_in = C_out = None
                assert len(inputs) == 2 and len(
                    inputs[0]["shape"]) == 2 and len(inputs[1]["shape"]) == 2
                found = False
                for i in range(2):
                    if "kernel" in inputs[i]["name"] or "ReadVariableOp" in inputs[i]["name"]:
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
                    B, C_in = inputs[0]["shape"]
                    C_out = inputs[1]["shape"][1]
                    assert inputs[1]["shape"][0] == B, self.tf_meta[op_name]
                self.cache_hyper_para[op_name] = [C_in, C_out, B]
            elif op_type == "Cast":
                assert len(outputs) == 1
                assert len(inputs) == 1 
                dtype_in_size = self.dtype2size(inputs[0]["dtype"])
                dtype_out_size = self.dtype2size(outputs[0]["dtype"])
                self.cache_hyper_para[op_name] = [
                    np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), dtype_in_size, dtype_out_size, batch_size]
            ### above is operator, below is activation, weight and tensors
            elif op_type == "Gradient":
                dtype_in_size = self.dtype2size(inputs[0]["dtype"])
                dtype_out_size = self.dtype2size(outputs[0]["dtype"])
                self.cache_hyper_para[op_name] = [
                    np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), dtype_in_size, dtype_out_size]
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
