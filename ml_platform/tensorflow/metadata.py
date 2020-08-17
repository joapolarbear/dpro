import numpy as np
import json
from enum import Enum
from ml_platform.tensorflow.amp_lists import whitelist, blacklist, greylist, clearlist

class MetaInfo:
    def __init__(self, tf_meta_path):
        with open(tf_meta_path, 'r') as fp:
            self.tf_meta = json.load(fp)

    def ret_op_type(self, op_name):
        if op_name not in self.tf_meta:
            raise KeyError("{} not in the metadata".format(op_name))
        if "op" not in self.tf_meta[op_name]:
            raise KeyError("op {} has not been given a type".format(op_name))

        ### Special cases
        if "Cast" == self.tf_meta[op_name]["op"] and len(self.tf_meta[op_name]["input"][0]["shape"]) == 0:
            return "vCast"
        return self.tf_meta[op_name]["op"]

    def ret_tf_metadata(self, op_name, batch_size=None):
        '''
        Return:
            S_mul, S_add, S_in, S_out, S_weight
        '''
        if op_name not in self.tf_meta:
            raise KeyError("{} not in the metadata".format(op_name))
        inputs = self.tf_meta[op_name]["input"]
        outputs = self.tf_meta[op_name]["output"]
        op_type = self.tf_meta[op_name]["op"]
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
            if batch_size is None:
                batch_size = N
            return op_type, batch_size*K*P*Q*C*R*S, batch_size*K*P*Q*(C*R*S-1), batch_size*H*W*C, batch_size*P*Q*K, R*S*C*K
        elif op_type == "MatMul":
            assert len(outputs) == 1
            shape_ = outputs[0]["shape"]
            assert len(shape_) == 2
            B = shape_[0]
            C_out = shape_[1]

            assert len(inputs) == 2
            C_in = None
            for input_ in inputs:
                shape_ = input_["shape"]
                assert len(shape_) == 2
                ### decide which is the input, which is the kernel
                if "kernel" in input_["name"] or "ReadVariableOp" in input_["name"]:
                    ### weight
                    assert C_out == shape_[1] or C_out == shape_[0]
                    if C_in is None:
                        C_in = shape_[0] if C_out == shape_[1] else shape_[1]
                    else:
                        assert C_in == (shape_[0] if C_out == shape_[1] else shape_[1])
                else:
                    ### Input
                    assert shape_[0] == B, self.tf_meta[op_name]
                    if C_in is None:
                        C_in = shape_[1]
                    else:
                        assert C_in == shape_[1], (self.tf_meta[op_name], C_in, shape_[1])
            if batch_size is None:
                batch_size = B
            return op_type, batch_size*C_in*C_out, batch_size*(C_in-1)*C_out, batch_size*C_in, batch_size*C_out, C_in*C_out
        elif op_type == "Cast":
            assert len(outputs) == 1
            assert len(inputs) == 1
            if len(inputs[0]["shape"]) == 0:
                return "Cast", None, None, None, None
            if batch_size is not None:
                inputs[0]["shape"][0] = batch_size
                outputs[0]["shape"][0] = batch_size
            dtype_size = self.dtype2size(inputs[0]["dtype"])
            return op_type, 0, 0, np.prod(inputs[0]["shape"])*dtype_size, np.prod(outputs[0]["shape"])*dtype_size, 0
        else:
            raise NotImplementedError("Metadata for {} is not implemented yet.".format(op_name))

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
