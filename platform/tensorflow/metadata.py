import numpy as np
import json
from enum import Enum
from platform.tensorflow.metadata import whitelist, blacklist, greylist, clearlist

class MetaInfo:
    def __init__(self, tf_meta_path):
        with open(tf_graph_path, 'r') as fp:
            self.tf_meta = json.load(fp)

    def ret_tf_metadata(self, op_name):
        '''
        Return:
            S_mul, S_add, S_in, S_out, S_weight
        '''
        inputs = self.tf_meta[op_name]["input"]
        outputs = self.tf_meta[op_name]["output"]
        op_type = self.tf_meta[op_name]["op"]
        if op_type == "Conv2D":
            assert len(outputs) == 1
            shape_ = outputs[0]["shape"]
            assert len(shape_) == 4
            N = shape_[0]
            P = shape_[1]
            Q = shape_[2]
            K = shape_[3]

            assert len(inputs) == 2
            C = None         # input channel
            H = W = None     # Input height/weight
            R = S = None     # kernel size
            for input_ in inputs:
                shape_ = input_["shape"]
                assert len(shape_) == 4
                if "kernel" in input_["name"]:
                    ### weight
                    R, S = shape_[0], shape_[1]
                    if C is None:
                        C = shape_[2]
                    else:
                        assert C == shape_[2]
                    assert K == shape_[3]
                else:
                    ### Input
                    assert shape_[0] == N
                    H, W = shape_[1], shape_[2]
                    if C is None:
                        C = shape_[3]
                    else:
                        assert C == shape_[3]
            return op_type, N*K*P*Q*C*R*S, N*K*P*Q*(C*R*S-1), N*H*W*C, N*P*Q*K, R*S*C*K
        elif op_type == "MatMul":
            assert len(outputs) == 1
            shape_ = outputs[0]["shape"]
            assert len(shape_) == 2
            B = shape_[0]
            C_in = shape_[1]

            assert len(inputs) == 2
            C_out = None
            for input_ in inputs:
                shape_ = input_["shape"]
                assert len(shape_) == 2
                if "kernel" in input_["name"]:
                    ### weight
                    assert C_in == shape_[0]
                    if C_out is None:
                        C_out = shape_[1]
                    else:
                        assert C_out == shape_[1]
                else:
                    ### Input
                    assert shape_[0] == B
                    if C_out is None:
                        C_out = shape_[1]
                    else:
                        assert C_out == shape_[1]
            return op_type, B*C_in*C_out, B*(C_in-1)*C_out, B*C_in, B*C_out, C_in*C_out
        elif op_type == "Cast":
            assert len(outputs) == 1
            assert len(inputs) == 1
            shape_ = outputs[0]["shape"]
            return op_type, 0, 0, np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), 0
        else:
            raise NotImplementedError("{} is not implemented yet.".format(op_name))

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
