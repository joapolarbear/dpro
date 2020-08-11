import numpy as np
import json
from enum import Enum
from ml_platform.tensorflow.amp_lists import whitelist, blacklist, greylist, clearlist

class MetaInfo:
    def __init__(self, tf_meta_path):
        with open(tf_meta_path, 'r') as fp:
            self.tf_meta = json.load(fp)

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
                if "kernel" in input_["name"]:
                    ### weight
                    assert C_out == shape_[1] or C_out == shape_[0]
                    if C_in is None:
                        C_in = shape_[0] if C_out == shape_[1] else shape_[1]
                    else:
                        assert C_in == (shape_[0] if C_out == shape_[1] else shape_[1])
                else:
                    ### Input
                    assert shape_[0] == B
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
            if batch_size is not None:
                inputs[0]["shape"][0] = batch_size
                outputs[0]["shape"][0] = batch_size
            return op_type, 0, 0, np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), 0
        else:
            raise NotImplementedError("Metadata for {} is not implemented yet.".format(op_name))

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
