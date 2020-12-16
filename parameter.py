''' Manage the parameter info of a DNN model
### TODO (huhanpeng): mitigrate to ml_platform/mxnet/metadata.py
'''
import re
from trace_utils import *

class Parameter:
    def __init__(self, index, name, shape, dtype):
        self.index = index
        self.name = name
        self.shape = shape
        self.dtype = dtype

class ParameterDict:
    def __init__(self, raw_para_list):
        self.gradient_name_list = []
        self.parameters = []
        ### name2layeridx is used to map op_name to index to reduce op_name length
        ### TODO (huhanpeng), but sometimes, computation names can not be parsed from communication names
        ### The correct method should be using DAG
        self.name2layeridx = {}
        self.total_idx = 0
        for para_ in raw_para_list:
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
            self.parameters.append(Parameter(self.total_idx, name_, shape_, dtype_))
            self.total_idx += 1
            layer_name = parse_layer_name(name_)
            if layer_name not in self.name2layeridx:
                self.name2layeridx[layer_name] = self.total_idx
                self.total_idx += 1
        self.cnt = len(self.gradient_name_list)

        self.tensor2update = {}

    def map_tensors_to_update(self, aggregate_num=0):
        ''' Map each tensor id to its corresponding update operation
        For MXNet
        '''
        max_update_id = 0
        for idx in range(self.cnt):
            gra_idx = self.cnt - 1 - idx
            self.tensor2update[gra_idx] = idx if aggregate_num == 0 else int(idx / aggregate_num)
            max_update_id = max(max_update_id, self.tensor2update[gra_idx])
        self.tensor2update["max"] = max_update_id
        return self.tensor2update

    def name_to_tensor_id(self, name):
        return self.gradient_name_list.index(name)

    def tensor_id_to_name(self, tensor_id):
        return self.gradient_name_list[tensor_id]

    def parse_layer_index(self, layer_name):
        try:
            return self.name2layeridx[layer_name]
        except KeyError:
            ### Some layers do not have parameters, still need to register these nodes
            self.name2layeridx[layer_name] = self.total_idx
            self.total_idx += 1
            return self.total_idx - 1

    def index2layer_name(self, index):
        for name, idx in self.name2layeridx.items():
            if idx == index:
                return name
        return "Not found"
