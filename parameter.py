''' Manage the parameter info of a DNN model
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
        self.name2para = {}
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
            self.name2para[name_] = Parameter(self.total_idx, name_, shape_, dtype_)
            self.total_idx += 1
            layer_name = parse_layer_name(name_)
            if layer_name not in self.name2layeridx:
                self.name2layeridx[layer_name] = self.total_idx
                self.total_idx += 1
        self.cnt = len(self.gradient_name_list)

        self.tensor2update = {}

    def map_tensors_to_update(self, aggregate_num=0):
        ''' Map each tensor to its corresponding update operation
        For MXNet
        '''
        max_update_id = 0
        for idx in range(self.cnt):
            gra = self.gradient_name_list[self.cnt - 1 - idx]
            self.tensor2update[gra] = idx if aggregate_num == 0 else int(idx / aggregate_num)
            max_update_id = max(max_update_id, self.tensor2update[gra])
        self.tensor2update["max"] = max_update_id
        return self.tensor2update

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
