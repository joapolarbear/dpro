import re
from trace_utils import *

class Parameter:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

class ParameterDict:
    def __init__(self, path_):
        self.gradient_name_list = []
        self.para_dict = {}
        raw_para_list = load_list(path_)
        for para_ in raw_para_list:
            ### e.g., bertencoder0_position_weight;shape=(512, 1024);dtype=float16
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
            self.para_dict[name_] = Parameter(name_, shape_, dtype_)
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
