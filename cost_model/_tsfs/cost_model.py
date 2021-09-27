import numpy as np

def piecewise_linear_3seg(x, x0, y0, x1, y1, k2):
    return np.piecewise(x, [x <= x0, x > x1], 
        [
            lambda x: y0,
            lambda x: (x - x1) + y1,
            lambda x: k2 * (x - x0) + y0,])
p0_3seg = (1, 0, 6, 0, 1)

def piecewise_linear_2seg(x, x0, y0):
    return np.piecewise(x, [x <= x0], 
        [
            lambda x: y0,
            lambda x: (x - x0) + y0,])
p0_2seg = (6, 0)


class DataRepo:
    def __init__(self, tensor_time):
        self.para_2seg = None
        self.para_3seg = None
        self.tensor_time = tensor_time
    
    def dumps(self):
        print("2 seg: ", self.array_str(self.para_2seg))
        print("3 seg: ", self.array_str(self.para_3seg))
    
    def array_str(self, a):
        return "[" + ", ".join([str(n) for n in a]) + "]"

def wrap_predict(func, para, xdata):
    pred_ydata = func(np.log10(xdata), *para)
    return np.power(10, pred_ydata)
    # pred_ydata = func(xdata, *para)
    # return pred_ydata

def test_accuracy(func, para, xdata, ydata):
    pred_ydata = wrap_predict(func, para, xdata)
    mape = np.average(np.abs(pred_ydata - ydata) / ydata)
    return mape

### TCP
intra_2GPU_para = DataRepo(None)
intra_2GPU_para.para_2seg = [6.478717760741668, -0.7911850258660735]
intra_2GPU_para.para_3seg = [5.768569837527714, -0.8112763281978731, 7.378590861143234, 0.07736945356154445, 0.4601007391482461]
inter_100Gb_para = DataRepo(None)
inter_100Gb_para.para_2seg = [5.72967574893935, 0.27409744017295945]
inter_100Gb_para.para_3seg = [5.481425042939888, 0.24998168803732868, 523.1069698319661, 517.6116145143503, 0.8976445312387689]

### 20210909 profile PUSH and PULL standalone in 1wk*1gpu 1server
push_data = DataRepo(None)
push_data.para_2seg = [4.686307490183, -1.662961882088019]
push_data.para_3seg = [4.846827061369098, -1.6483260907019037, 626.2712890335985, 619.9568948850784, 1.1001192383975844]
pull_data = DataRepo(None)
pull_data.para_2seg = [4.803492695527605, -1.5562480802345402]
pull_data.para_3seg = [4.961341192845001, -1.5523328848981286, 626.2723641952061, 619.9558183092073, 1.119712390211427]

### 20210916 profile PUSH and PULL standalone in 2wk*8gpu 2server
push_data = DataRepo(None) 
push_data.para_2seg = [4.659495844497468, -1.7521176854189915]
push_data.para_3seg =  [4.70781790174102, -1.7521176854008658, 626.2991919900213, 619.9289905166377, 1.0343874361266248]
pull_data = DataRepo(None)
pull_data.para_2seg = [4.671509439964712, -1.6513319055098747]
pull_data.para_3seg = [4.7581024691199625, -1.6513319055201428, 626.2641292852817, 619.9640547461936, 1.063909142047732]

### 20210926 profile PUSH and PULL in a completed ResNet50 model
# push_data = DataRepo(None) 
# push_data.para_3seg =  [1.394117768140235, -2.7347276537801393, 6.265639039503503, 0.12958045808905536, 0.7140396422668894]
# pull_data = DataRepo(None)
# pull_data.para_3seg = push_data.para_3seg
# # pull_data.para_3seg = [2.866912655486207, -2.4697033402682265, 4.575557144977073, -1.5755571450053836, 1.3329606768736635]

# push_data = DataRepo(None) 
# push_data.para_3seg =  [2.403217615362119, -1.548809007772742, 9.522663714692643, 3.443890309353957, 0.5209623472565877]
# pull_data = DataRepo(None)
# pull_data.para_3seg = push_data.para_3seg

import json
# data_table_path = "/home/tiger/sub_op2tensor_size2avg.json"
# data_table_path = "/home/tiger/sub_op2tensor_size2avg_tcp_vgg16.json"
data_table_path = "/home/tiger/sub_op2tensor_size2avg_tcp_icptv3.json"
with open(data_table_path, 'r') as fp:
    data_table = json.load(fp)
def interpolation(tensor_size, tensor_size2avg):
    tensor_size_list = list(tensor_size2avg.keys())
    available_tensor_size = sorted(
        enumerate([float(key) for key in tensor_size_list]),
        key=lambda x: x[1])
    i = 0
    while i < len(available_tensor_size):
        if tensor_size < available_tensor_size[i][1]:
            break
        i += 1
    if i == 0:
        i = 1
        print("[TSFS CM] warning", (tensor_size, i, available_tensor_size[:5]))
    elif i == len(available_tensor_size):
        i = len(available_tensor_size) - 1
        print("[TSFS CM] warning", (tensor_size, i, available_tensor_size[:5]))
    x1, x2 = available_tensor_size[i-1][1], available_tensor_size[i][1]
    y1 = tensor_size2avg[tensor_size_list[available_tensor_size[i-1][0]]]
    y2 = tensor_size2avg[tensor_size_list[available_tensor_size[i][0]]]
    return ((y1 - y2) / (x1 - x2)) * (tensor_size - x1) + y1

piecewise_linear_func = piecewise_linear_3seg

def predict_ps_intra_comm_time(tensor_size):
    return wrap_predict(piecewise_linear_func, intra_2GPU_para.para_3seg, tensor_size)

USE_INTERPOLATION=False
def predict_ps_inter_comm_time(tensor_size, is_push):
    if USE_INTERPOLATION:
        if is_push:
            return interpolation(tensor_size, data_table["PUSH_REQ"])
        else:
            return interpolation(tensor_size, data_table["PULL_RES"])
    else:
        if is_push:
            return wrap_predict(piecewise_linear_func, push_data.para_3seg, tensor_size)
        else:
            return wrap_predict(piecewise_linear_func, pull_data.para_3seg, tensor_size)
        ### 20210827_01: Previous method using coarse grained profiled push_pull time
        # all_time = wrap_predict(piecewise_linear_3seg, inter_100Gb_para.para_3seg, tensor_size)
        # intra_time = predict_ps_intra_comm_time(tensor_size)
        # return all_time - intra_time

