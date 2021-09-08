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
push_data = DataRepo(None)
push_data.para_2seg = [4.686307490183, -1.662961882088019]
push_data.para_3seg = [4.846827061369098, -1.6483260907019037, 626.2712890335985, 619.9568948850784, 1.1001192383975844]
pull_data = DataRepo(None)
pull_data.para_2seg = [4.803492695527605, -1.5562480802345402]
pull_data.para_3seg = [4.961341192845001, -1.5523328848981286, 626.2723641952061, 619.9558183092073, 1.119712390211427]

def predict_ps_intra_comm_time(tensor_size):
    return wrap_predict(piecewise_linear_3seg, intra_2GPU_para.para_3seg, tensor_size)

def predict_ps_inter_comm_time(tensor_size, is_push):
    if is_push:
        return wrap_predict(piecewise_linear_3seg, push_data.para_3seg, tensor_size)
    else:
        return wrap_predict(piecewise_linear_3seg, pull_data.para_3seg, tensor_size)
    ### 20210827_01: Previous method using coarse grained profiled push_pull time
    # all_time = wrap_predict(piecewise_linear_3seg, inter_100Gb_para.para_3seg, tensor_size)
    # intra_time = predict_ps_intra_comm_time(tensor_size)
    # return all_time - intra_time

