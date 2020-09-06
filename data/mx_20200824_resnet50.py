"""
Platform: On one V100 GPU, single machine
Framework: Tensorflow 1.14, CUDA 10.2
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import os, sys
import seaborn as sns
from ml_platform.mxnet.metadata import MetaInfo

# is_show = [True, True, False, True, False, False, False, False]
is_show = None
GFLOPS_FP32 = 1
GFLOPS_FP16 = 2
BATCHSIZE_THESHOLD = 4
VAR_THREHOLD = 0.2
TRAIN_PERCENT = 1
ADD_ADDITIONAL = True

NAMELIST_32 = None
NAMELIST_16 = None
DATA_32 = {}
DATA_16 = {}
VAR_32 = {}
VAR_16 = {}
DEFAULT_BATCH_SIZE_STR="B=256"
DEFAULT_BATCH_SIZE=int(DEFAULT_BATCH_SIZE_STR.split("=")[1])
DEFAULT_KENREL_SIZE=3
BATCH_LIST_VALUE = []

data_folder = "/Users/hhp/0/git/byteprofile-analysis/data/data_20200824"
if sys.argv[1] == 'bert':
    RST_DIR=os.path.join(data_folder, "20200824_03")
elif sys.argv[1] == 'resnet':
    RST_DIR=os.path.join(data_folder, "20200824_02")
    # RST_DIR="/Users/hhp/0/git/byteprofile-analysis/data/data_20200824/20200828_01"
else:
    raise
meta_info = MetaInfo(os.path.join(RST_DIR, "host0/0"))

train_x = train_y = None
test_x = test_y = None
fp32_x = fp32_y = None
fp16_x = fp16_y = None

def str2list(_list, dtype=str):
    assert dtype in [int, float, str]
    elems = _list.split("[")[1].split("]")[0].split(", ")

    if dtype == str:
        return [str(e.split("'")[1]) for e in elems]
    else:
        return [dtype(e) for e in elems]

with open(os.path.join(RST_DIR, "name.txt"), 'r') as fp:
    lines = fp.read().split("\n")
    if lines[-1] == "":
        lines = lines[:-1]
    for line in lines:
        if "fp32" in line:
            NAMELIST_32 = str2list(line.split(":")[1])
        elif "fp16" in line:
            NAMELIST_16 = str2list(line.split(":")[1])
        else:
            raise

with open(os.path.join(RST_DIR, "avg.txt"), 'r') as fp:
    lines = fp.read().split("\n")
    idx = 0
    while idx < len(lines):
        if "huhanpeng" in lines[idx]:
            if idx+1 < len(lines) and ("huhanpeng" in lines[idx+1] or lines[idx+1]==""):
                ### avoid add addition batch size to BATCH_LIST_VALUE
                idx += 1
                continue
            if sys.argv[1] == 'bert':
                batchsize = int(lines[idx].split("--total_batch_size")[1].split("--")[0])
            elif sys.argv[1] == 'resnet':
                batchsize = int(lines[idx].split("--batch-size")[1].split("--")[0])
            else:
                raise
            if "fp32" in lines[idx]:
                BATCH_LIST_VALUE.append(batchsize)
                _DATA = DATA_32
                _VAR = VAR_32
            elif "fp16" in lines[idx]:
                _DATA = DATA_16
                _VAR = VAR_16
            else:
                raise
            idx += 1
            if idx >= len(lines) or "huhanpeng" not in lines[idx]:
                _DATA["B=%d"%batchsize] = str2list(lines[idx], dtype=float)
            else:
                continue
            idx += 1
            if idx >= len(lines) or "huhanpeng" not in lines[idx]:
                _VAR["B=%d"%batchsize] = str2list(lines[idx], dtype=float)
            else:
                continue
        idx += 1
# assert "network/resblock_3_1/conv_1/conv2d/Conv2D" in NAMELIST_32, "%s"%str(NAMELIST_32)

BATCH_LIST_VALUE = [e for e in BATCH_LIST_VALUE if (e >= 0 and e <=1024)]
BATCH_LIST_VALUE = sorted(BATCH_LIST_VALUE)
BATCH_LIST_STR = ["B=%d"%e for e in BATCH_LIST_VALUE]

def batchsize2avg(index, fp16=False):
    avg = []
    _DATA = DATA_16 if fp16 else DATA_32
    for e in BATCH_LIST_STR:
        avg.append(_DATA[e][index])
    return avg

def batchsize2sdt(index, fp16=False):
    vars_ = []
    _VAR = VAR_16 if fp16 else VAR_32
    for e in BATCH_LIST_STR:
        vars_.append(_VAR[e][index])
    return np.sqrt(np.array(vars_))

def numOfMulInDense(B, C_in, C_out):
    return B * C_in * C_out

def numOfMulInConv(B, width, K, C_in, C_out):
    return B * width * width * K * K * C_in * C_out

def numOfAddInDense(B, C_in, C_out):
    return B * (C_in - 1) * C_out

def numOfAddInConv(B, width, K, C_in, C_out):
    return B * width * width * (K * K * C_in - 1) * C_out

def sizeInOfDense(B, C_in, C_out):
    return B * C_in

def sizeInOfConv(B, width, K, C_in, C_out):
    return B * width * width * C_in

def sizeOutOfDense(B, C_in, C_out):
    return B * C_out

def sizeOutOfConv(B, width, K, C_in, C_out):
    return B * width * width * C_out

def sizeWeiOfDense(B, C_in, C_out):
    return C_in * C_out

def sizeWeiOfConv(B, width, K, C_in, C_out):
    return K * K * C_in * C_out

def infoOfConv(B, width, K, C_in, C_out):
    return numOfMulInConv(B, width, K, C_in, C_out), \
        numOfAddInConv(B, width, K, C_in, C_out), \
        sizeInOfConv(B, width, K, C_in, C_out), \
        sizeOutOfConv(B, width, K, C_in, C_out), \
        sizeWeiOfConv(B, width, K, C_in, C_out)

def infoOfDense(B, C_in, C_out):
    return numOfMulInDense(B, C_in, C_out), \
        numOfAddInDense(B, C_in, C_out), \
        sizeInOfDense(B, C_in, C_out), \
        sizeOutOfDense(B, C_in, C_out), \
        sizeWeiOfDense(B, C_in, C_out)

def init_fig_base(cnt):
    h = math.ceil(math.sqrt(cnt))
    w = math.ceil(cnt / h)
    fig_base = w * 100 + h * 10 + 1
    return fig_base, 0

# show_model_complexity()
if sys.argv[1] == "bert":
    OP_NAMES = [
"FW.bertencoder0_embedding0",
"FW.bertmodel0_token_type_embed_embedding0",
"FW.bertmodel0_word_embed_embedding0",
    ]
    OP_LABELS = ["".join(n.split("bert")[1]) for n in OP_NAMES]
elif sys.argv[1] == 'resnet':
    OP_NAMES = [
"FW.resnetv10_conv0",
"FW.resnetv10_stage1_conv0",
"FW.resnetv10_stage1_conv1",
"FW.resnetv10_stage1_conv2",
"FW.resnetv10_stage1_conv3",
"FW.resnetv10_stage1_conv4",
"FW.resnetv10_stage1_conv5",
"FW.resnetv10_stage1_conv6",
"FW.resnetv10_stage1_conv7",
"FW.resnetv10_stage1_conv8",
"FW.resnetv10_stage1_conv9",
"FW.resnetv10_stage2_conv0",
"FW.resnetv10_stage2_conv1",
"FW.resnetv10_stage2_conv2",
"FW.resnetv10_stage2_conv3",
"FW.resnetv10_stage2_conv4",
"FW.resnetv10_stage2_conv5",
"FW.resnetv10_stage2_conv6",
"FW.resnetv10_stage2_conv7",
"FW.resnetv10_stage2_conv8",
"FW.resnetv10_stage2_conv9",
"FW.resnetv10_stage2_conv10",
"FW.resnetv10_stage2_conv11",
"FW.resnetv10_stage2_conv12",
"FW.resnetv10_stage3_conv0",
"FW.resnetv10_stage3_conv1",
"FW.resnetv10_stage3_conv2",
"FW.resnetv10_stage3_conv3",
"FW.resnetv10_stage3_conv4",
"FW.resnetv10_stage3_conv5",
"FW.resnetv10_stage3_conv6",
"FW.resnetv10_stage3_conv7",
"FW.resnetv10_stage3_conv8",
"FW.resnetv10_stage3_conv9",
"FW.resnetv10_stage3_conv10",
"FW.resnetv10_stage3_conv11",
"FW.resnetv10_stage3_conv12",
"FW.resnetv10_stage3_conv13",
"FW.resnetv10_stage3_conv14",
"FW.resnetv10_stage3_conv15",
"FW.resnetv10_stage3_conv16",
"FW.resnetv10_stage3_conv17",
"FW.resnetv10_stage3_conv18",
"FW.resnetv10_stage4_conv0",
"FW.resnetv10_stage4_conv1",
"FW.resnetv10_stage4_conv2",
"FW.resnetv10_stage4_conv3",
"FW.resnetv10_stage4_conv4",
"FW.resnetv10_stage4_conv5",
"FW.resnetv10_stage4_conv6",
"FW.resnetv10_stage4_conv7",
"FW.resnetv10_stage4_conv8",
"FW.resnetv10_stage4_conv9",
            ]
    OP_LABELS = ["".join(n.split("resnetv10_")[1]) for n in OP_NAMES]
else:
    raise
OP_SHORT_LABELS = OP_LABELS
DOTS = ['.-', '^--', 'x-']
# for n in NAMELIST_32:
#     if "FW" in n and "reshape" not in n and "embedding" in n:
#         print(n)
# raise


### model_size[node_name][S_mul, S_add, ...][len= # of batch value]
model_size = np.array([list(zip(*[meta_info.ret_mx_metadata(op_name, batch_size=b)[1:] for b in BATCH_LIST_VALUE])) for op_name in OP_NAMES])
model_raw_info = np.array([list(zip(*[meta_info.ret_mx_rawmeta(op_name, batch_size=b) for b in BATCH_LIST_VALUE])) for op_name in OP_NAMES])

intensity = model_size[:, 0, :] / (model_size[:, 2, :] + model_size[:, 3, :] + model_size[:, 4, :])

THRESHOLD = 0.00 # in ms

def plot_varyB_resut(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False):
    plt.figure(num=1, figsize=(8, 6))

    x_axis_idx = None
    if S_mul:
        x_axis_idx = 0
    elif S_add:
        x_axis_idx = 1
    elif S_in:
        x_axis_idx = 2
    elif S_out:
        x_axis_idx = 3
    elif S_wei:
        x_axis_idx = 4

    x_axis_names = ["S_mul", "S_add", "S_in", "S_out", "S_weight"]
    x_axis_name = "Batch Size (B)" if x_axis_idx is None else x_axis_names[x_axis_idx]

    fig_base, _ = init_fig_base(len(OP_NAMES))
    print(fig_base)

    def __plot(fig_base, idx):
        ax = plt.subplot(fig_base + idx)
        x_axis = BATCH_LIST_VALUE if x_axis_idx is None else model_size[idx][x_axis_idx]
        avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[idx]))
        ax.plot(x_axis, avgs, marker='.', label=OP_LABELS[idx] + "_fp32")
        if NAMELIST_16 is not None:
            avgs_16 = batchsize2avg(NAMELIST_16.index(OP_NAMES[idx]), fp16=True)
            ax.plot(x_axis, avgs_16, marker='^', label=OP_LABELS[idx] + "_fp16")

        plt.legend()
        plt.ylabel('Average Time (ms)')
        plt.xlabel(x_axis_name)

    for i in range(len(OP_NAMES)):
        __plot(fig_base, i)

    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel(x_axis_name)

    plt.show()

def plot_batchsize_intensity():
    plt.figure(num=1, figsize=(8, 6))

    fig_base, fig_idx = init_fig_base(len(OP_NAMES))
    x_axis_name = "Batch Size (B)"

    def __plot(fig_base, op_id, flops=False):
        avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
        avgs_16 = batchsize2avg(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

        ax = plt.subplot(fig_base + op_id)
        ax.plot(BATCH_LIST_VALUE, intensity[op_id], '.-', label=OP_LABELS[op_id]+"_intensity")

        # y = intensity[op_id][BATCH_LIST_VALUE.index(4)]
        # ax.plot([4], [y], '^', label="B=4"+" intensity=%d"%y)
        # y = intensity[op_id][BATCH_LIST_VALUE.index(16)]
        # ax.plot([16], [y], '^', label="B=16"+" intensity=%d"%y)
        # y = intensity[op_id][BATCH_LIST_VALUE.index(64)]
        # ax.plot([64], [y], '^', label="B=64"+" intensity=%d"%y)

        plt.legend()
        plt.ylabel('Intensity')
        plt.xlabel(x_axis_name)
        op_id + 1

    for op_id in range(len(OP_NAMES)):
        __plot(fig_base, op_id)

    plt.show()

def plot_intensity_flops():
    plt.figure(num=1, figsize=(8, 6))

    fig_base, fig_idx = init_fig_base(len(OP_NAMES))
    x_axis_name = "Batch Size (B)"

    def __plot(fig_base, op_id):
        ax = plt.subplot(fig_base + op_id)

        avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
        avgs_16 = batchsize2avg(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
        flops_32 = model_size[op_id, 0, :] / np.array(avgs)
        flops_16 = model_size[op_id, 0, :] / np.array(avgs_16)
        ax.plot(intensity[op_id], flops_32, '.-', label=OP_LABELS[op_id]+"_fp32_flops")
        ax.plot(intensity[op_id], flops_16, '.-', label=OP_LABELS[op_id]+"_fp16_flops")

        plt.legend()
        plt.ylabel('FLOPS')
        plt.xlabel("arithmetic intensity")

    for op_id in range(len(OP_NAMES)):
        __plot(fig_base, op_id)

    plt.show()

def plot_batchsize_size():
    plt.figure(num=1, figsize=(8, 6))

    fig_base, fig_idx = init_fig_base(len(OP_NAMES))
    x_axis_name = "Batch Size (B)"

    def __plot(fig_base, op_id):
        ax = plt.subplot(fig_base + op_id)
        for fig_id, metric in enumerate([
                # "mul", 
                # "input", 
                # "output", 
                "weight",
                ]):
            ax.plot(BATCH_LIST_VALUE, model_size[op_id, fig_id if fig_id < 1 else fig_id+1, :], DOTS[fig_id%len(DOTS)], label=OP_SHORT_LABELS[op_id]+"_"+metric)
        plt.legend()
        plt.ylabel('size')
        plt.xlabel(x_axis_name)

    for op_id in range(len(OP_NAMES)):
        __plot(fig_base, op_id)

    plt.show()


def plot_avg_accum_distribution():
    plt.figure(num=1, figsize=(8, 6))
    def __plot(op_id):
        ax = plt.subplot(211 + op_id)

        avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
        avgs_16 = batchsize2avg(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

        op_id += 1

        avgs_ = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
        avgs_16_ = batchsize2avg(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

        avgs = np.concatenate((avgs, avgs_))
        avgs_16 = np.concatenate((avgs_16, avgs_16_))

        def _gen(_array):
            assert len(_array.shape)==1
            _array = np.sort(_array)
            _len = _array.shape[0]
            dist = []
            ret = None
            for i in range(_len):
                dist.append(100 * i / _len)
                if ret is None and _array[i] > THRESHOLD:
                    ret = 100 * i / _len
            return _array, np.array(dist), ret

        x1, y1, ret1 = _gen(avgs)
        x2, y2, ret2 = _gen(avgs_16)

        ax.plot(x1, y1, label="fp32" + ", fp32 == %f: %6.4f %%"%(THRESHOLD, ret1))
        ax.plot(x2, y2, label="fp16" + ", fp16 == %f: %6.4f %%"%(THRESHOLD, ret2))

        # ax.text(0.1, ret1, "fp32: {}, {} %".format(0.1, ret1))
        # ax.text(0.1, ret2, "fp16: {}, {} %".format(0.1, ret2))

        plt.legend()
        plt.ylabel('Cumulative Distribution (%)')
        plt.xlabel("Execution time (ms)")

    for op_id in range(2):
        __plot(op_id)

    plt.show()

def plot_varyK_result(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False):
    plt.figure(num=1, figsize=(8, 6))

    cnt = 1 + int(S_mul) + int(S_add) + int(S_in) + int(S_out) + int(S_wei)
    fig_base, fig_idx = init_fig_base(cnt)
    
    model_size = [list(zip(*[infoOfConv(DEFAULT_BATCH_SIZE, 28, k, 1, 32) for k in KENREL_LIST_VALUE])),
        list(zip(*[infoOfConv(DEFAULT_BATCH_SIZE, 14, k, 32, 64) for k in KENREL_LIST_VALUE]))]

    def __plot(x_axis, x_axis2, x_name, _fig_idx):
        ax = plt.subplot(fig_base+_fig_idx)
        ax.plot(x_axis, vary_kernel_size(NAMELIST_32.index('conv_layer1/conv2d/Conv2D')), marker='.', label="conv1 (K*K*1*32)")
        ax.plot(x_axis, vary_kernel_size(NAMELIST_16.index('conv_layer1/conv2d/Conv2D'), fp16=True), marker='^', label="conv1 (K*K*1*32) + fp16")
        ax.plot(x_axis2[:6], vary_kernel_size(NAMELIST_32.index('conv_layer2/conv2d/Conv2D'))[:6], marker='.', label="conv2 (K*K*32*64)")
        ax.plot(x_axis2[:6], vary_kernel_size(NAMELIST_16.index('conv_layer2/conv2d/Conv2D'), fp16=True)[:6], marker='^', label="conv2 (K*K*32*64) + fp16")
        plt.legend()
        plt.ylabel('Average Time (ms)')
        plt.xlabel(x_name)
        return _fig_idx + 1

    fig_idx = __plot(KENREL_LIST_VALUE, KENREL_LIST_VALUE, "Kernel Size (K)", fig_idx)

    if S_mul:
        fig_idx = __plot(model_size[0][0], model_size[1][0], "S_mul", fig_idx)

    if S_add:
        fig_idx = __plot(model_size[0][1], model_size[1][1], "S_add", fig_idx)

    if S_in:
        fig_idx = __plot(model_size[0][2], model_size[1][2], "S_in", fig_idx)

    if S_out:
        fig_idx = __plot(model_size[0][3], model_size[1][3], "S_out", fig_idx)

    if S_wei:
        fig_idx = __plot(model_size[0][4], model_size[1][4], "S_wei", fig_idx)

    plt.show()

def plot_varyD_result():
    plt.figure(num=1, figsize=(8, 6))
    ax = plt.subplot(121)
    ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_32.index('dense/MatMul')), marker='.', label="dense (3136*D)")
    ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_16.index('dense/MatMul'), fp16=True), marker='^', label="dense (3136*D) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Dense Size (D)")

    ax = plt.subplot(122)
    ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_32.index('dense_1/MatMul')), marker='.', label="dense1 (D*10)")
    ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_16.index('dense_1/MatMul'), fp16=True), marker='^', label="dense1 (D*10) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Dense Size (D)")
    plt.show()

def plot_varyB_resut_of_cast():
    plt.figure(num=1, figsize=(8, 6))

    ax = plt.subplot(221)
    avgs_16 = batchsize2avg(NAMELIST_16.index('conv_layer1/Cast_1'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="conv_layer1/Cast_1:fp16->fp32")
    avgs_16 = batchsize2avg(NAMELIST_16.index('conv_layer1/conv2d/Conv2D'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="conv1 (5*5*1*32) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Batch Size (B)")


    ax = plt.subplot(222)
    avgs_16 = batchsize2avg(NAMELIST_16.index('conv_layer2/Cast'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="conv_layer2/Cast:fp32->fp16")
    avgs_16 = batchsize2avg(NAMELIST_16.index('conv_layer2/conv2d/Conv2D'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="conv2 (5*5*32*64) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Batch Size (B)")

    ax = plt.subplot(223)
    avgs_16 = batchsize2avg(NAMELIST_16.index('Cast_2'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="Cast_2(dense):fp16->fp32")
    avgs_16 = batchsize2avg(NAMELIST_16.index('dense/MatMul'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="dense (3136*1024) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Batch Size (B)")

    ax = plt.subplot(224)
    avgs_16 = batchsize2avg(NAMELIST_16.index('Cast_3'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="Cast_3(dense1):fp32->fp16")
    avgs_16 = batchsize2avg(NAMELIST_16.index('dense_1/MatMul'), fp16=True)
    ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="dense1 (1024*10) + fp16")
    plt.legend()
    plt.ylabel('Average Time (ms)')
    plt.xlabel("Batch Size (B)")

    plt.show()

def plot_model_complexity_combine():
    plt.figure(num=1, figsize=(8, 6))
    # x_axis_names = ["S_mul", "S_add", "S_in", "S_out", "S_weight"]
    x_axis_name = "Batch Size (B)"

    fig_id = 0
    for fig_id, metric in enumerate(["mul", "input", "output", "weight"]):
        ax = plt.subplot(221 + fig_id)
        for op_id in range(len(OP_NAMES)):
            ax.plot(BATCH_LIST_VALUE, model_size[op_id, fig_id if fig_id < 1 else fig_id+1, :], DOTS[op_id%len(DOTS)], label=OP_LABELS[op_id])
        plt.legend()
        plt.ylabel('size of ' + metric)
        plt.xlabel(x_axis_name)

    plt.show()

# plot_varyK_result(S_mul=True, S_add=True, S_in=True, S_out=True, S_wei=True)
# plot_varyD_result()
# plot_varyB_resut(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False)
# plot_batchsize_intensity()
# plot_intensity_flops()
# plot_batchsize_size()
# plot_model_complexity_combine()
# plot_varyB_resut_of_cast()
# plot_avg_accum_distribution()
# raise


####################################################################################################
#############################        Start to Fit          #########################################
####################################################################################################

def exct_filter(target, others=None):
    if len(target.shape) == 1:
        _filter = np.where(target > THRESHOLD)
        if len(others.shape) == 1:
            return target[_filter], others[_filter]
        else:
            return target[_filter], others[:, _filter].reshape(list(others.shape[:-1]) + [-1])
    else:
        ### target[sample id][avg, G, S_mul, S_add, ...]
        _min = 100
        _max = 100000000
        print(target.shape)
        target = target[((target[:, 2] > _min) & (target[:, 2] < _max))]
        print(target.shape)
        return target

def predict_error(_list, _list_pred):
    _list_pred = np.array(_list_pred)
    _list = np.array(_list)
    _list, _list_pred = exct_filter(_list, _list_pred)

    if len(_list) == 0:
        return None, "Original time is too small. Ignore!!!"

    diff = np.abs(_list_pred - _list) / _list
    return diff, "%f %%"%(np.average(diff * 100))


# def cost_func(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3):
#     '''
#     gflops:
#         We only need a relative value of gflops, 
#         i.e., if we know fp32's is twice of fp16's, we can just fp32's = 2 and fp16's = 1,
#         the scale is hidden in the a2 
#         x[0]: relative gflops
#         x[1]: num of multiplication
#         x[2]: num of addition
#         x[3]: input size
#         x[4]: output size
#         x[5]: weight size

#         if len(x) > 6, there are some additional information, e.g., kernel size for Conv2D
#     '''
#     gflops = xs[0]
#     S_mul = xs[1]
#     S_add = xs[2]
#     wei_S_all = a3 * xs[3] + a4 * xs[4] + a5 * xs[5]
#     wei_S_all2 = a6 * xs[3] + a7 * xs[4] + a8 * xs[5]
#     if ADD_ADDITIONAL:
#         ### [H, W, C, R, S, P, Q, K, batch_size, use_bias]
#         addtional_term = a9 * xs[4] * xs[6+9]
#     else:
#         addtional_term = 0
#     return (a1 * S_mul + b1 + addtional_term) / (a2 * gflops + b2) + wei_S_all / gflops + b3 + gflops * wei_S_all2

# lower_bounds = tuple([0]*9 + [-np.inf]*3)

def cost_func(xs, a1, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3):
    '''
    gflops:
        We only need a relative value of gflops, 
        i.e., if we know fp32's is twice of fp16's, we can just fp32's = 2 and fp16's = 1,
        the scale is hidden in the a2 
        x[0]: relative gflops
        x[1]: num of multiplication
        x[2]: num of addition
        x[3]: input size
        x[4]: output size
        x[5]: weight size

        if len(x) > 6, there are some additional information, e.g., kernel size for Conv2D
    '''
    gflops = xs[0]
    S_mul = xs[1] / 3776446464.0
    S_add = xs[2] / 3776446464.0
    S_in = xs[3] / 25690112.0
    S_out = xs[4] / 25690112.0
    S_wei = xs[5] / 2359296.0
    wei_S_all = a3 * S_in + a4 * S_out + a5 * xs[5]
    # wei_S_all = a3 * xs[3] + a4 * xs[4] + a5 * xs[5]
    # wei_S_all2 = a6 * xs[3] + a7 * xs[4] + a8 * xs[5]

    flops_ = gflops * (1 / (1 + np.exp(-S_mul)) - a6)
    # flops_ = np.sqrt(a7**2 * (gflops ** 2 / (a6**2) - 1))
    if ADD_ADDITIONAL:
        ### [H, W, C, R, S, P, Q, K, batch_size, use_bias]
        addtional_term = S_out * xs[6+9]
    else:
        addtional_term = 0
    return (a1 * S_mul + b1 + a9 * (S_add + addtional_term)) / (flops_ + b2) + b3 + wei_S_all / gflops

lower_bounds = tuple([0]*8 + [-np.inf]*3)


# lower_bounds = tuple([0]*4 + [-np.inf]*1)
up_bounds = tuple(len(lower_bounds) * [np.inf])
p0=[0]*len(lower_bounds)
FIT_FUNC = cost_func

def collect_data(op_names_, add_=False, verbose=True):
    all_data_dict = {}

    def __record_xdata(S_mul, S_add, S_in, S_out, S_wei, gflops, avg, op_type, addition=None):
        if op_type not in all_data_dict:
            all_data_dict[op_type] = [[], [], [], [], [], [], []]
        all_data_dict[op_type][0].append(avg)
        all_data_dict[op_type][1].append(gflops)
        all_data_dict[op_type][2].append(S_mul)
        all_data_dict[op_type][3].append(S_add)
        all_data_dict[op_type][4].append(S_in)
        all_data_dict[op_type][5].append(S_out)
        all_data_dict[op_type][6].append(S_wei)

        if addition is not None and isinstance(addition, list):
            for idx, e in enumerate(addition):
                if idx+7 >= len(all_data_dict[op_type]):
                    all_data_dict[op_type].append([addition[idx]])
                else:
                    all_data_dict[op_type][idx+7].append(addition[idx])

    for i in range(len(op_names_)):
        for b in BATCH_LIST_VALUE:
            ### filter
            if b <= BATCHSIZE_THESHOLD:
                continue

            ### collect data
            op_type, S_mul, S_add, S_in, S_out, S_wei = meta_info.ret_mx_metadata(op_names_[i], batch_size=b)

            idx_in_32 = NAMELIST_32.index(op_names_[i])
            var_ = VAR_32["B=%d"%b][idx_in_32] if "B=%d"%b in VAR_32 else 0
            avg_ = DATA_32["B=%d"%b][idx_in_32]
            if (var_ / avg_) <= VAR_THREHOLD:
                ### [H, W, C, R, S, P, Q, K, batch_size]
                raw_meta = meta_info.ret_mx_rawmeta(op_names_[i], batch_size=b) if add_ else None
                __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP32, avg_, "Conv2D", addition=raw_meta)

            if NAMELIST_16 is not None:
                idx_in_16 = NAMELIST_16.index(op_names_[i])
                var_ = VAR_16["B=%d"%b][idx_in_16] if "B=%d"%b in VAR_16 else 0
                avg_ = DATA_16["B=%d"%b][idx_in_16]
                if (var_ / avg_) <= VAR_THREHOLD:
                    raw_meta = meta_info.ret_mx_rawmeta(op_names_[i], batch_size=b) if add_ else None
                    __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP16, avg_, "Conv2D", addition=raw_meta)

        _raw_meta =[0]*len(raw_meta) if add_ else None
        # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP32, 0, "Conv2D", addition=_raw_meta)
        # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP16, 0, "Conv2D", addition=_raw_meta)
        # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP32, 0, "Conv2D", addition=None)
        # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP16, 0, "Conv2D", addition=None)

    ### all_data[S_mul, ...][# of nodes * # of batch size values]
    try:
        all_data = np.array(all_data_dict["Conv2D"])
    except:
        print(op_names_, all_data_dict)
        raise
    ### all_data[# of nodes * # of batch size values][S_mul, ...]
    all_data = np.transpose(all_data)
    ### filter
    # all_data = exct_filter(all_data)

    ### the order has not been changed
    global fp32_x, fp32_y, fp16_x, fp16_y
    fp32_data = all_data[all_data[:, 1] == GFLOPS_FP32]
    fp32_data = np.split(fp32_data, [1], axis=1)
    fp32_x, fp32_y = fp32_data[1], fp32_data[0]
    fp16_data = all_data[all_data[:, 1] == GFLOPS_FP16]
    fp16_data = np.split(fp16_data, [1], axis=1)
    fp16_x, fp16_y = fp16_data[1], fp16_data[0]
    if verbose:
        print("Collect fp32 data - X:{}, Y:{}, fp16 data - X:{}, Y:{}".format(fp32_x.shape, fp32_y.shape, fp16_x.shape, fp16_y.shape))

    arg_num = all_data.shape[1]
    value_num = all_data.shape[0]
    np.random.shuffle(all_data)

    ### split data to training data and test data
    mask = np.zeros(value_num, dtype=bool)
    train_idx = np.random.choice(value_num, int(TRAIN_PERCENT * value_num), replace=False)
    mask[train_idx] = True
    train_data = np.split(all_data[mask, :], [1], axis=1)
    if TRAIN_PERCENT >= 1:
        test_data = train_data
    else:
        test_data = np.split(all_data[~mask, :], [1], axis=1)
    global train_x, train_y, test_x, test_y
    train_x, train_y = train_data[1], train_data[0]
    test_x, test_y = test_data[1], test_data[0]
    if verbose:
        print("Collect training data - X:{}, Y:{}, test data - X:{}, Y:{}".format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))


def fit_with_S_cal_gflops():
    _train_x = np.transpose(train_x)
    _train_y = np.transpose(train_y).flatten()
    popt, pcov = curve_fit(FIT_FUNC, _train_x, _train_y, 
        bounds=(lower_bounds, up_bounds), p0=p0, maxfev=10000)
    return popt, pcov

def plot_2d_fit_result(is_show, op_idxs=None):
    plt.figure(num=1, figsize=(8, 6))

    ratio_sum = []
    # cnt = sum([int(i) for i in is_show])
    if op_idxs is not None:
        op_names_ = np.array(OP_NAMES)[idxs]
    else:
        op_names_ = OP_NAMES
    cnt = len(op_names_)
    fig_base, fig_idx = init_fig_base(cnt)

    clrs = sns.color_palette("husl", 5)

    def __plot(op_id, fig_base, fig_idx):
        x_axis_names = ["Batch Size (B)", "S_mul", "S_add", "S_in", "S_out", "S_weight"]
        x_axis_idx = 0
        xaxis = [b if x_axis_idx == 0 else meta_info.ret_mx_metadata(OP_NAMES[op_id], batch_size=b)[x_axis_idx] for b in BATCH_LIST_VALUE]
        model_size_32 = np.concatenate((np.array([[GFLOPS_FP32]*len(model_size[op_id, 0, :])]), model_size[op_id, :, :], model_raw_info[op_id, :, :]), axis=0) \
            if ADD_ADDITIONAL else np.concatenate((np.array([[GFLOPS_FP32]*len(model_size[op_id, 0, :])]), model_size[op_id, :, :]), axis=0)
        model_size_16 = np.concatenate((np.array([[GFLOPS_FP16]*len(model_size[op_id, 0, :])]), model_size[op_id, :, :], model_raw_info[op_id, :, :]), axis=0) \
            if ADD_ADDITIONAL else np.concatenate((np.array([[GFLOPS_FP16]*len(model_size[op_id, 0, :])]), model_size[op_id, :, :]), axis=0)
        ax = plt.subplot(fig_base)

        avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
        ax.plot(xaxis, avgs, marker='.', label=OP_SHORT_LABELS[op_id]+"_fp32", c=clrs[0])
        try:
            stds = batchsize2sdt(NAMELIST_32.index(OP_NAMES[op_id]))
            ax.fill_between(xaxis, avgs+stds, avgs-stds, alpha=0.3, facecolor=clrs[0])
        except KeyError:
            pass

        avgs_pred = FIT_FUNC(model_size_32, *popt)
        ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp32"+"_pred", c=clrs[1])
        # ax.fill_between(xaxis, FIT_FUNC(model_size_32, *popt+perr), FIT_FUNC(model_size_32, *popt-perr), alpha=0.3, facecolor=clrs[1])
        diff, ratio = predict_error(avgs, avgs_pred)
        print(OP_SHORT_LABELS[op_id]+"_fp32", ratio, diff)
        fig_idx += 1
        if "%" in ratio:
            ratio_sum.append(float(ratio.split("%")[0]))

        if NAMELIST_16 is not None:
            avgs_16 = batchsize2avg(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
            ax.plot(xaxis, avgs_16, marker='^', label=OP_SHORT_LABELS[op_id]+"_fp16", c=clrs[2])
            try:
                stds_16 = batchsize2sdt(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
                ax.fill_between(xaxis, avgs_16+stds_16, avgs_16-stds_16, alpha=0.3, facecolor=clrs[2])
            except KeyError:
                pass
            avgs_pred = FIT_FUNC(model_size_16, *popt)
            ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp16"+"_pred", c=clrs[3])
            # ax.fill_between(xaxis, FIT_FUNC(model_size_16, *popt+perr), FIT_FUNC(model_size_16, *popt-perr), alpha=0.3, facecolor=clrs[3])
            diff, ratio = predict_error(avgs_16, avgs_pred)
            print(OP_SHORT_LABELS[op_id]+"_fp16", ratio, diff)
            fig_idx += 1
            if "%" in ratio:
                ratio_sum.append(float(ratio.split("%")[0]))
        
        plt.legend(fontsize=8)
        plt.ylabel('Average Time (ms)')
        # plt.ylim(0, 2)
        plt.xlabel(x_axis_names[x_axis_idx])
        return fig_base+1, fig_idx

    for op_id in op_idxs:
        # if is_show[op_id]:
        # op_id = 2
        fig_base, fig_idx = __plot(op_id, fig_base, fig_idx)
    
    print("average error: %f %%"%(sum(ratio_sum)/len(ratio_sum)))
    plt.show()

def test(verbose=True):
    _test_x = np.transpose(test_x)
    _test_y = np.transpose(test_y).flatten()
    avgs_pred = FIT_FUNC(_test_x, *popt)
    # print(_test_y, avgs_pred)
    diff, ratio = predict_error(_test_y, avgs_pred)
    error = float(ratio.split("%")[0])
    if verbose:
        print("average error: %f %%"%(error))
    return error

def test_speedup_ratio():
    _fp32_x = np.transpose(fp32_x)
    _fp32_y = np.transpose(fp32_y).flatten()
    _fp32_y_pred = FIT_FUNC(_fp32_x, *popt)
    diff, ratio = predict_error(_fp32_y, _fp32_y_pred)
    print("fp32 average error: %f %%"%(float(ratio.split("%")[0])))

    _fp16_x = np.transpose(fp16_x)
    _fp16_y = np.transpose(fp16_y).flatten()
    _fp16_y_pred = FIT_FUNC(_fp16_x, *popt)
    diff, ratio = predict_error(_fp16_y, _fp16_y_pred)
    print("fp16 average error: %f %%"%(float(ratio.split("%")[0])))

    diff, ratio = predict_error(_fp16_y, (_fp16_y_pred / _fp32_y_pred) * _fp32_y)
    print("speedup ratio average error: %f %%"%(float(ratio.split("%")[0])))

'''
#####################################################################
### Corss validation grouped by block
resnetLayernumOfBlock = [1, 10, 13, 19, 10]
resnetRangeUp = [sum(resnetLayernumOfBlock[:(i+1)]) for i in range(len(resnetLayernumOfBlock))]
resnetRangeDown = [0] + resnetRangeUp[:-1]
for i in range(len(resnetLayernumOfBlock)):
    print("The {}th block of Resnet, from {} to {}".format(i, resnetRangeDown[i], resnetRangeUp[i]))
    op_names_ = OP_NAMES[resnetRangeDown[i]:resnetRangeUp[i]]
    collect_data(op_names_)
    popt, pcov = fit_with_S_cal_gflops()
    test()
'''

#####################################################################
### Cross validation grouped by individual op
# collect_data(OP_NAMES)
# popt, pcov = fit_with_S_cal_gflops()
# for i in range(len(OP_NAMES)):
#     op_names_ = [OP_NAMES[i]]
#     collect_data(op_names_)
#     popt, pcov = fit_with_S_cal_gflops()
#     test()

#####################################################################
### Output attribute of ops
# for op_name in OP_NAMES:
#     _, S_mul, _, S_in, S_out, S_wei = meta_info.ret_mx_metadata(op_name, batch_size=32)
#     ### [H, W, C, R, S, P, Q, K, batch_size]
#     raw_meta = meta_info.ret_mx_rawmeta(op_name, batch_size=32)
#     print(raw_meta[0]/raw_meta[5])

#####################################################################
### Test the error on all OPs
# idxs = np.array([3, 4])
# op_names_ = np.array(OP_NAMES)[idxs]
# collect_data(op_names_, add_=ADD_ADDITIONAL)
# popt, pcov = fit_with_S_cal_gflops()
# print(popt)
# # plot_2d_fit_result(is_show)
# test()

#####################################################################
#####################################################################
# 20200831 (MON)
### exp id 1: vary the number of operators in the training dataset
# for i in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     pct = i
#     error_list = []
#     idxs_list = []
#     for _ in range(100):
#         idxs = np.random.choice(len(OP_NAMES), int(pct * len(OP_NAMES)), replace=False)
#         op_names_ = np.array(OP_NAMES)[idxs]
#         collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#         popt, pcov = fit_with_S_cal_gflops()
#         error_list.append(test(verbose=False))
#         idxs_list.append(idxs)
#     error = sum(error_list) / len(error_list)
#     rst = zip(error_list, idxs_list)
#     rst = sorted(rst, key=lambda x: x[0])
#     print("Percent: {}, error: {} %, std.dev: {}".format(pct, error, np.std(error_list)))
#     print("best:", rst[:2])
#     print("worse:", rst[-2:])

#####################################################################
### exp id 2: according to the fitting result, build dependency graph, edge connect two 'homo' nodes
### make partition of the ops in resnet50_v1 --> Failed !!!
# import networkx as nx
# import matplotlib.pyplot as plt
# graph = nx.Graph()
# no_change_cnt = 0
# while no_change_cnt < 10:
#     ### if no new edge is added to the graph for continuously 10 times, stop
#     pct = 0.1
#     idxs = np.random.choice(len(OP_NAMES), int(pct * len(OP_NAMES)), replace=False)
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#     popt, pcov = fit_with_S_cal_gflops()
#     add_cnt = 0
#     if test(verbose=False) < 10:
#         for i in range(len(idxs)-1):
#             u = idxs[i]
#             v = idxs[i+1]
#             if (u, v) not in graph.edges():
#                 graph.add_edge(u, v)
#                 add_cnt += 1
#         if add_cnt == 0:
#             ### no new edge is added to the graph, add the no_change_cnt by 1
#             no_change_cnt += 1
#         else:
#             ### add new edges to the graph, reset the no_change_cnt
#             no_change_cnt = 0

# def visualize_gml(graph, layout="spectral"):
#     if layout == "spectral":
#         pos = nx.spectral_layout(graph, dim=2, scale=0.5)
#     elif layout == "circular":
#         pos = nx.circular_layout(graph)
#     elif layout == "random":
#         pos = nx.random_layout(graph)
#     nx.draw(graph, pos, with_labels=True, font_size=6)
#     plt.show()

# cpnts = list(nx.connected_components(graph))
# for c in cpnts:
#     op_names_ = np.array(OP_NAMES)[np.array(list(c)).astype(int)]
#     collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#     popt, pcov = fit_with_S_cal_gflops()
#     print(c, test(verbose=False))
# visualize_gml(graph)

#####################################################################
### exp id 3: according to the fitting result, create hyper_edges, where each contain multiple nodes
# import time
# hyper_edges = []
# no_change_cnt = 0
# st = time.time()
# while no_change_cnt < 10:
#     ### if no new edge is added to the graph for continuously 10 times, stop
#     pct = np.random.rand()
#     train_size = int(pct * len(OP_NAMES))
#     if train_size <= 1:
#         continue
#     idxs = np.random.choice(len(OP_NAMES), train_size, replace=False)
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#     popt, pcov = fit_with_S_cal_gflops()
#     error = test(verbose=False)
#     if error < 10:
#         hyper_edge = set(idxs)
#         tmp = []
#         insert = True
#         for edge, error_ in hyper_edges:
#             if hyper_edge.issubset(edge):
#                 insert = False
#                 print("{} is subset of {}".format(hyper_edge, edge))
#                 break
#             elif edge.issubset(hyper_edge):
#                 print("Remove {}".format(edge))
#                 pass
#             else:
#                 tmp.append((edge, error_))

#         if insert:
#             tmp.append((hyper_edge, error))
#             print("Add {}".format(hyper_edge))
#             print("Time: {}, Size: {}".format(time.time() - st, len(tmp)))
#             hyper_edges = tmp
#             no_change_cnt = 0
#         else:
#             no_change_cnt += 1
# print("Final Hyperedges:")
# for edge, error in hyper_edges:
#     print(error, edge)

#####################################################################
#####################################################################
# 20200901 (Tue)
#####################################################################
### exp id 1: Fix the percent to 0.1 
### according to the fitting result, create hyper_edges, where each contain multiple nodes, and 
# hyper_edges = []
# no_change_cnt = 0
# while no_change_cnt < 100:
#     ### if no new edge is added to the graph for continuously 10 times, stop
#     train_size = int(0.1 * len(OP_NAMES))
#     if train_size <= 1:
#         continue
#     idxs = np.random.choice(len(OP_NAMES), train_size, replace=False)
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#     popt, pcov = fit_with_S_cal_gflops()
#     error = test(verbose=False)
#     if error > 20:
#         hyper_edge = set(idxs)
#         tmp = []
#         if hyper_edge in hyper_edges:
#             no_change_cnt += 1
#         else:
#             no_change_cnt = 0
#             hyper_edges.append(hyper_edge)

# for edge, error in hyper_edges:
#     print(error, edge)

#####################################################################
#####################################################################
# 20200903 (Thu)
#####################################################################
### exp id 1: fix the # of ops to 2
### since the variance when # of ops = 2 is very large, see which combinations lead to large error
idxs = np.array([1, 2])
op_names_ = np.array(OP_NAMES)[idxs]
# op_names_ = OP_NAMES
collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=True)
popt, pcov = fit_with_S_cal_gflops()
perr = np.sqrt(np.diag(pcov))
print(popt, perr)
test(verbose=True)
plot_2d_fit_result(is_show, op_idxs=idxs)


# pct = -1
# error_list = []
# idxs_list = []
# for _ in range(100):
#     # idxs = np.random.choice(len(OP_NAMES), 2, replace=False)
#     idxs = np.array([3, 4])
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
#     popt, pcov = fit_with_S_cal_gflops()
#     error_list.append(test(verbose=False))
#     idxs_list.append(idxs)
# error = sum(error_list) / len(error_list)
# rst = zip(error_list, idxs_list)
# rst = sorted(rst, key=lambda x: x[0])
# print("Percent: {}, error: {} %, std.dev: {}".format(pct, error, np.std(error_list)))
# print("best:", rst[:2])
# print("worse:", rst[-2:])

#####################################################################
### exp id 2: fix the # of ops to 2, plot the heat map
target = 'error'
force = False
total = len(OP_NAMES)
if target == 'error':
    file_path = os.path.join(RST_DIR, "2ops_cross_error.txt")
    if not force and os.path.exists(file_path):
        rst = np.loadtxt(file_path)
    else:
        rst = np.zeros((total, total), dtype=np.float)
        for i in range(total-1):
            for j in range(i+1, total):
                op_names_ = [OP_NAMES[i], OP_NAMES[j]]
                collect_data(op_names_, add_=ADD_ADDITIONAL, verbose=False)
                popt, pcov = fit_with_S_cal_gflops()
                error = test(verbose=False)
                rst[i][j] = error
        np.savetxt(file_path, rst)
else:
    ### for meta data
    meta_list = []
    for op_name in OP_NAMES:
        _, S_mul, _, S_in, S_out, S_wei = meta_info.ret_mx_metadata(op_name, batch_size=32)
        ### [H, W, C, R, S, P, Q, K, batch_size]
        raw_meta = meta_info.ret_mx_rawmeta(op_name, batch_size=32)
        if target == 'S_mul':
            meta_list.append(S_mul)
        elif target == 'S_in':
            meta_list.append(S_in)
        elif target == 'S_out':
            meta_list.append(S_out)
        elif target == 'S_wei':
            meta_list.append(S_wei)
        elif target == 'stride':
            meta_list.append(raw_meta[0]/raw_meta[5])
        elif target == 'kernel_size':
            meta_list.append(raw_meta[3])
        else:
            raise
    rst = np.zeros((total, total), dtype=np.float)
    for i in range(total-1):
        for j in range(i+1, total):
            rst[i][j] = np.abs(meta_list[i] - meta_list[j])

rst = np.transpose(rst)
# mask = np.zeros_like(rst, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(8,5))
# ax = sns.heatmap(rst, mask=mask, cmap="YlGnBu")
# if target == 'error':
#     title = 'fitting error'
# elif target == 'S_mul':
#     title = 'diff of S_mul'
# elif target == 'S_in':
#     title = 'diff of S_in'
# elif target == 'S_out':
#     title = 'diff of S_out'
# elif target == 'S_wei':
#     title = 'diff of S_wei'
# elif target == 'stride':
#     title = 'diff of stride'
# elif target == 'kernel_size':
#     title = 'diff of kernel size'
# else:
#     raise
# plt.title("Heatmap for {} for each pair of OPs in Resnet50_v1 on V100-16G".format(title))
# plt.show()





