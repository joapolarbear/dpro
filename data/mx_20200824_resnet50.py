"""
Platform: On one V100 GPU, single machine
Framework: Tensorflow 1.14, CUDA 10.2
Model/Dataset: 2conv + 2 dense with MNIST
    # ---------
    # Variables: name (type shape) [size]
    # ---------
    # conv_layer1/conv2d/kernel:0 (float32_ref 5x5x1x32) [800, bytes: 3200]
    # conv_layer1/conv2d/bias:0 (float32_ref 32) [32, bytes: 128]
    # conv_layer2/conv2d/kernel:0 (float32_ref 5x5x32x64) [51200, bytes: 204800]
    # conv_layer2/conv2d/bias:0 (float32_ref 64) [64, bytes: 256]
    # dense/kernel:0 (float32_ref 3136x1024) [3211264, bytes: 12845056]
    # dense/bias:0 (float32_ref 1024) [1024, bytes: 4096]
    # dense_1/kernel:0 (float32_ref 1024x10) [10240, bytes: 40960]
    # dense_1/bias:0 (float32_ref 10) [10, bytes: 40]
    # Total size of variables: 3274634
    # Total bytes of variables: 13098536
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import os, sys
import seaborn as sns
from ml_platform.mxnet.metadata import MetaInfo

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

if sys.argv[1] == 'bert':
    RST_DIR="/Users/hhp/0/git/byteprofile-analysis/data/data_20200824/20200824_03"
else:
    RST_DIR="/Users/hhp/0/git/byteprofile-analysis/data/data_20200824/20200824_02"
meta_info = MetaInfo(os.path.join(RST_DIR, "host0/0"))

# is_show = [True, True, False, True, False, False, False, False]
is_show = None
GFLOPS_FP32 = 1
GFLOPS_FP16 = 2

# BATCHSIZE_THESHOLD = [4, 8, 64, 64]
BATCHSIZE_THESHOLD = 4
VAR_THREHOLD = 0.2
train_x = train_y = None
test_x = test_y = None
TRAIN_PERCENT = 1
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
            else:
                batchsize = int(lines[idx].split("--batch-size")[1].split("--")[0])
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
    w = math.ceil(math.sqrt(cnt))
    h = math.ceil(cnt / w)
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
else:
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
"FW.resnetv10_stage2_conv10",
"FW.resnetv10_stage2_conv11",
"FW.resnetv10_stage2_conv12",
"FW.resnetv10_stage2_conv2",
"FW.resnetv10_stage2_conv3",
"FW.resnetv10_stage2_conv4",
"FW.resnetv10_stage2_conv5",
"FW.resnetv10_stage2_conv6",
"FW.resnetv10_stage2_conv7",
"FW.resnetv10_stage2_conv8",
"FW.resnetv10_stage2_conv9",
"FW.resnetv10_stage3_conv0",
"FW.resnetv10_stage3_conv1",
"FW.resnetv10_stage3_conv10",
"FW.resnetv10_stage3_conv11",
"FW.resnetv10_stage3_conv12",
"FW.resnetv10_stage3_conv13",
"FW.resnetv10_stage3_conv14",
"FW.resnetv10_stage3_conv15",
"FW.resnetv10_stage3_conv16",
"FW.resnetv10_stage3_conv17",
"FW.resnetv10_stage3_conv18",
"FW.resnetv10_stage3_conv2",
"FW.resnetv10_stage3_conv3",
"FW.resnetv10_stage3_conv4",
"FW.resnetv10_stage3_conv5",
"FW.resnetv10_stage3_conv6",
"FW.resnetv10_stage3_conv7",
"FW.resnetv10_stage3_conv8",
"FW.resnetv10_stage3_conv9",
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
OP_SHORT_LABELS = OP_LABELS
DOTS = ['.-', '^--', 'x-']
# for n in NAMELIST_32:
#     if "FW" in n and "reshape" not in n and "embedding" in n:
#         print(n)
# raise


### model_size[node_name][S_mul, S_add, ...][len= # of batch value]
model_size = np.array([list(zip(*[meta_info.ret_mx_metadata(op_name, batch_size=b)[1:] for b in BATCH_LIST_VALUE])) for op_name in OP_NAMES])
# model_size = np.array([
#         list(zip(*[infoOfConv(b, 32, DEFAULT_KENREL_SIZE, 32, 32) for b in BATCH_LIST_VALUE])),
#         list(zip(*[infoOfConv(b, 16, DEFAULT_KENREL_SIZE, 64, 64) for b in BATCH_LIST_VALUE])),
#         list(zip(*[infoOfConv(b, 8, DEFAULT_KENREL_SIZE, 128, 128) for b in BATCH_LIST_VALUE])),
#         list(zip(*[infoOfConv(b, 4, DEFAULT_KENREL_SIZE, 256, 256) for b in BATCH_LIST_VALUE])),
#     ])

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

def cost_func(xs, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3):
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
    '''
    gflops = xs[0]
    S_mul = xs[1]
    S_add = xs[2]
    # intensity = S_mul / xs[3] + xs[4] + xs[5]
    intensity = 1
    wei_S_all = a3 * xs[3] + a4 * xs[4] + a5 * xs[5]
    wei_S_all2 = a6 * xs[3] + a7 * xs[4] + a8 * xs[5]
    return intensity * (a1 * S_mul + b1) / (a2 * gflops + b2) + (wei_S_all / gflops + b3 + gflops * wei_S_all2) / intensity

def wrap_curve_fit(xs, ys):
    assert isinstance(xs, list) and isinstance(ys, list)
    return curve_fit(time2batch_size, [0] + xs, [0] + ys)

lower_bounds = tuple([0]*8 + [-np.inf]*3)
# lower_bounds = tuple([0]*4 + [-np.inf]*1)
up_bounds = tuple(len(lower_bounds) * [np.inf])
p0=[0]*len(lower_bounds)
FIT_FUNC = cost_func

def collect_data(is_show):
    all_data_dict = {}

    def __record_xdata(S_mul, S_add, S_in, S_out, S_wei, gflops, avg, op_type):
        if op_type not in all_data_dict:
            all_data_dict[op_type] = [[], [], [], [], [], [], []]
        all_data_dict[op_type][0].append(avg)
        all_data_dict[op_type][1].append(gflops)
        all_data_dict[op_type][2].append(S_mul)
        all_data_dict[op_type][3].append(S_add)
        all_data_dict[op_type][4].append(S_in)
        all_data_dict[op_type][5].append(S_out)
        all_data_dict[op_type][6].append(S_wei)

    for i in range(len(OP_NAMES)):
        if is_show is None or is_show[i]:
            for b in BATCH_LIST_VALUE:
                ### filter
                if b <= BATCHSIZE_THESHOLD:
                    continue

                ### collect data
                op_type, S_mul, S_add, S_in, S_out, S_wei = meta_info.ret_mx_metadata(OP_NAMES[i], batch_size=b)

                idx_in_32 = NAMELIST_32.index(OP_NAMES[i])
                var_ = VAR_32["B=%d"%b][idx_in_32] if "B=%d"%b in VAR_32 else 0
                avg_ = DATA_32["B=%d"%b][idx_in_32]
               	if (var_ / avg_) <= VAR_THREHOLD:
                    __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP32, avg_, "Conv2D")

                if NAMELIST_16 is not None:
                    idx_in_16 = NAMELIST_16.index(OP_NAMES[i])
                    var_ = VAR_16["B=%d"%b][idx_in_16] if "B=%d"%b in VAR_16 else 0
                    avg_ = DATA_16["B=%d"%b][idx_in_16]
                    if (var_ / avg_) <= VAR_THREHOLD:
                        __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP16, avg_, "Conv2D")
            # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP32, 0, "Conv2D")
            # __record_xdata(0, 0, 0, 0, 0, GFLOPS_FP16, 0, "Conv2D")


    ### all_data[S_mul, ...][# of nodes * # of batch size values]
    all_data = np.array(all_data_dict["Conv2D"])
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

    print("Collect training data - X:{}, Y:{}, test data - X:{}, Y:{}".format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))


def fit_with_S_cal_gflops():
    _train_x = np.transpose(train_x)
    _train_y = np.transpose(train_y).flatten()
    popt, pcov = curve_fit(FIT_FUNC, _train_x, _train_y, 
        bounds=(lower_bounds, up_bounds), p0=p0, maxfev=10000)
    return popt, pcov

collect_data(is_show)
popt, pcov = fit_with_S_cal_gflops()
print(popt)
UNIT_LEN = len(BATCH_LIST_VALUE)

def plot_intensity2flops():
    assert not is_show[4]
    intensity = xdata[1] / (xdata[3] + xdata[4] + xdata[5])
    flops = xdata[1] / ydata

    from mpl_toolkits.mplot3d import Axes3D
    plt.figure(num=1, figsize=(8, 6))
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(intensity, xdata[0], flops)
    plt.xlabel("arithmetic intensity")
    plt.ylabel("precision")

    ax = plt.subplot(122)
    index32 = np.where(xdata[0,:]==GFLOPS_FP32)
    index16 = np.where(xdata[0,:]==GFLOPS_FP16)
    intensity32 = intensity[index32]
    flops_32_to_16 = flops[index32] / flops[index16]
    ax.plot(intensity32, flops_32_to_16)
    plt.xlabel("arithmetic intensity")
    plt.ylabel("flops 32 / flops 16")
    plt.show()

def plot_3d_fit_result():
    from mpl_toolkits.mplot3d import Axes3D
    xdata = np.array([gflopsList, S_mul_list, S_in_list, S_out_list, S_wei_list])
    pred = FIT_FUNC(xdata, *popt)
    plt.figure(num=1, figsize=(8, 6))
    fig = plt.figure()
    fig_idx = 0

    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
    plt.legend()
    plt.xlabel("Sin Conv1 32")
    plt.ylabel("Swei")
    fig_idx += 1

    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
    plt.legend()
    plt.xlabel("Sin Conv1 16")
    plt.ylabel("Swei")
    fig_idx += 1

    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
    plt.legend()
    plt.xlabel("Sin Conv2 32")
    plt.ylabel("Swei")
    fig_idx += 1

    ax = fig.add_subplot(224, projection='3d')
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
    ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
        pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
    plt.legend()
    plt.xlabel("Sin Conv2 16")
    plt.ylabel("Swei")
    fig_idx += 1

    plt.show()

def plot_2d_fit_result(is_show):
    plt.figure(num=1, figsize=(8, 6))

    ratio_sum = []
    # cnt = sum([int(i) for i in is_show])
    cnt = len(OP_NAMES)
    fig_base, fig_idx = init_fig_base(cnt)

    clrs = sns.color_palette("husl", 5)

    def __plot(op_id, fig_base, fig_idx, only_16=False):
        x_axis_names = ["Batch Size (B)", "S_mul", "S_add", "S_in", "S_out", "S_weight"]
        x_axis_idx = 0
        xaxis = [b if x_axis_idx == 0 else meta_info.ret_mx_metadata(OP_NAMES[op_id], batch_size=b)[x_axis_idx] for b in BATCH_LIST_VALUE]

        ax = plt.subplot(fig_base)
        if not only_16:
            avgs = batchsize2avg(NAMELIST_32.index(OP_NAMES[op_id]))
            ax.plot(xaxis, avgs, marker='.', label=OP_SHORT_LABELS[op_id]+"_fp32", c=clrs[0])
            try:
                stds = batchsize2sdt(NAMELIST_32.index(OP_NAMES[op_id]))
                ax.fill_between(xaxis, avgs+stds, avgs-stds, alpha=0.3, facecolor=clrs[0])
            except KeyError:
                pass

            avgs_pred = FIT_FUNC(np.concatenate((np.array([[GFLOPS_FP32]*len(model_size[fig_idx//2, 0, :])]), model_size[fig_idx//2, :, :]), axis=0), *popt)
            ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp32"+"_pred", c=clrs[1])
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
            avgs_pred = FIT_FUNC(np.concatenate((np.array([[GFLOPS_FP16]*len(model_size[fig_idx//2, 0, :])]), model_size[fig_idx//2, :, :]), axis=0), *popt)
            ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp16"+"_pred", c=clrs[3])
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

    for op_id in range(cnt):
        # if is_show[op_id]:
        # op_id = 2
        fig_base, fig_idx = __plot(op_id, fig_base, fig_idx)
    
    print("average error: %f %%"%(sum(ratio_sum)/len(ratio_sum)))
    plt.show()

def test():
    _test_x = np.transpose(test_x)
    _test_y = np.transpose(test_y).flatten()
    avgs_pred = FIT_FUNC(_test_x, *popt)
    diff, ratio = predict_error(_test_y, avgs_pred)
    print("average error: %f %%"%(float(ratio.split("%")[0])))

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


# plot_intensity2flops()
# plot_2d_fit_result(is_show)
test()
# test_speedup_ratio()




