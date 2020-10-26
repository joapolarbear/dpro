"""
Platform: On one V100 GPU, single machine
Framework: Tensorflow 1.14, CUDA 10.2
"""
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from dataloader import DataLoader, MultiDataLoader, init_fig_base, FP32_OR_FP16, FULL_HEADERS, GFLOPS_FP32, GFLOPS_FP16
import seaborn as sns
from amp_cost_model import CurveFiter

OPTYPES = ["conv", "dense"]

if sys.argv[1] == 'bert':
    RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200824", "20200824_03")
elif sys.argv[1] == 'resnet':
    TARGET_OPTYPE = OPTYPES[0]
    ### V100
    RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200824", "20200824_02")
    ### 1080Ti
    # RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200930", "20200930_05")
elif sys.argv[1] == 'dense':
    TARGET_OPTYPE = OPTYPES[1]
    # RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200917", "20200917_06")
    # RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200924", "20200924_01")
    ### 1080Ti
    # RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200930", "20200930_03")
    # RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200930", "20200930_06")
    RST_DIR=os.path.join("/Users/hhp/0/git/byteprofile-analysis/data/data_20200930", "20201020_01")
else:
    raise

### below shows OP_NAMES we focus on for each type of model
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
elif sys.argv[1] == 'dense':
    OP_NAMES = [
"FW.densemodel0_dense0",
"FW.densemodel0_dense1",
"FW.densemodel0_dense2",
"FW.densemodel0_dense3",
"FW.densemodel0_dense4",
"FW.densemodel0_dense5",
"FW.densemodel0_dense6",
"FW.densemodel0_dense7",
"FW.densemodel0_dense8",
"FW.densemodel0_dense9",
"FW.densemodel0_dense10",
"FW.densemodel0_dense11",
"FW.densemodel0_dense12",
"FW.densemodel0_dense13",
"FW.densemodel0_dense14",
"FW.densemodel0_dense15",
"FW.densemodel0_dense16",
"FW.densemodel0_dense17",
"FW.densemodel0_dense18",
"FW.densemodel0_dense19",
"FW.densemodel0_dense20",
"FW.densemodel0_dense21",
"FW.densemodel0_dense22",
"FW.densemodel0_dense23",
"FW.densemodel0_dense24",
"FW.densemodel0_dense25",
"FW.densemodel0_dense26",
"FW.densemodel0_dense27",
"FW.densemodel0_dense28",
"FW.densemodel0_dense29",
"FW.densemodel0_dense30",
"FW.densemodel0_dense31",
"FW.densemodel0_dense32",
"FW.densemodel0_dense33",
"FW.densemodel0_dense34",
"FW.densemodel0_dense35",
"FW.densemodel0_dense36",
"FW.densemodel0_dense37",
"FW.densemodel0_dense38",
"FW.densemodel0_dense39",
"FW.densemodel0_dense40",
"FW.densemodel0_dense41",
"FW.densemodel0_dense42",
"FW.densemodel0_dense43",
"FW.densemodel0_dense44",
"FW.densemodel0_dense45",
"FW.densemodel0_dense46",
"FW.densemodel0_dense47",
"FW.densemodel0_dense48",
"FW.densemodel0_dense49",
"FW.densemodel0_dense50",
]
    OP_LABELS = ["".join(n.split("densemodel0_")[1]) for n in OP_NAMES]
else:
    raise
OP_SHORT_LABELS = OP_LABELS

data_ld = DataLoader(data_dir=RST_DIR, metadata_path=os.path.join(RST_DIR, "host0/0"), model=sys.argv[1])
data_ld.pick_some_ops(OP_NAMES)
####################################################################################################
#############################        Start to Fit          #########################################
####################################################################################################
THRESHOLD = 0.00 # in ms
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
    if len(_list) == 0:
        return None, "Original time is too small. Ignore!!!"

    diff = np.abs(_list_pred - _list) / _list
    return diff, "%f %%"%(np.average(diff * 100))

def plot_2d_fit_result(predictor, op_idxs=None):
    assert isinstance(data_ld, DataLoader)
    plt.figure(num=1, figsize=(8, 6))

    ratio_sum = []
    if op_idxs is None:
        op_idxs = np.arange(len(OP_NAMES))
    cnt = len(op_idxs)
    fig_base, fig_idx = init_fig_base(cnt)

    clrs = sns.color_palette("husl", 5)

    max_avg = data_ld.max_of_each_dim[FULL_HEADERS[TARGET_OPTYPE].index('avg')]

    def __plot(op_id, fig_base, fig_idx):
        x_axis_names = ["Batch Size", "S_mul", "S_add", "S_in", "S_out", "S_weight"]
        x_axis_idx = 0
        xaxis = [b if x_axis_idx == 0 else data_ld.meta_info.ret_mx_metadata(OP_NAMES[op_id], batch_size=b)[x_axis_idx] for b in data_ld.BATCH_LIST_VALUE]
        model_size_32 = np.concatenate((np.array([[GFLOPS_FP32]*len(data_ld.model_size[op_id, 0, :])]), data_ld.model_size[op_id, :, :], data_ld.model_raw_info[op_id, :, :]), axis=0)
        model_size_16 = np.concatenate((np.array([[GFLOPS_FP16]*len(data_ld.model_size[op_id, 0, :])]), data_ld.model_size[op_id, :, :], data_ld.model_raw_info[op_id, :, :]), axis=0)
        
        model_size_32 = np.divide(model_size_32, data_ld.max_of_each_dim[1:, None])
        model_size_16 = np.divide(model_size_16, data_ld.max_of_each_dim[1:, None])

        ax = plt.subplot(fig_base)

        if FP32_OR_FP16[0]:
            avgs = np.array(data_ld.batchsize2avg(data_ld.NAMELIST_32.index(OP_NAMES[op_id]))) / max_avg
            ax.plot(xaxis, avgs, marker='.', label=OP_SHORT_LABELS[op_id]+"_fp32", c=clrs[0])
            try:
                stds = np.array(data_ld.batchsize2dev(data_ld.NAMELIST_32.index(OP_NAMES[op_id]))) / max_avg
                ax.fill_between(xaxis, avgs+stds, avgs-stds, alpha=0.3, facecolor=clrs[0])
            except KeyError:
                pass

            avgs_pred = predictor.predict(model_size_32)
            # ax.fill_between(xaxis, predictor.predict(model_size_32, *popt+perr), predictor.predict(model_size_32, *popt-perr), alpha=0.3, facecolor=clrs[1])
            diff, ratio = predict_error(avgs, avgs_pred)
            ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp32"+"_pred\nerr={}".format(ratio), c=clrs[1])
            print(OP_SHORT_LABELS[op_id]+"_fp32", ratio)
            fig_idx += 1
            if "%" in ratio:
                ratio_sum.append(float(ratio.split("%")[0]))

        if FP32_OR_FP16[1]:
            avgs_16 = np.array(data_ld.batchsize2avg(data_ld.NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)) / max_avg
            ax.plot(xaxis, avgs_16, marker='^', label=OP_SHORT_LABELS[op_id]+"_fp16", c=clrs[2])
            try:
                stds_16 = np.array(data_ld.batchsize2dev(data_ld.NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)) / max_avg
                ax.fill_between(xaxis, avgs_16+stds_16, avgs_16-stds_16, alpha=0.3, facecolor=clrs[2])
            except KeyError:
                pass
            avgs_pred = predictor.predict(model_size_16)
            # ax.fill_between(xaxis, predictor.predict(model_size_16, *popt+perr), predictor.predict(model_size_16, *popt-perr), alpha=0.3, facecolor=clrs[3])
            diff, ratio = predict_error(avgs_16, avgs_pred)
            ax.plot(xaxis, avgs_pred, "--", label=OP_SHORT_LABELS[op_id]+"_fp16"+"_pred\nerr={}".format(ratio), c=clrs[3])
            print(OP_SHORT_LABELS[op_id]+"_fp16", ratio)
            fig_idx += 1
            if "%" in ratio:
                ratio_sum.append(float(ratio.split("%")[0]))
        
        plt.legend(fontsize=8)
        plt.ylabel('Normalized Average Time')
        # plt.ylim(0, 2)
        plt.xlabel(x_axis_names[x_axis_idx])
        return fig_base+1, fig_idx

    for op_id in op_idxs:
        # op_id = 2
        fig_base, fig_idx = __plot(op_id, fig_base, fig_idx)
    
    print("average error: %f %%"%(sum(ratio_sum)/len(ratio_sum)))
    plt.show()

def test_speedup_ratio():
    raise NotImplementedError()
    _fp32_x = np.transpose(fp32_x)
    _fp32_y = np.transpose(fp32_y).flatten()
    _fp32_y_pred = predictor.predict(_fp32_x, *popt)
    diff, ratio = predict_error(_fp32_y, _fp32_y_pred)
    print("fp32 average error: %f %%"%(float(ratio.split("%")[0])))

    _fp16_x = np.transpose(fp16_x)
    _fp16_y = np.transpose(fp16_y).flatten()
    _fp16_y_pred = predictor.predict(_fp16_x, *popt)
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
### Output attribute of ops
def output_rawdata_all_op(attr, default_B=32):
    assert isinstance(data_ld, DataLoader)
    if attr == 'avg':
        for op_name in OP_NAMES:
            print(batchsize2avg(NAMELIST_32.index(op_name))[data_ld.BATCH_LIST_VALUE.index(default_B)])
        return
    elif attr == 'std.dev':
        for op_name in OP_NAMES:
            print(batchsize2dev(NAMELIST_32.index(op_name))[data_ld.BATCH_LIST_VALUE.index(default_B)])
        return
    idx = FULL_HEADERS[TARGET_OPTYPE].index(attr)
    if idx >= len(FULL_HEADERS['base']):
        for op_name in OP_NAMES:
            raw_meta = data_ld.meta_info.ret_mx_rawmeta(op_name, batch_size=32)
            print(raw_meta[idx-len(FULL_HEADERS['base'])])
    elif idx >= 2:
        for op_name in OP_NAMES:
            meta = data_ld.meta_info.ret_mx_metadata(op_name, batch_size=32)
            print(meta[1+idx-2])
# output_rawdata_all_op('R')
# raise

#####################################################################
### Test the error on all OPs
### Cross validation grouped by individual op
def test_grouped_by_individual_op(visual=True):
    idxs = np.arange(40, 49)
    if visual:
        plt.figure(num=1, figsize=(8, 6))
    cnt = len(idxs)
    fig_base, fig_idx = init_fig_base(cnt)
    clrs = sns.color_palette("husl", 5)

    def __plot(op_id, fig_base, fig_idx, err):
        x_axis_names = ["Batch Size", "S_mul", "S_add", "S_in", "S_out", "S_weight"]
        x_axis_idx = 0
        xaxis = [b if x_axis_idx == 0 else data_ld.meta_info.ret_mx_metadata(OP_NAMES[op_id], batch_size=b)[x_axis_idx] for b in data_ld.BATCH_LIST_VALUE]
        if visual:
            ax = plt.subplot(fig_base)

        if FP32_OR_FP16[0]:
            avgs = np.array(data_ld.batchsize2avg(data_ld.NAMELIST_32.index(OP_NAMES[op_id])))
            if visual:
                ax.plot(xaxis, avgs, marker='.', label=OP_SHORT_LABELS[op_id]+"_fp32", c=clrs[0])
                try:
                    stds = np.array(data_ld.batchsize2dev(data_ld.NAMELIST_32.index(OP_NAMES[op_id])))
                    ax.fill_between(xaxis, avgs+stds, avgs-stds, alpha=0.3, facecolor=clrs[0])
                except KeyError:
                    pass

        if FP32_OR_FP16[1]:
            avgs_16 = np.array(data_ld.batchsize2avg(data_ld.NAMELIST_16.index(OP_NAMES[op_id]), fp16=True))
            if visual:
                ax.plot(xaxis, avgs_16, marker='^', label=OP_SHORT_LABELS[op_id]+"_fp16", c=clrs[2])
                try:
                    stds_16 = np.array(data_ld.batchsize2dev(data_ld.NAMELIST_16.index(OP_NAMES[op_id]), fp16=True))
                    ax.fill_between(xaxis, avgs_16+stds_16, avgs_16-stds_16, alpha=0.3, facecolor=clrs[2])
                except KeyError:
                    pass
        if visual:
            plt.legend(fontsize=8)
            plt.ylabel('Average Time (ms)')
            # plt.ylim(0, 2)
            plt.xlabel(x_axis_names[x_axis_idx])
            plt.title("OP ID {} - fitting error: {} %".format(op_id, err))
        return fig_base+1, fig_idx

    for op_id in idxs:
        op_names_ = np.array([np.array(OP_NAMES)[op_id]])
        train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=False)
        pred = CurveFiter(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE], op_type=TARGET_OPTYPE)
        popt, pcov = pred.train()
        err = pred.test(verbose=False)
        print(op_id, op_names_[0], err)
        fig_base, fig_idx = __plot(op_id, fig_base, fig_idx, err)
    if visual:
        plt.show()
# test_grouped_by_individual_op(visual=True)
# raise

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
#         collect_data(op_names_, verbose=False)
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
#     collect_data(op_names_, verbose=False)
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
#     collect_data(op_names_, verbose=False)
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
#     collect_data(op_names_, verbose=False)
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
#     collect_data(op_names_, verbose=False)
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
# 20200903 (Thu)  ******!!!!!******
#####################################################################
### exp id 1: fix the # of ops to 2
### since the variance when # of ops = 2 is very large, see which combinations lead to large error

def try_diff_combination():
    # idxs = np.array([1]) # normalized S_mul = 0.1 
    # idxs = np.array([11, 24, 43]) # normalized S_mul = 0.2
    # idxs = np.array([3, 4, 5, 7, 8, 10, 13, 15, 17, 18, 20, 21, 23, 26, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 45, 47, 49, 50, 52])
    # idxs = np.array([14,27,46]) # normalized S_mul = 0.8
    # # idxs = np.array([2,6,9,12,16,19,22,25,29,32,35,38,41,44,48,51]) # normalized S_mul = 0.9
    # # idxs = np.array([0]) # normalized S_mul = 1 

    # idxs = np.array([1, 2])
    # idxs = np.array([19, 22])

    ### dense
    # idxs = np.arange(0, 9)
    # idxs = np.array([0, 1])
    idxs = np.arange(0, 8)
    # data_ld.plot_group_by_op(OP_NAMES, OP_LABELS, op_idxs=idxs, xaxis='B', yaxiss=['avg'])

    op_names_ = np.array(OP_NAMES)[idxs]
    # # op_names_ = OP_NAMES
    train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)
    pred = CurveFiter(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE], op_type=TARGET_OPTYPE)
    popt, pcov = pred.train()
    print(popt)
    pred.test(verbose=True)

    # 
    plot_2d_fit_result(pred, op_idxs=idxs)
# try_diff_combination()
# raise


# pct = -1
# error_list = []
# idxs_list = []
# for _ in range(100):
#     # idxs = np.random.choice(len(OP_NAMES), 2, replace=False)
#     idxs = np.array([3, 4])
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, verbose=False)
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
# target = 'error'
# force = False
# total = len(OP_NAMES)
# if target == 'error':
#     file_path = os.path.join(RST_DIR, "2ops_cross_error.txt")
#     if not force and os.path.exists(file_path):
#         rst = np.loadtxt(file_path)
#     else:
#         rst = np.zeros((total, total), dtype=np.float)
#         for i in range(total-1):
#             for j in range(i+1, total):
#                 op_names_ = [OP_NAMES[i], OP_NAMES[j]]
#                 collect_data(op_names_, verbose=False)
#                 popt, pcov = fit_with_S_cal_gflops()
#                 error = test(verbose=False)
#                 rst[i][j] = error
#         np.savetxt(file_path, rst)
# else:
#     ### for meta data
#     meta_list = []
#     for op_name in OP_NAMES:
#         _, S_mul, _, S_in, S_out, S_wei = data_ld.meta_info.ret_mx_metadata(op_name, batch_size=32)
#         ### [H, W, C, R, S, P, Q, K, batch_size]
#         raw_meta = data_ld.meta_info.ret_mx_rawmeta(op_name, batch_size=32)
#         if target == 'S_mul':
#             meta_list.append(S_mul)
#         elif target == 'S_in':
#             meta_list.append(S_in)
#         elif target == 'S_out':
#             meta_list.append(S_out)
#         elif target == 'S_wei':
#             meta_list.append(S_wei)
#         elif target == 'stride':
#             meta_list.append(raw_meta[0]/raw_meta[5])
#         elif target == 'kernel_size':
#             meta_list.append(raw_meta[3])
#         else:
#             raise
#     rst = np.zeros((total, total), dtype=np.float)
#     for i in range(total-1):
#         for j in range(i+1, total):
#             rst[i][j] = np.abs(meta_list[i] - meta_list[j])

# rst = np.transpose(rst)
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



#####################################################################
#####################################################################
# 20200909 (Wed)
#####################################################################
### exp id 1: for each op, test whether the weight in the cost function for computation term and communication term

# for idx in range(len(OP_NAMES)):
#     idxs = np.array([idx])
#     op_names_ = np.array(OP_NAMES)[idxs]
#     collect_data(op_names_, verbose=False)
#     rst = []
#     wei1, wei2 = 1, 1
#     popt, pcov = fit_with_S_cal_gflops()
#     rst.append(test(verbose=False))
#     wei1, wei2 = 0, 1
#     popt, pcov = fit_with_S_cal_gflops()
#     rst.append(test(verbose=False))
#     wei1, wei2 = 1, 0
#     popt, pcov = fit_with_S_cal_gflops()
#     rst.append(test(verbose=False))
#     print(rst)

#####################################################################
#####################################################################
# 20200910 (Thu)
#####################################################################
### exp id 1: use DNN to predict
# from amp_cost_model import DNNPredictor
# op_names_ = OP_NAMES
# train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)
# pred = DNNPredictor(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE])
# pred.train()
# # pred.test()
# pred.predict()

### exp id 2: use Bayes to predict
# from amp_cost_model import BayesPredictor
# op_names_ = OP_NAMES
# train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)
# pred = BayesPredictor(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE])
# pred.train()


## exp id 3:
#### group by S_mul
# idxs_list = [
#     np.array([1]),
#     np.array([11, 24, 43]),
#     np.array([3, 4, 5, 7, 8, 10, 13, 15, 17, 18, 20, 21, 23, 26, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 45, 47, 49, 50, 52]),
#     np.array([14,27,46]),
#     np.array([2,6,9,12,16,19,22,25,29,32,35,38,41,44,48,51]),
#     np.array([0]),
#     -1
# ]

#### group by kernel size
# idxs_list = [
#     np.array([1,3,4,5,7,8,10,11,13,14,15,17,18,20,21,23,24,26,27,28,30,31,33,34,36,37,39,40,42,43,45,46,47,49,50,52]),
#     np.array([2,6,9,12,16,19,22,25,29,32,35,38,41,44,48,51]),
#     np.array([0]),
#     -1
# ]

#### grouped by stride
# idxs_list = [
#     np.array([1,2,3,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,47,48,49,50,51,52]),
#     np.array([0,11,14,24,27,43,46]),
#     -1
# ]


### in the first group of different kernel size, more fine-grained
# idxs_list = [
#     np.array([1,3,4,5,7,8,10,11,13,14,15,17,18,20,21,23,24,26,27,28,30,31,33,34,36,37,39,40,42,43,45,46,47,49,50,52]),
#     # stride=1
#     np.array([1,3,4,5,7,8,10,13,15,17,18,20,21,23,26,28,30,31,33,34,36,37,39,40,42,45,47,49,50,52]),
#     # stride=2
#     np.array([11,14,24,27,43,46]),
#     # use_bias
#     np.array([1,3,5,7,8,10,11,13,15,17,18,20,21,23,24,26,28,30,31,33,34,36,37,39,40,42,43,45,47,49,50,52])
# ]

### group by CUDA kernels used
# idxs_list = [
#     np.array([25, 35, 38, 41, 29, 32, 44, 48, 51]),
#     np.array([3, 7, 10, 11, 21, 23, 13, 15, 17, 18, 20, 24, 34, 36, 37, 39, 40, 42, 26, 28, 30, 31, 33, 43, 45, 47, 49, 50, 52]),
#     np.array([1, 5, 8]),
#     np.array([0]),
#     np.array([4, 14, 27, 46]),
#     np.array([2, 6, 9]),
#     np.array([12, 22, 16, 19]),
#     -1
# ]

### dense
idxs_list = [
    np.arange(0,9),
    np.arange(10,20),
    np.arange(20,30),
    np.arange(30,40),
    np.arange(40,50),
    # np.arange(0,10),
    # np.arange(40,50),
    # np.concatenate((np.arange(0,10), np.arange(40,50))),
    # np.arange(10,30),
    # np.array([43]),
    -1
]

for idxs in idxs_list:
    if isinstance(idxs, np.ndarray):
        op_names_ = np.array(OP_NAMES)[idxs]
        print(list(idxs))
    else:
        op_names_ = OP_NAMES
        print("all ops")
    train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)

    # data_ld.plot_max_comp_mem(TARGET_OPTYPE)
    # data_ld.plot_avg2distribution(TARGET_OPTYPE)
    # raise
    pred = CurveFiter(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE], op_type=TARGET_OPTYPE)
    try:
        popt, pcov = pred.train()
    except:
        print(train_x)
        print(idxs)
        raise
    
    print("\t".join([str(e) for e in popt]))
    pred.test(verbose=True)

# raise

#### verify the popts trained with group by kernel with all ops
# op_names_ = OP_NAMES
# train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)
# pred = CurveFiter(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE])
# popts = [
#     [2.6056619411766158, 1.2959646709885562, 0.08360241234592686, 0.06120795436936255, 0.1559150708255634, 0.10354411725669714],
#     [0.9575921242401884,145.0597191195498,0.4319982825529537, 0.43199827997673484,0.15762755558430627,0.07185529379153197]
# ]
# pred.test(verbose=True, popt=popts[0])

#####################################################################
#####################################################################
# 20200917 (Thu)
#####################################################################
### exp id 1: add K^n to W1 term, K^m to W2 term, find the optimal n and m
# op_names_ = OP_NAMES
# train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)
# for ni in range(20):
#     n = (ni + 1) * 0.1
#     errors = []
#     for mi in range(30):
#         m = (mi + 1) * 0.1
#         pred = CurveFiter(train_x, train_y, test_x, test_y, FULL_HEADERS[TARGET_OPTYPE], E1_=n, E2_=m)
#         try:
#             popt, pcov = pred.train()
#         except:
#             print(train_x)
#             print(idxs)
#             raise
#         errors.append(pred.test(verbose=False))
#     print("n={}: {}".format(n, "\t".join([str(e) for e in errors])))


#####################################################################
#####################################################################
# 20201008 (Thu)
#####################################################################
### exp id 1: create a group divider, control the td_len and fe_len

# '''
op_names_ = OP_NAMES
train_x, train_y, test_x, test_y = data_ld.collect_data(op_names_, TARGET_OPTYPE, verbose=True)

from grouper import Grouper, Delimiter
grp = Grouper()

#############################################
### define delimieter
# dels = Delimiter("avg", td_len=0.1, fd_len=0.01, unit_len=0.001)
# dels = Delimiter("S_mul", td_len=0.01, fd_len=0.001, unit_len=0.0001)

### used for euqally divided
# dels = Delimiter("S_mul", max_grp_size=10)

### divided by kernel size, note do not use K, since K denotes the output channel
# dels = [
#     Delimiter("R", td_len=0.1, fd_len=0., unit_len=0.1),
#     Delimiter("G", td_len=0.1, fd_len=0., unit_len=0.1)
# ]

### divided by C_in
dels = [
    # Delimiter("S_mul", td_len=0.01, fd_len=0., unit_len=0.0001),
    Delimiter("C_in", td_len=0.1, fd_len=0., unit_len=0.1),
    Delimiter("C_out", td_len=0.1, fd_len=0., unit_len=0.1),
    Delimiter("G", td_len=0.1, fd_len=0., unit_len=0.1)
]

#############################################
### apply the delimiter
# grp.divide_with_upper(dels, train_x, train_y, test_x, test_y, headers=FULL_HEADERS[TARGET_OPTYPE])
grp.divide_by_len(dels, train_x, train_y, test_x, test_y, headers=FULL_HEADERS[TARGET_OPTYPE])


### train and test
grp.train_test(dels, headers=FULL_HEADERS[TARGET_OPTYPE], op_type=TARGET_OPTYPE)

# '''

