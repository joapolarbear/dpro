import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from logger_utils import Singleton, SingleLogger
from trace_utils import parse_op_name
import arg_utils
args_ = arg_utils.SingleArg().args

if args_.platform == "MXNET":
    from ml_platform.mxnet.metadata import MetaInfo, FULL_HEADERS, OP_HYPER_PARAMETERS, BASE_HEADER_LEN
    SingleLogger().info("Use MXNET metadata")
elif args_.platform == "TENSORFLOW":
    from ml_platform.tensorflow.metadata import MetaInfo, FULL_HEADERS, OP_HYPER_PARAMETERS, BASE_HEADER_LEN
    SingleLogger().info("Use TENSORFLOW metadata")
else:
    raise NotImplementedError()
from cost_model.gpu_models_info import ret_gpu_config, gpu_filter

BATCHSIZE_THESHOLD = 4
BATCHSIZE_UPPER = 1e6
STDDEV_THREHOLD = 0.2
AVG_THREHOLD = 0
TRAIN_PERCENT = 0.9
FP32_OR_FP16 = (False, True)

SYSNAME = "dPRO"

## the first two dim are avg, G.
METANAME = FULL_HEADERS["base"][2:]
# METANAME = ["S_mul", "S_add", "S_in", "S_out", "S_wei"]


def str2list(_list, dtype=str):
    assert dtype in [int, float, str]
    elems = _list.split("[")[1].split("]")[0].split(", ")

    if dtype == str:
        return [str(e.split("'")[1]) for e in elems]
    else:
        return [dtype(e) for e in elems]

def init_fig_base(cnt):
    h = math.ceil(math.sqrt(cnt))
    w = math.ceil(cnt / h)
    fig_base = w * 100 + h * 10 + 1
    return fig_base, 0

def print_array_values(a, idx):
    ''' Used to check the values in an array
    Parameters
    ----------
    a : array-like, shape = (n_samples, n_features)
    idx: integer, target index of the second dimension
    '''
    value_dict = []
    for i in range(a.shape[0]):
        if a[i, idx] not in value_dict:
            value_dict.append(a[i, idx])
            print(a[i, idx])

class BasicLoader:
    def gen_train_test_data(self, target_optype_, verbose=True):
        ''' Collect the training data and test data from the source data for a specific optype,
        and perform some filtering
        -----
        Return:
            self.all_data_dict: shape = (n_samples, n_features)
        '''
        all_data = np.array(self.all_data_dict[target_optype_]).astype(np.float)

        ### all_data[# of nodes * # of batch size values][avg, G, S_mul, ...]
        ###  ==> shape = (n_samples, n_features)
        all_data = np.transpose(all_data)

        # ### the order has not been changed
        # fp32_data = all_data[all_data[:, 1] == self.gpu_config.flops_fp32]
        # fp32_data = np.split(fp32_data, [1], axis=1)
        # self.fp32_x, self.fp32_y = fp32_data[1], fp32_data[0]
        # fp16_data = all_data[all_data[:, 1] == self.gpu_config.flops_fp16]
        # fp16_data = np.split(fp16_data, [1], axis=1)
        # self.fp16_x, self.fp16_y = fp16_data[1], fp16_data[0]
        # if verbose:
        #     SingleLogger().info("Collect fp32 data - X:{}, Y:{}, fp16 data - X:{}, Y:{}".format(
        #         self.fp32_x.shape, self.fp32_y.shape, self.fp16_x.shape, self.fp16_y.shape))

        # print_array_values(all_data, 10)

        ### normalize data
        ### record the max value of each dimension
        self.max_of_each_dim = []
        if "use_bias" in FULL_HEADERS[target_optype_]:
            for i in range(0, all_data.shape[1]):
                if i == FULL_HEADERS[target_optype_].index("use_bias"):
                    self.max_of_each_dim.append(1)
                else:
                    self.max_of_each_dim.append(max(all_data[:, i]))
                    all_data[:, i] = all_data[:, i] / max(all_data[:, i])     
        else:
            for i in range(0, all_data.shape[1]):
                self.max_of_each_dim.append(max(all_data[:, i]))
                all_data[:, i] = all_data[:, i] / max(all_data[:, i])
        self.max_of_each_dim = np.array(self.max_of_each_dim)
        if verbose:
            SingleLogger().info("Headers for {}: {}".format(target_optype_, FULL_HEADERS[target_optype_]))
            SingleLogger().info("Maximum value for each dimension: {}".format(self.max_of_each_dim))

        # print_array_values(all_data, 10)

        n_samples = all_data.shape[0]
        n_features = all_data.shape[1]
        np.random.shuffle(all_data)

        ### split data to training data and test data
        mask = np.zeros(n_samples, dtype=bool)
        train_idx = np.random.choice(n_samples, int(TRAIN_PERCENT * n_samples), replace=False)
        mask[train_idx] = True
        train_data = np.split(all_data[mask, :], [1], axis=1)
        if TRAIN_PERCENT >= 1:
            test_data = train_data
        else:
            test_data = np.split(all_data[~mask, :], [1], axis=1)
        self.train_x, self.train_y = train_data[1], train_data[0]
        self.test_x, self.test_y = test_data[1], test_data[0]
        if verbose:
            SingleLogger().info("Collect training data - X:{}, Y:{}, test data - X:{}, Y:{}".format(
                    self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape))
        
        # from dim_reduce import DimReducer
        # dim_reducer = DimReducer(train_x, train_y)
        # dim_reducer.do_reduction(algo=['MDS', 'LLE', 'LDA'])
        # raise

class DataLoader(BasicLoader):
    ''' Load all data, including execution time and metadata of each operator
    '''
    def __init__(self, data_dir, metadata_path, gpu_model):
        ''' Load operator execution time and operator metadata
        Parameters
        ----------
        data_dir: str
            The path where operator execution time is stored
        metadata_path: str
            The path where operator metadata is stored
        gpu_model: str
            GPU Model that this training job runs on
        '''
        self.NAMELIST_32 = None
        self.NAMELIST_16 = None

        self.DATA_32 = {}
        self.DATA_16 = {}
        self.VAR_32 = {}
        self.VAR_16 = {}

        self.BATCH_LIST_VALUE = []

        self.data_dir = data_dir
        self.meta_info = MetaInfo(metadata_path)
        self.gpu_config = ret_gpu_config(gpu_model)

        ### record all names
        with open(os.path.join(self.data_dir, "name.txt"), 'r') as fp:
            lines = fp.read().split("\n")
            if lines[-1] == "":
                lines = lines[:-1]
            for line in lines:
                if "fp32" in line:
                    self.NAMELIST_32 = str2list(line.split(":")[1])
                    if "->" in self.NAMELIST_32[0] and "." in "->" in self.NAMELIST_32[0]:
                        self.NAMELIST_32 = [parse_op_name(n) for n in self.NAMELIST_32]
                elif "fp16" in line:
                    self.NAMELIST_16 = str2list(line.split(":")[1])
                    if "->" in self.NAMELIST_16[0] and "." in "->" in self.NAMELIST_16[0]:
                        self.NAMELIST_16 = [parse_op_name(n) for n in self.NAMELIST_16]
                else:
                    raise

        ### record all execution time/variance grouped by batch size
        with open(os.path.join(self.data_dir, "avg.txt"), 'r') as fp:
            lines = fp.read().split("\n")
            idx = 0
            while idx < len(lines):
                if SYSNAME in lines[idx]:
                    if idx+1 < len(lines) and (SYSNAME in lines[idx+1] or lines[idx+1]==""):
                        ### avoid add addition batch size to self.BATCH_LIST_VALUE
                        idx += 1
                        continue
                    batchsize = int(lines[idx].split("BS=")[1].split(":")[0])
                    if "fp32" in lines[idx]:
                        self.BATCH_LIST_VALUE.append(batchsize)
                        _DATA = self.DATA_32
                        _VAR = self.VAR_32
                    elif "fp16" in lines[idx]:
                        _DATA = self.DATA_16
                        _VAR = self.VAR_16
                    else:
                        raise
                    idx += 1
                    if idx >= len(lines) or SYSNAME not in lines[idx]:
                        _DATA["B=%d"%batchsize] = str2list(lines[idx], dtype=float)
                    else:
                        continue
                    idx += 1
                    if idx >= len(lines) or SYSNAME not in lines[idx]:
                        _VAR["B=%d"%batchsize] = str2list(lines[idx], dtype=float)
                    else:
                        continue
                idx += 1

        self.BATCH_LIST_VALUE = [e for e in self.BATCH_LIST_VALUE if (e >= BATCHSIZE_THESHOLD and e <= BATCHSIZE_UPPER)]
        self.BATCH_LIST_VALUE = sorted(self.BATCH_LIST_VALUE)
        self.BATCH_LIST_STR = ["B=%d"%e for e in self.BATCH_LIST_VALUE]

    def pick_opnames_by_op_type(self, op_type):
        if op_type == "CastToFp16" or op_type == "CastToFp32":
            ret = []
            for n in self.NAMELIST_16:
                if op_type not in n or self.meta_info.ret_metadata(n) is None:
                    continue
                ret.append(n)
            return ret

        return [n for n in self.meta_info.pick_opnames_by_op_type(op_type) if n in self.NAMELIST_32]

    def pick_some_ops(self, target_optype):
        ''' generate metadata for some specific operators
        --------
        target_optype: str
            operator type we want to build a cost model for
        '''
        ### a list of operator names we focus on. e.g., pick all Conv2D OPs in this job
        op_names = self.pick_opnames_by_op_type(target_optype)
        if len(op_names) == 0:
            SingleLogger().warn("No ops found for {}".format(target_optype))
            return op_names
        ### shape = (n_nodes, len(METANAME), n_batchsize_values)
        ### TODO (hphu): useless, can be deleted
        self.model_size = np.array([list(zip(*[self.meta_info.ret_metadata(op_name, batch_size=b) for b in self.BATCH_LIST_VALUE])) for op_name in op_names])
        self.model_raw_info = np.array([list(zip(*[self.meta_info.ret_rawmeta(op_name) for b in self.BATCH_LIST_VALUE])) for op_name in op_names])
        self.intensity = self.model_size[:, 0, :] / (self.model_size[:, 2, :] + self.model_size[:, 3, :] + self.model_size[:, 4, :])
        SingleLogger().info("Source metadata set, shape = (n_ops, n_dim, n_batch): {}".format(self.model_size.shape))
        return op_names

    def bs_avg_stddev(self, op_name, fp16=False):
        ''' Return a array, shape = (n_batch, [BS, avg, dev])
        '''
        _NAMELIST = self.NAMELIST_16 if fp16 else self.NAMELIST_32
        index = _NAMELIST.index(op_name)
        avg = self.batchsize2avg(index, fp16=fp16)
        stddev = self.batchsize2dev(index, fp16=fp16)
        return np.array([self.BATCH_LIST_VALUE,avg,stddev]).T

    def batchsize2avg(self, index, fp16=False):
        avg = []
        _DATA = self.DATA_16 if fp16 else self.DATA_32
        for e in self.BATCH_LIST_STR:
            avg.append(_DATA[e][index])
        return avg

    def batchsize2dev(self, index, fp16=False):
        vars_ = []
        _VAR = self.VAR_16 if fp16 else self.VAR_32
        for e in self.BATCH_LIST_STR:
            vars_.append(_VAR[e][index])
        return np.sqrt(np.array(vars_))

    def collect_all_data(self, op_names_, multi_data_dict=None):
        ''' Collect all data into one array from the source data dict and perform some filtering
        -----
        Return:
            self.all_data_dict: shape = (n_features, n_samples)
        '''
        if multi_data_dict is None:
            self.all_data_dict = {}
            _all_data_dict = self.all_data_dict
        else:
            ### Used to collect data from multiple dataloaders
            _all_data_dict = multi_data_dict

        def __record_xdata(metadata, gflops, avg, op_type, addition=None, batch_size=None):
            """ record one data sample 
            Parameters
            ----------
            metadata: list
            """
            S_mul, S_add, S_in, S_out, S_wei = metadata
            if op_type not in _all_data_dict:
                _all_data_dict[op_type] = [[], [], [], [], [], [], []]
            _all_data_dict[op_type][0].append(avg)
            _all_data_dict[op_type][1].append(gflops)
            _all_data_dict[op_type][2].append(S_mul)
            _all_data_dict[op_type][3].append(S_add)
            _all_data_dict[op_type][4].append(S_in)
            _all_data_dict[op_type][5].append(S_out)
            _all_data_dict[op_type][6].append(S_wei)

            ### Dynamically add hyperparametes to the dataset
            if addition is not None and isinstance(addition, list):
                ### update the value of batch size
                batch_size_idx = OP_HYPER_PARAMETERS[op_type].index('B') if "B" in OP_HYPER_PARAMETERS[op_type] else -1
                for idx, e in enumerate(addition):
                    value = addition[idx] if idx != batch_size_idx else batch_size
                    if idx+BASE_HEADER_LEN >= len(_all_data_dict[op_type]):
                        _all_data_dict[op_type].append([value])
                    else:
                        _all_data_dict[op_type][idx+BASE_HEADER_LEN].append(value)

        for i in range(len(op_names_)):
            for b in self.BATCH_LIST_VALUE:
                ### filter
                if b <= BATCHSIZE_THESHOLD:
                    continue

                ### collect data
                op_type = self.meta_info.parse_op_type(op_names_[i])
                metadata = self.meta_info.ret_metadata(op_names_[i], batch_size=b)

                intensity = metadata[0] / (metadata[2] + metadata[3] + metadata[4])

                if FP32_OR_FP16[0] and not "CastTo" in op_names_[i]:
                    ### CastTo operators only exist in FP16 trace files
                    idx_in_32 = self.NAMELIST_32.index(op_names_[i])
                    var_ = self.VAR_32["B=%d"%b][idx_in_32] if "B=%d"%b in self.VAR_32 else 0
                    avg_ = self.DATA_32["B=%d"%b][idx_in_32]
                    if avg_ >= AVG_THREHOLD and (np.sqrt(var_) / avg_) <= STDDEV_THREHOLD:
                        ### [H, W, C, R, S, P, Q, K, batch_size]
                        raw_meta = self.meta_info.ret_rawmeta(op_names_[i])
                        __record_xdata(metadata, self.gpu_config.flops_fp32, avg_, op_type, addition=raw_meta, batch_size=b)

                if FP32_OR_FP16[1]:
                    idx_in_16 = self.NAMELIST_16.index(op_names_[i])
                    var_ = self.VAR_16["B=%d"%b][idx_in_16] if "B=%d"%b in self.VAR_16 else 0
                    avg_ = self.DATA_16["B=%d"%b][idx_in_16]
                    if avg_ >= AVG_THREHOLD and (np.sqrt(var_) / avg_) <= STDDEV_THREHOLD:
                        raw_meta = self.meta_info.ret_rawmeta(op_names_[i])
                        __record_xdata(metadata, self.gpu_config.flops_fp16, avg_, op_type, addition=raw_meta, batch_size=b)

            # _raw_meta =[0]*len(raw_meta)
            # __record_xdata(0, 0, 0, 0, 0, self.gpu_config.flops_fp32, 0, "Conv2D", addition=_raw_meta)
            # __record_xdata(0, 0, 0, 0, 0, self.gpu_config.flops_fp16, 0, "Conv2D", addition=_raw_meta)
            # __record_xdata(0, 0, 0, 0, 0, self.gpu_config.flops_fp32, 0, "Conv2D", addition=None)
            # __record_xdata(0, 0, 0, 0, 0, self.gpu_config.flops_fp16, 0, "Conv2D", addition=None)

    def collect_data(self, op_names_, target_optype_, verbose=True):
        ''' Collect the training data and test data from the source data for a specific optype,
        and perform some filtering
        -----
        Return:
            self.all_data_dict: shape = (n_features, n_samples)
        '''
        self.collect_all_data(op_names_)
        self.gen_train_test_data(target_optype_, verbose=verbose)
        return self.train_x, self.train_y, self.test_x, self.test_y

    def plot_group_by_op(self, op_names, op_labels, op_idxs=None, xaxis='B', yaxiss=['avg']):
        if op_idxs is None:
            op_idxs = np.arange(len(op_names))  
        op_num = len(op_idxs)
        assert op_num <= 9, "Too many ops to visual analyze, {} ops are given".format(op_num) 

        fig_base, fig_idx = init_fig_base(op_num)
        plt.figure(num=1, figsize=(8, 6))
        clrs = sns.color_palette("husl", 5)

        def __fetch_data(_axis):
            dev32 = dev16 = None
            if _axis == 'avg':
                data32 = self.batchsize2avg(self.NAMELIST_32.index(op_names[op_id]))
                dev32 = self.batchsize2dev(self.NAMELIST_32.index(op_names[op_id]))
                data16 = self.batchsize2avg(self.NAMELIST_16.index(op_names[op_id]), fp16=True)
                dev16 = self.batchsize2dev(self.NAMELIST_16.index(op_names[op_id]), fp16=True)
            elif _axis in METANAME:
                data32 = self.model_size[op_id, METANAME.index(_axis), :]
                data16 = None
            elif _axis == 'intensity':
                data32 = intensity[op_id]
                data16 = None
            elif _axis == 'flops':
                avgs_32 = self.batchsize2avg(self.NAMELIST_32.index(op_names[op_id]))
                avgs_16 = self.batchsize2avg(self.NAMELIST_16.index(op_names[op_id]), fp16=True)
                data32 = self.model_size[op_id, 0, :] / np.array(avgs_32)
                data16 = self.model_size[op_id, 0, :] / np.array(avgs_16)
            elif _axis == 'B':
                data32 = self.BATCH_LIST_VALUE
                data16 = None
            else:
                raise ValueError("axis choice should be in {} or avg, intensity, flops. {} is given".format(METANAME, _axis))

            return data32, data16, dev32, dev16

        def __plot(op_id, fig_base):
            ax = plt.subplot(fig_base)

            ### select x-axis
            x_axis, _, _, _ = __fetch_data(xaxis)
            if isinstance(yaxiss, list):
                for yaxis in yaxiss:
                    ### select y-axis
                    data32, data16, dev32, dev16 = __fetch_data(yaxis)
                    if data16 is None:
                        ax.plot(x_axis, data32, marker='.', label=op_labels[op_id] + "_" + yaxis, c=clrs[0])
                    else:
                        ax.plot(x_axis, data32, marker='.', label=op_labels[op_id] + "_" + yaxis + "_fp32", c=clrs[0])
                        if dev32 is not None:
                            ax.fill_between(x_axis, data32+dev32, data32-dev32, alpha=0.3, facecolor=clrs[0])
                        ax.plot(x_axis, data16, marker='^', label=op_labels[op_id] + "_" + yaxis + "_fp16", c=clrs[1])
                        if dev16 is not None:
                            ax.fill_between(x_axis, data16+dev16, data16-dev16, alpha=0.3, facecolor=clrs[1])
            else:
                yaxis = yaxiss
                data32, data16, dev32, dev16 = __fetch_data(yaxis)
                if data16 is None:
                    ax.plot(x_axis, data32, marker='.', label=op_labels[op_id], c=clrs[0])
                    if dev32 is not None:
                        ax.fill_between(x_axis, data32+dev32, data32-dev32, alpha=0.3, facecolor=clrs[0])
                else:
                    ax.plot(x_axis, data32, marker='.', label=op_labels[op_id] + "_fp32", c=clrs[0])
                    if dev32 is not None:
                        ax.fill_between(x_axis, data32+dev32, data32-dev32, alpha=0.3, facecolor=clrs[0])
                    ax.plot(x_axis, data16, marker='^', label=op_labels[op_id] + "_fp16", c=clrs[1])
                    if dev16 is not None:
                        ax.fill_between(x_axis, data16+dev16, data16-dev16, alpha=0.3, facecolor=clrs[1])

            plt.legend()
            AXIS_NAMES = {"B":"Batch Size", "avg": "Duration (ms)"}
            plt.ylabel(yaxis if yaxis not in AXIS_NAMES else AXIS_NAMES[yaxis])
            plt.xlabel(xaxis if xaxis not in AXIS_NAMES else AXIS_NAMES[xaxis])
            # plt.ylim(0., 0.2)
            return fig_base+1

        for op_id in op_idxs:
            fig_base = __plot(op_id, fig_base)

        plt.show()

    def plot_B2cost(self, op_names, op_labels, op_idxs=None):
        self.plot_group_by_op(op_names, op_labels, op_idxs, xaxis='B', yaxiss='avg')

    def plot_B2intensity(self, op_names, op_labels, op_idxs):
        self.plot_group_by_op(op_names, op_labels, op_idxs, xaxis='B', yaxiss='intensity')

    def plot_intensity2flops(self, op_names, op_labels, op_idxs):
        self.plot_group_by_op(op_names, op_labels, op_idxs, xaxis='intensity', yaxiss='flops')

    def plot_B2metadata(self, op_names, op_labels, op_idxs, metric):
        assert metric in METANAME
        plot_group_by_op(op_names, op_labels, op_idxs, xaxis='B', yaxiss=metric)

    def plot_avg2distribution(self, target_optype_):
        plt.figure(num=1, figsize=(8, 6))
        THRESHOLD = 0.1
        all_data = np.array(self.all_data_dict[target_optype_]).astype(np.float)
        all_avgs = np.copy(all_data[FULL_HEADERS[target_optype_].index("avg")])

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

        x1, y1, ret1 = _gen(all_avgs)
        plt.plot(x1, y1, label="%f: %6.4f %%"%(THRESHOLD, ret1))
        plt.legend()
        plt.ylabel('Cumulative Distribution (%)')
        plt.xlabel("Execution time (ms)")

        plt.show()

    def plot_max_comp_mem(self, target_optype_):
        all_data = np.array(self.all_data_dict[target_optype_]).astype(np.float)
        if 'use_bias' in FULL_HEADERS[target_optype_]:
            comp = all_data[FULL_HEADERS[target_optype_].index("S_mul")] + \
                all_data[FULL_HEADERS[target_optype_].index("S_out")] * all_data[FULL_HEADERS[target_optype_].index("use_bias")]
        else:
            comp = all_data[FULL_HEADERS[target_optype_].index("S_mul")]

        comp = comp / max(comp)

        mem = all_data[FULL_HEADERS[target_optype_].index("S_in")] + all_data[FULL_HEADERS[target_optype_].index("S_out")] \
            + all_data[FULL_HEADERS[target_optype_].index("S_wei")]
        mem = mem / max(mem)

        avgs = all_data[FULL_HEADERS[target_optype_].index("avg")]
        avgs = avgs / max(avgs)

        fig = plt.figure(num=1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(comp, mem, avgs, label='Norm Execution Time')
        ax.scatter(comp, mem, np.maximum(comp, mem), label='Max')
        plt.xlabel("Computation Term")
        plt.ylabel("Memory Term")
        plt.legend()
        plt.show()

    def plot_B2cost_of_cast():
        raise NotImplementedError()
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


class MultiDataLoader(BasicLoader):
    def __init__(self, data_dirs, metadata_paths, gpu_models):
        assert len(data_dirs) == len(metadata_paths)
        assert len(data_dirs) == len(gpu_models)
        self.dataloaders = [DataLoader(data_dirs[i], metadata_paths[i], gpu_models[i]) for i in range(len(data_dirs))]

    def collect_data(self, target_optype_, verbose=True):
        _all_data_dict = {}
        for _data_loader in self.dataloaders:
            # print(_data_loader.gpu_config.name)
            op_names_ = _data_loader.pick_opnames_by_op_type(target_optype_)
            if len(op_names_) > 0 and gpu_filter(_data_loader.gpu_config.name):
                _data_loader.collect_all_data(op_names_, multi_data_dict=_all_data_dict)

        self.all_data_dict = _all_data_dict
        self.gen_train_test_data(target_optype_, verbose=verbose)
        return self.train_x, self.train_y, self.test_x, self.test_y
        











