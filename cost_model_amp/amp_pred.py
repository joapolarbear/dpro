import os, sys
import math
import ujson as json
import numpy as np
from scipy.optimize import curve_fit
from trace_utils import *
from ml_platform.tensorflow.metadata import MetaInfo

GFLOPS_FP32 = 1
GFLOPS_FP16 = 2
TRAIN_PERCENT = 0.99
BATCH_LIST_VALUE = 0
VAR_THREHOLD = 0.2


def str2list(_list, dtype=str):
    assert dtype in [int, float, str]
    elems = _list.split("[")[1].split("]")[0].split(", ")

    if dtype == str:
        return [str(e.split("'")[1]) for e in elems]
    else:
        return [dtype(e) for e in elems]

def predict_error(_list, _list_pred):
    _list_pred = np.array(_list_pred)
    _list = np.array(_list)
    diff = np.abs(_list_pred - _list) / _list
    return np.average(diff)

### cost function
LOWER_BOUNDS = tuple([0]*8 + [-np.inf]*3)
UPPER_BOUNDS = tuple(len(LOWER_BOUNDS) * [np.inf])
P0=[0]*len(LOWER_BOUNDS)
def func_pred_time(xs, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3):
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
    # intensity = S_mul / (xs[3] + xs[4] + xs[5])
    intensity = 1
    wei_S_all = a3 * xs[3] + a4 * xs[4] + a5 * xs[5]
    wei_S_all2 = a6 * xs[3] + a7 * xs[4] + a8 * xs[5]
    return intensity * (a1 * S_mul + b1) / (a2 * gflops + b2) + (wei_S_all / gflops + b3 + gflops * wei_S_all2) / intensity

class AMPPredictor:
    def __init__(self, meta_path, cost_model_path):
        self.meta_info = MetaInfo(meta_path)
        with open(cost_model_path, 'r') as fp:
            self.cost_model = json.load(cost_model_path)


    def pred_amp_avg(self, op_name, _avg=None):
        op_type, S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_tf_metadata(op_name)
        try:
            popt = self.cost_model[op_type]["popt"]
        except KeyError as e:
            SingleLogger().error("the AMP cost model for {} has not been built. OP: {}".format(op_type, op_name))
            return
        avg_fp32 = func_pred_time([GFLOPS_FP32, S_mul, S_add, S_in, S_out, S_wei], *popt)
        avg_fp16 = func_pred_time([GFLOPS_FP16, S_mul, S_add, S_in, S_out, S_wei], *popt)
        if _avg is not None:
            return _avg * avg_fp16 / avg_fp32
        else:
            return avg_fp16

    def pre_cast_time(self, op_name):
        op_type, S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_tf_metadata(op_name)
        try:
            popt = self.cost_model["Cast"]["popt"]
        except KeyError as e:
            SingleLogger().error("the AMP cost model for {} has not been built. OP: {}".format("Cast", op_name))
            return
        in_cast = func_pred_time([0, 0, 0, S_in, S_in, 0], *popt)
        out_cast = func_pred_time([0, 0, 0, S_out, S_out, 0], *popt)
        wei_cast = func_pred_time([0, 0, 0, S_wei, S_wei, 0], *popt)
        return in_cast, out_cast, wei_cast

    def quantize(self, dag, op_name):
        in_cast, out_cast, wei_cast = self.pre_cast_time(op_name)
        assert dag.nodes[op_name].get("dtype", "fp32") == "fp32", op_name

        ### handle parent nodes
        for u, v in dag.in_edges(op_name):
            if "AMPCastToFp32" in u:
                ### not the boundary of mixed precision, remove the cast
                prevs = dag.in_edges(u)
                assert len(prevs) == 1, prevs
                dag.add_edge(prevs[0][0], op_name)
                dag.remove_edge(*prevs[0])
            else:
                ### the boundary of mixed precision, add a cast op
                dag.add_edge(u, "%s.AMPCastToFp16"%op_name)
                dag.add_edge("%s.AMPCastToFp16"%op_name, op_name)
                # TODO (huhanpeng): need verify
                dag.nodes["%s.AMPCastToFp16"%op_name]["avg"] = wei_cast if "weight" in u.lower else in_cast

        ### handle successors
        for succ in dag.successors(op_name):
            if "AMPCastToFp16" in u:
                nnexts = dag.successors(succ)
                assert len(nnexts) == 1, nnexts
                dag.add_edge(op_name, nnexts[0])
                self.remove_edge(succ, nnexts[0])
            else:
                dag.add_edge(op_name, "%s.AMPCastToFp32"%op_name)
                dag.add_edge("%s.AMPCastToFp32"%op_name, succ)
                # TODO (huhanpeng): need verify
                dag.nodes["%s.AMPCastToFp32"%op_name]["avg"] = out_cast

        ### update the meta info of current node
        dag.nodes[op_name]["avg"] = self.pred_amp_avg(op_name, _avg=dag.nodes[op_name]["avg"])
        dag.nodes[op_name]["dtype"] = "fp16"

    def is_white_for_amp(self, dag, op_name):
        ''' check whether an OP is finally white or not, according the propogation rules in AMP of TensorFlow '''
        if dag.nodes[op_name].get("is_white", "none") == "none":
            amp_color = self.meta_info.check_amp_lists(op_name)
            if amp_color == "white":
                ### cache the intermediate result
                dag.nodes[op_name]["is_white"] = True
                return True
            if amp_color == "black":
                ### cache the intermediate result
                dag.nodes[op_name]["is_white"] = False
                return False

            ### need to further check the parent nodes
            is_white = True
            for u, _ in dag.in_edges(op_name):
                is_white &= self.is_white_for_amp(dag, u)
                if not is_white:
                    break
            ### cache the intermediate result
            dag.nodes[op_name]["is_white"] = is_white
            return is_white
        else:
            ### return the cached results
            return dag.nodes[op_name]["is_white"]

    def is_need_amp(self, dag, op_name):
        ''' check whether an OP need be quantized, only those with fp32 and in the final white list need be quantized'''
        if dag.nodes[op_name].get("dtype", "fp32") != "fp32":
            return False

        ### TODO (huhanpeng) do not consider gradients/ nodes for mixed precision trainign
        if "gradients/" in op_name:
            return False

        return self.is_white_for_amp(dag, op_name)

class AMPTrainer:
    def __init__(self, meta_path, cost_model_dir):
        self.meta_info = MetaInfo(meta_path)
        self.cost_model_path = os.path.join(cost_model_dir, 'cost_model.json')
        if os.path.exists(self.cost_model_path):
            with open(self.cost_model_path, 'r') as fp:
                self.cost_model = json.load(cost_model_dir)
        else:
            self.cost_model = {}

        self.all_data_dict = {}

    def collect_raw_data(self, rst_dir):
        ''' collect op names and batch sizes
        '''
        # rst_dir="/Users/hhp/0/traces/traces20200806/traces20200807_01_bytedance"
        # rst_dir = "/Users/hhp/0/git/byteprofile-analysis/data/data_20200817_resnet50/v100"
        self.NAMELIST_32 = None
        self.NAMELIST_16 = None
        self.DATA_32 = {}
        self.DATA_16 = {}
        self.VAR_32 = {}
        self.VAR_16 = {}

        self.BATCH_LIST_VALUE = []

        with open(os.path.join(rst_dir, "name.txt"), 'r') as fp:
            lines = fp.read().split("\n")
            if lines[-1] == "":
                lines = lines[:-1]
            for line in lines:
                if "fp32" in line:
                    self.NAMELIST_32 = str2list(line.split(":")[1])
                elif "fp16" in line:
                    self.NAMELIST_16 = str2list(line.split(":")[1])
                else:
                    raise

        with open(os.path.join(rst_dir, "avg.txt"), 'r') as fp:
            lines = fp.read().split("\n")
            idx = 0
            while idx < len(lines):
                if "huhanpeng" in lines[idx]:
                    if idx+1 < len(lines) and ("huhanpeng" in lines[idx+1] or lines[idx+1]==""):
                        ### avoid add addition batch size to BATCH_LIST_VALUE
                        idx += 1
                        continue
                    batchsize = int(lines[idx].split("--batch_size")[1].split("--")[0])
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
                    if idx >= len(lines) or "huhanpeng" not in lines[idx]:
                        _DATA["B=%d"%batchsize] = str2list(lines[idx+1], dtype=float)
                    else:
                        continue
                    idx += 1
                    if idx >= len(lines) or "huhanpeng" not in lines[idx]:
                        _VAR["B=%d"%batchsize] = str2list(lines[idx+1], dtype=float)
                    else:
                        continue
                idx += 1

        self.BATCH_LIST_VALUE = [e for e in self.BATCH_LIST_VALUE if (e >= 0 and e <=1024)]
        self.BATCH_LIST_VALUE = sorted(self.BATCH_LIST_VALUE)

    def gen_train_data(self, dag):
        ######################################################################
        ### collect training data and test data for each op type
        ######################################################################
        def __record_xdata(S_mul, S_add, S_in, S_out, S_wei, gflops, avg, op_type):
            if op_type not in self.all_data_dict:
                self.all_data_dict[op_type] = [[], [], [], [], [], [], []]
            self.all_data_dict[op_type][0].append(avg)
            self.all_data_dict[op_type][1].append(gflops)
            self.all_data_dict[op_type][2].append(S_mul)
            self.all_data_dict[op_type][3].append(S_add)
            self.all_data_dict[op_type][4].append(S_in)
            self.all_data_dict[op_type][5].append(S_out)
            self.all_data_dict[op_type][6].append(S_wei)

        for op_name in dag.nodes:
            op_name = parse_layer_name(op_name)
            if op_name not in self.NAMELIST_32 or op_name not in self.NAMELIST_16:
                continue
            if "gradients/" in op_name:
                continue
            op_type = self.meta_info.ret_op_type(op_name)
            if op_type not in [
                                # "Conv2D",
                                # "MatMul",
                                "Cast",
                                ]:
                continue

            print(op_name)
            for b in self.BATCH_LIST_VALUE:
                ### filter
                if b <= BATCH_LIST_VALUE:
                    continue

                ### collect data
                try:
                    op_type, S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_tf_metadata(op_name, batch_size=b)
                except (NotImplementedError, KeyError) as e:
                    break
                idx_in_32 = NAMELIST_32.index(op_name)
                avg_ = self.DATA_32["B=%d"%b][idx_in_32]
                var_ = self.VAR_32["B=%d"%b][idx_in_32] if "B=%d"%b in VAR_32 else 0
                if (var_ / avg_) <= VAR_THREHOLD:
                    __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP32, avg_, op_type)

                idx_in_16 = NAMELIST_16.index(op_name)
                avg_ = self.DATA_16["B=%d"%b][idx_in_16]
                var_ = self.VAR_16["B=%d"%b][idx_in_16] if "B=%d"%b in VAR_16 else 0
                if (var_ / avg_) <= VAR_THREHOLD:
                    __record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP16, avg_, op_type)
        
        for op_type in self.all_data_dict.keys():   
            ### all_data[S_mul, ...][# of nodes * # of batch size values]
            all_data = np.array(self.all_data_dict[op_type])
            ### all_data[# of nodes * # of batch size values][S_mul, ...]
            all_data = np.transpose(all_data)
            ### filter
            # all_data = exct_filter(all_data)
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
                SingleLogger().info("Since TRAIN_PERCENT is set to be >= 1, set test data equal to the training data")
            else:
                test_data = np.split(all_data[~mask, :], [1], axis=1)
            test_data = np.split(all_data[~mask, :], [1], axis=1)
            self.all_data_dict[op_type]["train_x"], self.all_data_dict[op_type]["train_y"] = train_data[1], train_data[0]
            self.all_data_dict[op_type]["test_x"], self.all_data_dict[op_type]["test_y"] = test_data[1], test_data[0]
            SingleLogger().info("OP {}: Collect training data - X:{}, Y:{}, test data - X:{}, Y:{}".format(
                op_type, train_data[1].shape, train_data[0].shape, test_data[1].shape, test_data[0].shape))

    def train_one_op(self, op_type, test=False):
        train_x, train_y = self.all_data_dict[op_type]["train_x"], self.all_data_dict[op_type]["train_y"]
        test_x, test_y = self.all_data_dict[op_type]["test_x"], self.all_data_dict[op_type]["test_y"]

        _train_x = np.transpose(train_x)
        _train_y = np.transpose(train_y).flatten()
        popt, pcov = curve_fit(func_pred_time, _train_x, _train_y, bounds=(LOWER_BOUNDS, UPPER_BOUNDS), p0=P0, maxfev=10000)
        try:
            self.cost_model["Conv2D"]["popt"] = popt
        except KeyError as e:
            self.cost_model["Conv2D"] = {"popt": popt}
        if test:
            _test_x = np.transpose(test_x)
            _test_y = np.transpose(test_y).flatten()
            avgs_pred = func_pred_time(_test_x, *popt)
            error = predict_error(_test_y, avgs_pred)
            SingleLogger().info("average error: %f %%"%(error * 100))

    def train(self, op_type=None, test=False):
        if op_type is None:
            for key in self.all_data_dict:
                self.train_one_op(key, test)

    def dump_cost_model(self):
        with open(self.cost_model_path, 'w') as fp:
            json.dump(self.cost_model, fp)
