import os, sys
import math
import ujson as json
import numpy as np
from trace_utils import *
from logger_utils import Singleton, SingleLogger
import arg_utils
import pickle
args_ = arg_utils.SingleArg().args

if args_.platform == "MXNET":
    from ml_platform.mxnet.metadata import MetaInfo, FULL_HEADERS, OP_HYPER_PARAMETERS, BASE_HEADER_LEN
    from ml_platform.mxnet.metadata import GFLOPS_FP32, GFLOPS_FP16
elif args_.platform == "TENSORFLOW":
    from ml_platform.tensorflow.metadata import MetaInfo, FULL_HEADERS, OP_HYPER_PARAMETERS, BASE_HEADER_LEN
    from ml_platform.tensorflow.metadata import GFLOPS_FP32, GFLOPS_FP16
else:
    raise NotImplementedError()

class AMPPredictor:
    def __init__(self, meta_info):
        ### load AMP cost model
        self.cost_model = {}
        cm_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), ".cost_model")
        # cm_dir = "/Users/bytedance/0/git/byteprofile-analysis/cost_model_amp/.cost_model"
        for file_ in os.listdir(cm_dir):
            cm_path = os.path.join(cm_dir, file_)
            assert os.path.isfile(cm_path)
            grp = grouper.load_grouper(cm_path)
            assert grp.op_type in file_
            self.cost_model[grp.op_type] = grp
        if len(self.cost_model) == 0:
            SingleLogger().warn("No AMP cost model at {}".format(cm_dir))
        
        self.meta_info = meta_info

        ### load checkpoint
        self.ckpt_path = os.path.join(args_.workspace, "ckpt_amp.pickle")
        self.cast_cnt = 0
        self.num_nonvar_casts_to_fp16 = 0
        self.op_status = {}
    
    def checkpoint(self):
        with open(self.ckpt_path, "wb") as f:
            pickle.dump([self.cast_cnt, self.num_nonvar_casts_to_fp16, self.op_status], f)
    
    def load_ckpt(self):
        assert os.path.isfile(self.ckpt_path)
        with open(self.ckpt_path, "rb") as f:
            self.cast_cnt, self.num_nonvar_casts_to_fp16, self.op_status = pickle.load(f)

    def pred_amp_avg(self, op_name, _avg=None):
        ''' Predict fp16 time of fp32 operators
            If _avg is given, use the relative value
        '''
        rawname, _ = self.meta_info.standarize_name(op_name)
        op_type = self.meta_info.parse_op_type(rawname)
        if op_type not in self.cost_model:
            # SingleLogger().warn("{}({}) is not supported for AMP now".format(op_name, op_type))
            return _avg / 2.0
        S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_metadata(rawname)
        raw_meta = self.meta_info.ret_rawmeta(rawname)
        _xdata = [S_mul, S_add, S_in, S_out, S_wei] + raw_meta
        _xdata32 = [GFLOPS_FP32] + _xdata
        _xdata16 = [GFLOPS_FP16] + _xdata

        avg_fp32 = self.cost_model[op_type].predict(_xdata32)
        avg_fp16 = self.cost_model[op_type].predict(_xdata16)

        if _avg is not None:
            return _avg * avg_fp16 / avg_fp32
        else:
            return avg_fp16

    def output_cast_time(self, op_name, tofp16=True):
        ''' 
        Parameters
        ----------
        op_name: str, long name
            pid->op_type.name.sub_op~>suffix
        '''
        rawname, cat = self.meta_info.standarize_name(op_name)
        assert cat != CatName.COMM.value
        cm_name = "CastToFp16" if tofp16 else "CastToFp32"
        if cm_name not in self.cost_model:
            SingleLogger().error("({}) is not supported for AMP now".format(cm_name))
        try:
            _, _, S_in, S_out, S_wei = self.meta_info.ret_metadata(rawname)
        except:
            print(op_name)
            raise
        cast_time = self.cost_model[cm_name].predict([1, 1, 1, S_out, 1, 1])
        return cast_time

    def quantize(self, dag, op_name):
        if op_name not in self.op_status:
            self.init_op_statu(op_name)
        assert self.op_status[op_name]["dtype"] == "float32", (
            op_name, self.op_status[op_name]["dtype"])

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = []

        def _add_cast_op(u, v, cast_time, to16=True):
            if to16:
                cast_op = "{}~>AMPCastToFp16_{}".format(u, self.cast_cnt)

                raw_u, _ = self.meta_info.standarize_name(u)
                if not self.meta_info.is_const(raw_u) and not self.meta_info.is_variable(raw_u):
                    self.num_nonvar_casts_to_fp16 += 1
            else:
                cast_op = "{}~>AMPCastToFp32_{}".format(u, self.cast_cnt)
            self.cast_cnt += 1
            nodes_to_add.append((cast_op, {"avg": cast_time}))
            edges_to_add.append((u, cast_op))
            edges_to_add.append((cast_op, v))
            edges_to_rm.append((u, v))
        
        def _rm_cast_op(u, cast_op, v):
            edges_to_add.append((u, v))
            edges_to_rm.append((u, cast_op))
            edges_to_rm.append((cast_op, v))
            nodes_to_rm.append(cast_op)

        ### handle parent nodes
        for u, _ in dag.in_edges(op_name):
            to_process = [u]
            while len(to_process) > 0:
                _u = to_process.pop(0)
                if "ConstantFolding" in _u or "LayoutOptimizer" in _u or "group_deps" in _u:
                    to_process += [x for x, _ in dag.in_edges(_u) if x != _u]
                    continue
                if "AMPCastToFp32" in _u:
                    ### not MixedPrecision Boundary, remove the cast
                    prevs = list(dag.in_edges(_u))
                    assert len(prevs) == 1, prevs
                    _rm_cast_op(prevs[0][0], _u, op_name)
                else:
                    try:
                        cast_time = self.output_cast_time(_u)
                    except (KeyError, IndexError):
                        to_process += [x for x, _ in dag.in_edges(_u) if x != _u]
                        continue
                    ### the boundary of mixed precision, add a cast op
                    _add_cast_op(_u, op_name, cast_time, to16=True)

        ### handle successors
        out_cast_time = self.output_cast_time(op_name, tofp16=False)
        for succ in dag.successors(op_name):
            if "AMPCastToFp16" in succ:
                ### not MixedPrecision Boundary, remove the Cast
                _nexts = list(dag.successors(succ))
                assert len(_nexts) == 1, (op_name, succ, _nexts)
                _rm_cast_op(op_name, succ, _nexts[0])
                self.num_nonvar_casts_to_fp16 -= 1
            elif "Comm" in succ:
                ### For BW->Comm edges, 
                ### * Sync time, Memcopy time, Send/Recv becomes 1/2
                ### * Add AMPCastToFp32 after the last memcpy and before update op
                def recursive_convert_comm(_node):
                    if _node not in self.op_status:
                        self.op_status[_node] = {"dtype": "float32"}
                    if self.op_status[_node]["dtype"] == "float16":
                        ### This comm and its downstream operators have been converted
                        return

                    ### Comm operators with sub op in ["Sync", "MEMCPY_IN_FUSION_BUFFER", "MEMCPY_OUT_FUSION_BUFFER", "SEND", "RECV"]:
                    ### avg time = half
                    half_avg = True
                    for sub_op in ["QUEUE"]:
                        if sub_op in _node:
                            half_avg = False
                            break
                    if half_avg:
                        dag.nodes[_node]["avg"] = dag.nodes[_node]["avg"] / 2.0
                    self.op_status[_node]["dtype"] = "float16"

                    for _succ in dag.successors(_node):
                        if "UPDATE_" in _succ:
                            ### Use UPDATE operator's name to let cast op run on computation device
                            _add_cast_op(_node, _succ, out_cast_time, to16=False)
                        else:
                            recursive_convert_comm(_succ)
                recursive_convert_comm(succ)
            else:
                _add_cast_op(op_name, succ, out_cast_time, to16=False)

        ### update the meta info of current node
        prev_avg = dag.nodes[op_name]["avg"]
        dag.nodes[op_name]["avg"] = self.pred_amp_avg(op_name, _avg=prev_avg)
        self.op_status[op_name]["dtype"] = "float16"
        # SingleLogger().info("Convert {} from {} ms to {} ms".format(op_name, prev_avg, dag.nodes[op_name]["avg"]))

        dag.add_nodes_from(nodes_to_add)
        dag.add_edges_from(edges_to_add)
        dag.remove_edges_from(edges_to_rm)
        dag.remove_nodes_from(nodes_to_rm)

        # if not self.check_dag(dag):
        #     raise ValueError("Incorrect dag when quantizing {}".format(op_name))

        return [n for n, d in nodes_to_add]
    
    def check_dag(self, dag):
        for n in dag.nodes():
            if "AMPCastTo" in n:
                succ_list = list(dag.successors(n))
                if len(succ_list) != 1:
                    print("{} has incorrect number of succs: {}".format(n, succ_list))
                    return False
        return True

    def init_op_statu(self, op_name):
        rawname, _ = self.meta_info.standarize_name(op_name)
        dtype = self.meta_info.ret_op_precision(rawname)
        amp_color = self.meta_info.check_amp_lists(rawname)
        self.op_status[op_name] = {"dtype": dtype, "is_white": "none", "color": amp_color}

    def op_amp_color(self, op_name):
        if op_name not in self.op_status:
            self.init_op_statu(op_name)
        return self.op_status[op_name]["color"]

    def is_white_for_amp(self, dag, op_name):
        ''' check whether an OP is finally white or not, according the propogation rules in AMP of TensorFlow '''
        if op_name not in self.op_status:
            self.init_op_statu(op_name)
        if self.op_status[op_name]["is_white"] == "none":
            amp_color = self.op_amp_color(op_name)
            if amp_color == "white" or amp_color == "clear":
                ### cache the intermediate result
                self.op_status[op_name]["is_white"] = True
                return True
            if amp_color == "black":
                ### cache the intermediate result
                self.op_status[op_name]["is_white"] = False
                return False
            
            ### need to further check the parent nodes
            is_white = True
            for u, _ in dag.in_edges(op_name):
                is_white &= self.is_white_for_amp(dag, u)
                if not is_white:
                    break
            ### cache the intermediate result
            self.op_status[op_name]["is_white"] = is_white
            return is_white
        else:
            ### return the cached results
            return self.op_status[op_name]["is_white"]

    def is_need_amp(self, dag, op_name):
        ''' check whether an OP need be quantized, only those with fp32 and in the final white list need be quantized'''
        if "Comm" in op_name or "AMPCastTo" in op_name or "UPDATE_" in op_name:
            return False
        try:
            rawname, _ = self.meta_info.standarize_name(op_name)
            op_type = self.meta_info.parse_op_type(rawname)
            if op_type in ["ReadVariableOp", "Const", "VarHandleOp", "AssignVariableOp"]:
                return False
        except KeyError:
            return False
        if not self.check_dtype_fp32(op_name):
            return False
        return self.is_white_for_amp(dag, op_name)
    
    def check_dtype_fp32(self, op_name):
        ''' Return true if this op is fp32 else False '''
        if op_name not in self.op_status:
            self.init_op_statu(op_name)
        return self.op_status[op_name]["dtype"] == "float32"

from cost_model_amp import dataloader, grouper

def train_amp_model():
    OP_TYPES = {
        # "Conv2D": {
        #     "data_dir": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
        #     "metadata_path": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
        #     "del": [
        #     grouper.Delimiter("R", td_len=0.1, fd_len=0., unit_len=0.1),
        #     grouper.Delimiter("G", td_len=0.1, fd_len=0., unit_len=0.1)]
        # },
        "CastToFp16": {
            "data_dir": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
            "metadata_path": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
            "del": []
        },
        "CastToFp32": {
            "data_dir": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
            "metadata_path": "/Users/bytedance/0/data/20201209/20201209_03_tf_resnet_b=4~256",
            "del": []
        }
    }

    grp_dict = {}
    for target_optype in OP_TYPES.keys():
        data_ld = dataloader.DataLoader(data_dir=OP_TYPES[target_optype]["data_dir"],
                                        metadata_path=OP_TYPES[target_optype]["metadata_path"])
        ### metadata name is raw name
        op_names = data_ld.pick_some_ops(target_optype)
        if len(op_names) == 0:
            continue
        
        train_x, train_y, test_x, test_y = data_ld.collect_data(
            op_names, target_optype, verbose=True)

        dels = OP_TYPES[target_optype]["del"]
        grp = grouper.Grouper(dels, headers=dataloader.FULL_HEADERS[target_optype],
                            op_type=target_optype, max_of_each_dim=data_ld.max_of_each_dim)

        grp.divide_by_len(train_x, train_y, test_x, test_y)
        grp.train_all()
        grp.test_all(visual=False)
        grp_dict[target_optype] = grp
        grp.dump()
