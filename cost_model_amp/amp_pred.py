import os, sys
import math
import ujson as json
import numpy as np
from trace_utils import *
from logger_utils import Singleton, SingleLogger
import arg_utils
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
        
    def pred_amp_avg(self, op_name, _avg=None):
        ''' Predict fp16 time of fp32 operators
            If _avg is given, use the relative value
        '''
        op_type = self.meta_info.parse_op_type(op_name)
        if op_type not in self.cost_model:
            SingleLogger().warn("{}({}) is not supported for AMP now".format(op_name, op_type))
            return _avg / 2.0
        S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_metadata(op_name)
        raw_meta = self.meta_info.ret_rawmeta(op_name)
        _xdata = [S_mul, S_add, S_in, S_out, S_wei] + raw_meta
        _xdata32 = [GFLOPS_FP32] + _xdata
        _xdata16 = [GFLOPS_FP16] + _xdata

        avg_fp32 = self.cost_model[op_type].predict(_xdata32)
        avg_fp16 = self.cost_model[op_type].predict(_xdata16)

        if _avg is not None:
            return _avg * avg_fp16 / avg_fp32
        else:
            return avg_fp16

    def pre_cast_time(self, op_name):
        if "Cast" not in self.cost_model:
            SingleLogger().error("({}) is not supported for AMP now".format("Cast"))
        _, _, S_in, S_out, S_wei = self.meta_info.ret_metadata(op_name)

        in_cast = self.cost_model["Cast"].predict([None, None, None, S_in, None, None])
        out_cast = self.cost_model["Cast"].predict([None, None, None, S_out, None, None])
        wei_cast = self.cost_model["Cast"].predict([None, None, None, S_wei, None, None])
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
                dag.add_edge(u, "%s~>AMPCastToFp16"%op_name)
                dag.add_edge("%s~>AMPCastToFp16"%op_name, op_name)
                # TODO (huhanpeng): need verify
                dag.nodes["%s~>AMPCastToFp16"%op_name]["avg"] = wei_cast if "weight" in u.lower else in_cast

        ### handle successors
        for succ in dag.successors(op_name):
            if "AMPCastToFp16" in u:
                nnexts = dag.successors(succ)
                assert len(nnexts) == 1, nnexts
                dag.add_edge(op_name, nnexts[0])
                self.remove_edge(succ, nnexts[0])
            else:
                dag.add_edge(op_name, "%s~>AMPCastToFp32"%op_name)
                dag.add_edge("%s~>AMPCastToFp32"%op_name, succ)
                # TODO (huhanpeng): need verify
                dag.nodes["%s~>AMPCastToFp32"%op_name]["avg"] = out_cast

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

        # TODO (huhanpeng): not implemented
        return False
        ### TODO (huhanpeng) do not consider gradients/ nodes for mixed precision trainign
        if "gradients/" in op_name:
            return False

        return self.is_white_for_amp(dag, op_name)

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
