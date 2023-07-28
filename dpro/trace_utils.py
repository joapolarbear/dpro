import os
import ujson as json
import random
import math
import xlsxwriter
import threading
from enum import Enum
import numpy as np

from .logger_utils import Singleton, SingleLogger
from .base import bcolors

QUEUETYPE = {
    "NCCL": {
        "fine": [
            "Sync",
            "QUEUE",
            "MEMCPY_IN_FUSION_BUFFER",
            "NCCL_ALLREDUCE",
            "MEMCPY_OUT_FUSION_BUFFER"
            ],
        "coarse": [
            "NEGOTIATE_ALLREDUCE",
            "ALLREDUCE"
            ], 
        }
}

### The delimiter bettwen the pid and the  op_name in the standard long name format
DEL = "->"
DDEL = "~>"

MAX_CNT = None
ITER_GAP_LOWER_BOUND_US = 5000
### based on the assumption that FW or IO is the start of a step
AS_START_CAT = ["I/O", "operator.FW"]


@Singleton
class QueueType:
    def __init__(self, backend, fine_grained=True):
        self.queue_type_list = QUEUETYPE[backend]["fine" if fine_grained else "coarse"]

    def ret_list(self):
        return self.queue_type_list

class DirLevel(Enum):
    GPU=0
    WORKER=1
    TRIAL=2

class FileName(Enum):
    # Overall profiling/analysis results
    TRACE="bps_trace_final.json"
    STATISTIC="statistic.txt"
    TRAIL_DAG="trail_dag.gml"
    LOCAL_DFG="local_dfg.gml" # single-worker DFG

    # Per rank
    METADATA="metadata.json"
    IO="io.json" # (Optional)
    DAG="dag.gml"
    COMP = "trace.json.gz"

    ## Communication-related
    COMM="comm.json"
    COMM_DETAIL="comm_detail.json"
    TENSOR_NAME="gradient_name_list.json"
    
    ### For All-Reduce
    NCCL_GRAPH="nccl_graph.txt"
    NCCL_RANK_GRAPH="nccl_rank_graph.json"
    
    ### For BPS
    BPS_COMM_DETAIL="comm_timeline.json"
    BPS_SERVER_TRACE="server_timeline.json"
    IP_TO_RANK="ip_to_rank.txt"
    KEY_DICT="key_dict.txt"
    BYTEPS_CACHE="bps_cache.pickle"
    BPS_ALIGNED_TRACE="bps_comm_aligned.json"
    
    ### Deperacated
    GRAPHDEF = "final_graph.json"
    SYMBOL="symbol_debug_str.txt"
    INFO="info.json"  

class CatName(Enum):
    OPERATOR="operator"
    PS_SERVER_OPERATOR="ServerOp"
    COMM="Comm"
    IO="I/O"
    DEBUG="virtual"

class PlatformName(Enum):
    MXNET = "MXNET"
    TENSORFLOW = "TENSORFLOW"

GAP_STR_OP2OP = "%sGAP%s"%(CatName.OPERATOR.value, CatName.OPERATOR.value)
GAP_STR_COMM2COMM = "%sGAP%s"%(CatName.COMM.value, CatName.COMM.value)
GAP_STR_OP2COMM = "%sGAP%s"%(CatName.OPERATOR.value, CatName.COMM.value)
GAP_STR_COMM2OP = "%sGAP%s"%(CatName.COMM.value, CatName.OPERATOR.value)
GAP_STR_COMM2SVOP = "%sGAP%s"%(CatName.COMM.value, CatName.PS_SERVER_OPERATOR.value)
GAP_STR_SVOP2COMM = "%sGAP%s"%(CatName.PS_SERVER_OPERATOR.value, CatName.COMM.value)
GAP_STR_INTERNODE = "INTERNODEGAP"

# GAP_INTERDEVICE = "INTERDEVICE"
# GAP_INTRADEVICE = "INTRADEVICE"

def read_traces(traces_path):
    '''
    Return: a list of traces
    '''
    with open(traces_path, 'r') as fp:
        _traces = json.load(fp)
    if isinstance(_traces, dict):
        traces = _traces.get("traceEvents")
    elif isinstance(_traces, list):
        traces = _traces
    else:
        raise ValueError("The output file not follow the stardard chrome tracing format!: " + traces_path)
    return traces

def first_valid_dir(_path):
    for _dir in os.listdir(_path):
        if _dir.startswith('.'):
            continue
        return _dir
    raise ValueError("No valid directory under {}".format(_path))

def formal_dpro_rawname(_name):
    return _name.replace(".", "_")

def gen_long_name(prefix, raw_name, suffix=None):
    if prefix is None:
        pre = ""
    else:
        assert DEL not in raw_name
        pre = prefix + DEL
    if suffix is None:
        suf = ""
    else:
        assert DDEL not in raw_name
        suf = DDEL + suffix
    return pre + raw_name + suf

def _parse_long_name(long_name):
    if DEL in long_name:
        pid, long_name = long_name.split(DEL)
    else:
        pid = None
    if DDEL in long_name:
        std_name, suffix = long_name.split(DDEL)
    else:
        std_name, suffix = long_name, None
    name_split = std_name.split(".")
    if len(name_split) == 2:
        _, op_name = name_split
        sub_op = None
    elif len(name_split) == 3:
        _, op_name, sub_op = name_split
    else:
        raise ValueError(long_name)
    return pid, std_name, op_name, sub_op, suffix

def parse_op_name(name):
    pid, std_name, op_name, sub_op, suffix = _parse_long_name(name)
    return op_name

def parse_rawname(name):
    if DEL not in name:
        return name
    else:
        return name.split(DEL)[1]

def parse_pid_from_name(name):
    if "+" in name and "Comm" not in name:
        name = name.split("+")[0]
    pid, std_name, op_name, sub_op, suffix = _parse_long_name(name)
    return "none" if pid is None else pid

def parse_suffix_from_name(name):
    pid, std_name, op_name, sub_op, suffix = _parse_long_name(name)
    return suffix

def _parse_tf_layer_names(name):
    raise NotImplementedError("Move to tf directory")
    layer_names = []
    if "+" in name:
        for _name in name.split("+"):
            layer_names += _parse_tf_layer_names(_name)
        return layer_names
    op_type = None
    try:
        op_type = parse_cat_fine_grained(name)
        name = parse_op_name(name)
    except ValueError:
        pass
    if op_type is None:
        op_name_split = parse_op_name(name).split("/")
        op_type = "FW"
        idx = 0
        if op_name_split[idx] == "gradients" or op_name_split[idx] == "gradient_tape":
            op_type = "BW"
            idx += 1
        if op_name_split[idx].lower() in ["inception_v3", "resnet50", "vgg16", "bert"]:
            idx += 1

        if idx >= len(op_name_split) - 2:
            layer_names.append("{}.{}".format(op_type, op_name_split[idx]))
        elif op_name_split[idx-1].lower() == "bert":
            layer_names.append(
                "{}.{}/{}".format(op_type, op_name_split[idx], op_name_split[idx+1]))
        else:
            layer_names.append("{}.{}".format(op_type, op_name_split[idx]))
    return layer_names

def _parse_tf_rough_layer(name, model_name):
    ''' Roughly decide tensorlow op's layer
        * Only for BW ops in the main skeleton
        * Based on empirical knowledge, may be error-prone
    '''
    assert "+" not in name
    op_type = parse_cat_fine_grained(name)
    if op_type in ["operator.FW", "operator.UPDATE"] or "Comm" in op_type:
        return op_type
    elif op_type == "operator.BW":
        op_name_split = parse_op_name(name).split("/")
        if model_name in op_name_split:
            rough_layer = op_name_split[op_name_split.index(model_name)+1]
            if "_" in rough_layer:
                rough_layer = "_".join(rough_layer.split("_")[:-1])
            return "BW." + rough_layer
        else:
            return "BW." + "none"
    else:
        raise ValueError(name, op_type)

def _map_tf_op2layer(local_dfg, model_name,
    use_rough_layer=True, check_pred=True):
        
    op2layer = {}
    layer2ops = {}
    todo = set()

    def same_rough_layer(bw_u, bw_v):
        ''' Use some empirical experience to tell bw_u 
            and bw_v must be not in the same `layer`
        '''
        if "FW" in bw_v:
            return False
        rough_layer_u, rough_layer_v = _parse_tf_rough_layer(bw_u, model_name), _parse_tf_rough_layer(bw_v, model_name)
        if "BW" in rough_layer_u and "BW" in rough_layer_v and "none" not in rough_layer_u and "none" not in rough_layer_v:
            if  rough_layer_u != rough_layer_v:
                return False
        return True

    def recur_parse_layer(bw_op, visited_set = set()):
        ''' Recursively decide bw_op's layer in post-order
        * If `use_rough_layer` is set True, operators with different `rough_layer` belong to different layers
        * If `check_pred` is set True, check the predecessors to decide the layer
        '''
        if bw_op in op2layer:
            return op2layer[bw_op]

        for succ in local_dfg.successors(bw_op):
            if (visited_set and succ in visited_set) or (use_rough_layer and not same_rough_layer(bw_op, succ)):
                continue
            if "FW" in succ or "UPDATE_" in succ or "Comm" in succ:
                continue
            else:
                visited_set.add(bw_op)
                layer = recur_parse_layer(succ, visited_set = visited_set)
                visited_set.remove(bw_op)
            
        layer = None
        layers_from_comm_succs = [succ.split("Comm.")[1] for succ in local_dfg.successors(bw_op) if "Comm" in succ]
        layers_from_exist = [op2layer[succ] for succ in local_dfg.successors(bw_op) if succ in op2layer]
        if len(layers_from_comm_succs) > 0:
            layers_from_comm_succs = set(layers_from_comm_succs)
            if len(layers_from_comm_succs) > 1:
                print("[Layer View] {}: {} possible layers from comm: {}".format(bw_op,
                    len(layers_from_comm_succs), layers_from_comm_succs))
            layer = layers_from_comm_succs.pop()
        elif len(layers_from_exist) > 0:
            layers_from_exist = set(layers_from_exist)
            if len(layers_from_exist) > 1:
                print("[Layer View] {}: {} possible layers from succ: {}.".format(bw_op,
                    len(layers_from_exist), layers_from_exist), 
                    " We currently do not assign any this node to the same layer view of any its successors.")
                layer = bw_op
            else:
                layer = layers_from_exist.pop()
        else:
            pass

        if layer is None and check_pred and False:
            for pred in local_dfg.predecessors(bw_op):
                if (visited_set and pred in visited_set) or (use_rough_layer and not same_rough_layer(bw_op, pred)):
                    continue
                if "FW" in pred or "UPDATE_" in pred:
                    continue
                assert "Comm" not in pred, (bw_op, pred)    
                if pred in op2layer:
                    layer = op2layer[pred]
                else:            
                    visited_set.add(bw_op)
                    layer = recur_parse_layer(pred, visited_set = visited_set)
                    visited_set.remove(bw_op)
                
                if layer is not None:
                    break

        if layer is not None:
            # for _op in [bw_op] + list(visited_set):
            _op = bw_op
            assert _op not in op2layer, (_op, visited_set)
            op2layer[_op] = layer
            if _op in todo:
                todo.remove(_op)
            if layer not in layer2ops:
                layer2ops[layer] = set()
            layer2ops[layer].add(_op)

            if "FW" in _op or "Comm" in _op:
                print(bw_op, _op, visited_set)
                raise
        else:
            todo.add(bw_op)
        return layer

    for node in local_dfg.nodes:
        if "FW" in node or "UPDATE_" in node or "Comm" in node:
            continue
        recur_parse_layer(node)
    
    if len(todo) > 0:
        SingleLogger().warn("Fail to parse layer names for {} BW operators.".format(len(todo)))
    SingleLogger().info("Parse {} BW layers for {}.".format(len(layer2ops), model_name))

    for layer in layer2ops.keys():
        layer2ops[layer] = list(layer2ops[layer])

    # with open("/home/tiger/layer_map.json", 'w') as fp:
    #     json.dump({"layer2ops": layer2ops, "todo": list(todo)}, fp)
        
    return op2layer, layer2ops

def parse_cat_from_name(name):
    if "I/O" in name:
        return CatName.IO.value
    elif "COPY_FIRST" in name or "SUM" in name or "COPY_MERGED" in name:
        return CatName.PS_SERVER_OPERATOR.value
    elif "Comm" in name or "PUSH" in name or "PULL" in name:
        return CatName.COMM.value
    elif "FW" in name or "BW" in name or "COMP" in name or "UPDATE" in name or "OUTPUT" in name:
        return CatName.OPERATOR.value
    elif name == "END":
        return CatName.DEBUG.value
    else:
        raise ValueError("Can not decide the cat of %s" % name)

def parse_allinfo_from_name(name):
    pid, std_name, op_name, sub_op, suffix = _parse_long_name(name)
    pid = "none" if pid is None else pid
    cat = parse_cat_from_name(std_name)
    return pid, std_name, cat, suffix

def parse_allinfo_from_name_v2(name):
    pid, std_name, op_name, sub_op, suffix = _parse_long_name(name)
    return op_name, sub_op, suffix

### CATs that will be affected if we change the GPU/CPU rate
COMP_CAT = ["operator.FW", "operator.BW", "operator.UPDATE",
    "operator.OUTPUT", "ServerOp"]
### CATs that will be affected if we change the bandwidth
### TODO (huhanpeng): check whether it is correct for BytePS
COMM_CAT = ["Comm.SEND", "Comm.RECV", "Comm.PUSH_REQ",
            "Comm.PUSH_RES", "Comm.PULL_REQ", "Comm.PULL_RES"]
def parse_cat_fine_grained(name_):
    ### PS communication traces
    if "COPY_FIRST" in name_ or "SUM" in name_ or "COPY_MERGED" in name_:
        return "ServerOp"
    elif "PUSH_REQ" in name_:
        return "Comm.PUSH_REQ"
    elif "PUSH_RES" in name_:
        return "Comm.PUSH_RES"
    elif "PULL_REQ" in name_:
        return "Comm.PULL_REQ"
    elif "PULL_RES" in name_:
        return "Comm.PULL_RES"
    ### Common communication traces
    elif "Comm" in name_:
        if "SEND" in name_:
            ret_cat = "Comm.SEND"
        elif "RECV" in name_:
            ret_cat = "Comm.RECV"
        else:
            ret_cat = "Comm.other"
    ### Computation traces
    elif "BW" in name_ and "FW" in name_:
        ret_cat = "operator.FW+BW"
    elif "BW" in name_:
        ret_cat = "operator.BW"
    elif "FW" in name_:
        ret_cat = "operator.FW"
    elif "COMM" in name_:
        # TODO (delete)
        ret_cat = "operator.COMM"
    elif "COMP" in name_:
        # TODO (delete)
        ret_cat = "operator.COMP"
    ### Others
    elif "I/O" in name_:
        ret_cat = "I/O"
    elif "UPDATE_" in name_:
        ret_cat = "operator.UPDATE"
    elif "OUTPUT" in name_:
        ret_cat = "operator.OUTPUT"
    elif name_ == "END":
        ret_cat = "virtual"
    else:
        raise ValueError("Can not decide the cat of %s" % name_)

    return ret_cat


def parse_special_from_name(name):
    '''Sometimes we need some special information from the name, e.g., if it's Negotiate...
    (TODO) huahanpeng: It should be dedicated for BytePS or Horovod,
    '''
    if "NEGOTIATE" in name:
        return 

def load_list(path):
    ''' read a list from a file, ignore the last line if it is empty
    '''
    with open(path, 'r') as f:
        l = f.read().split("\n")
    l = l[:-1] if l[-1] == '' else l
    return l

def gen_pid_name(comm_backend, worker_dir, gpu_id):
    if comm_backend == "NONE":
        return "default"
    elif worker_dir is None and gpu_id is None:
        ### host pid
        return "host0.rank0" if comm_backend == "NCCL" else "traces_0.rank0"
    else:
        return str(worker_dir)+".rank{}".format(gpu_id)

def is_standard_pid(pid):
    return pid.startswith("host") or pid.startswith("traces_") or pid.startswith("default")

class TraceManager:
    def __init__(self, traces=None, dir_level=None, check=False):
        if traces is None:
            return
        self.traces = self.check_traces(traces) if check else traces
        self.traces = sorted(self.traces, key=lambda x: (x["ts"], x["name"]), reverse=False)
        
        self.dir_level = dir_level
        self.max_step = 0
        self.opt_step = 0   # which step has the cloest performance to the average performance
        self.iter_time = -1

        self.name2sta = None
        self.cat2sta = None
        self.all_prefix = None
        self.ret_stat()

    def dump(self, dir_):
        trace_thread = threading.Thread(target=self._dump, args=(dir_,))
        trace_thread.start()

    def _dump(self, dir_):
        rst_traces = sorted(self.traces, key=lambda x: (x["pid"], x["tid"]))
        with open(os.path.join(dir_, FileName.TRACE.value), 'w') as f:
            json.dump({"traceEvents": rst_traces, "all_prefix": self.all_prefix}, f)
        str_ = "%d,%d,%d,%f\n"%(self.dir_level.value, self.max_step, self.opt_step, self.iter_time)
        str_ += str(self.name2sta) + "\n"
        str_ += str(self.cat2sta)
        with open(os.path.join(dir_, FileName.STATISTIC.value), 'w') as fp:
            fp.write(str_)

    def load(self, dir_):
        with open(os.path.join(dir_, FileName.TRACE.value), 'r') as fp:
            _info = json.load(fp)
        self.traces = _info["traceEvents"]
        self.all_prefix = _info["all_prefix"]
        self.traces = sorted(self.traces, key=lambda x: (x["ts"], x["name"]), reverse=False)
        with open(os.path.join(dir_, FileName.STATISTIC.value), 'r') as fp:
            str_ = fp.read().split("\n")

        dir_level, max_step, opt_step, iter_time = str_[0].split(",")
        self.dir_level = DirLevel(int(dir_level))
        self.max_step = int(max_step)
        self.opt_step = int(opt_step)
        self.iter_time = float(iter_time)

        self.name2sta = eval(str_[1])
        self.cat2sta = eval(str_[2])

    def check_traces(self, traces):
        for trace in traces:
            if trace.get("name", None) is None or trace.get("ts", None) is None:
                print(trace)
                raise RuntimeError("Check trace failed.")
        return traces
        
    def _is_ignore_for_sta(self, event):
        ### Some traces are ignored for statistic
        ### including 1) instance traces, 2) for debug 3) local_num_masks
        return event["ph"].lower() == "i" or event["cat"] == "debug" 
        # \
        #         or "local_num_masks" in event["name"]

    def ret_unique_name(self, event):
        ### Returen unique name for statistic index
        if "args" in event and "chunkId" in event["args"]:
            suffix = "%d_%d_%d_%d"%(event["args"]["loopId"], event["args"]["channelId"], event["args"]["chunkId"], event["args"]["sliceId"])
        else:
            suffix=None
        return gen_long_name(event["pid"], event["name"], suffix=suffix)

    def ret_stat(self, cal_median=False):
        """ 1. Basic Statistic;
            2. add step field
            3. iteration time
        """
        self.name2sta = {}
        self.cat2sta = {}
        prefix_dict = {}
        
        ### Step 1: tranvers all traces
        for event in self.traces:
            if self._is_ignore_for_sta(event):
                continue
            prefix = event["pid"]
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {
                    "cat_cnt": {"operator.FW": 0, "operator.BW": 0, "operator.UPDATE": 0},
                    "step_cnt": 0,
                    "time_base": None,

                    "trace_name_cursor": None,
                    "time_cursor": None,
                    "step_start_ts": None,
                    "fw_end": None,
                    "bw_start": None,
                    "bw_end": None,

                    ### used for calculating iteration time, and fw time
                    "cur_step": None,

                    "fw_multi_steps": [],
                    "bw_multi_steps": [],
                    "update_multi_steps": [],
                    "iter_multi_steps": [],                    
                }
            cat = parse_cat_fine_grained(event["name"])

            ### statistic info
            unique_name = self.ret_unique_name(event)
            dur_in_ms = event["dur"] / 1000.0
            if unique_name in self.name2sta:
                if MAX_CNT is not None and self.name2sta[unique_name]["cnt"] >= MAX_CNT:
                    event["args"]["cnt"] = -1
                    continue
                self.name2sta[unique_name]["cnt"] += 1
                self.name2sta[unique_name]["time"].append(dur_in_ms)
                self.name2sta[unique_name]["min_t"] = min(self.name2sta[unique_name]["min_t"], dur_in_ms)
                self.name2sta[unique_name]["max_t"] = max(self.name2sta[unique_name]["max_t"], dur_in_ms)
            else:
                self.name2sta[unique_name] = {
                    "cnt": 1, 
                    "time": [dur_in_ms], 
                    "min_t": dur_in_ms, 
                    "max_t": dur_in_ms,
                    "cat": cat,
                    "id": len(self.name2sta)
                    }
            ### Stat the appearance of this unique op
            event["args"]["cnt"] = self.name2sta[unique_name]["cnt"] - 1

            pid_info = prefix_dict[prefix]
            if pid_info["time_base"] is None:
                pid_info["time_base"] = event["ts"]
            ### Add the `step` field to the `event`
            if "step" not in event["args"]:
                ### TODO (huhanpeng): Can this adapt to MXNet
                # if pid_info["time_cursor"] is None:
                #     pass
                # elif event["ts"] - pid_info["time_cursor"] - pid_info["time_base"] > ITER_GAP_LOWER_BOUND_US and \
                #         cat in AS_START_CAT and \
                #         pid_info["cat_cnt"]["operator.BW"] > 0 and \
                #         pid_info["cat_cnt"]["operator.UPDATE"] > 0:
                #     pid_info["step_cnt"] += 1
                
                if "server_" in prefix:
                    ### For BytePS, Comm is not in the same pid as computation
                    event["args"]["step"] = -1
                    for ref_pid, ref_pid_info in prefix_dict.items():
                        if "server_" not in ref_pid and "step_cnt" in ref_pid_info and ref_pid_info["step_cnt"] >= 0:
                            event["args"]["step"] = ref_pid_info["step_cnt"]
                            break
                else:
                    event["args"]["step"] = pid_info["step_cnt"]
            else:
                ### For TensorFlow 2.4, step info is directly given the TF profiler
                event["args"]["step"] = int(event["args"]["step"])
                pid_info["step_cnt"] = event["args"]["step"]

            ### Statistic time grouped by fine-grained cat
            if parse_cat_from_name(event["name"]) in [CatName.OPERATOR.value,
                                                        CatName.IO.value,
                                                        CatName.PS_SERVER_OPERATOR.value]:
                if cat not in pid_info["cat_cnt"]:
                    pid_info["cat_cnt"][cat] = 0
                pid_info["cat_cnt"][cat] += dur_in_ms
            self.max_step = max(event["args"]["step"], self.max_step)

            ### Calculate the iteration time
            ### only check the iteration time when current node is FW/BW/UPDATE op
            # * and for byteps traces, there exists pids in the form like server_3_t2....
            #   do not need to calculate iteration time for those pids
            if parse_cat_from_name(event["name"]) != CatName.OPERATOR.value or \
                    not is_standard_pid(prefix):
                continue
            if pid_info["cur_step"] is None:
                ### initialization
                pid_info["step_start_ts"] = event['ts'] - pid_info["time_base"]
                pid_info["time_cursor"] = event['ts'] + event['dur'] - pid_info["time_base"]
                pid_info["cur_step"] = event["args"]["step"]
            elif pid_info["cur_step"] != event["args"]["step"]:
                ### a new iteration
                assert pid_info["step_start_ts"] is not None
                if pid_info["cur_step"] == -1:
                    continue
                assert event["args"]["step"] > pid_info["cur_step"], (event, pid_info)
                pid_info["iter_multi_steps"].append((pid_info["time_cursor"] - pid_info["step_start_ts"]) / 1000.0)
                try:
                    pid_info["fw_multi_steps"].append(pid_info["cat_cnt"]["operator.FW"])
                    pid_info["bw_multi_steps"].append(pid_info["cat_cnt"]["operator.BW"])
                    pid_info["update_multi_steps"].append(pid_info["cat_cnt"]["operator.UPDATE"])
                    pid_info["cat_cnt"]["operator.FW"] = pid_info["cat_cnt"]["operator.BW"] = pid_info["cat_cnt"]["operator.UPDATE"] = 0
                except:
                    print(event, pid_info)
                    raise
                assert pid_info["cur_step"] == len(pid_info["iter_multi_steps"]) - 1
                SingleLogger().debug("%s - the %d th iteration: FW: %f, BW: %f, Iteration time: %f" % (prefix, len(pid_info["iter_multi_steps"]), pid_info["fw_multi_steps"][-1], pid_info["bw_multi_steps"][-1], pid_info["iter_multi_steps"][-1]))
                pid_info["step_start_ts"] = event['ts'] - pid_info["time_base"]
                pid_info["bw_start"] = None
                pid_info["time_cursor"] = event['ts'] + event['dur'] - pid_info["time_base"]
                pid_info["cur_step"] = event["args"]["step"]
            else:
                ### during an iteration
                pid_info["time_cursor"] = event['ts'] + event['dur'] - pid_info["time_base"]

                ### TODO (huhanpeng): change after fine-tune update
                ### here we assume UPDATE is following the last BP op.
                if "FW" in event["name"]:
                    if pid_info["step_start_ts"] is None:
                        pid_info["step_start_ts"] = event['ts'] - pid_info["time_base"]
                    pid_info["fw_end"] = pid_info["time_cursor"]
                if "BW" in event["name"]:
                    if pid_info["bw_start"] is None:
                        pid_info["bw_start"] = event['ts'] - pid_info["time_base"]
                    pid_info["bw_end"] = pid_info["time_cursor"]
            
            if "input_barrier" in unique_name:
                pid_info["step_start_ts"] = None
            
            if parse_cat_from_name(event["name"]) in [CatName.OPERATOR.value,
                                                      CatName.IO.value,
                                                      CatName.PS_SERVER_OPERATOR.value]:
                pid_info["trace_name_cursor"] = event["name"]

        ### Step 2: calculate the iteration time of each iteration
        iter_list_all = []
        step_num_upper = None
        for prefix in sorted(prefix_dict.keys()):
            if not is_standard_pid(prefix):
                continue
            pid_info = prefix_dict[prefix]

            ### Check and statistic the LAST iteration
            if len(pid_info["iter_multi_steps"]) > 0 and (not pid_info["fw_end"] or not pid_info["bw_end"] or not pid_info["bw_start"]):
                ### TODO (hhp): ignore the last iteration now, since sometimes the last iteration
                #   is not completed
                pass
            else:
                pid_info["iter_multi_steps"].append((pid_info["time_cursor"] - pid_info["step_start_ts"]) / 1000.0)
                pid_info["fw_multi_steps"].append(pid_info["cat_cnt"]["operator.FW"])
                try:
                    pid_info["bw_multi_steps"].append(pid_info["cat_cnt"]["operator.BW"])
                except KeyError:
                    ### for fused op, there may be not BW nodes
                    # append -1 as abnormal cases
                    pid_info["bw_multi_steps"].append(-1)
                pid_info["update_multi_steps"].append(pid_info["cat_cnt"]["operator.UPDATE"])
                pid_info["cat_cnt"]["operator.FW"] = pid_info["cat_cnt"]["operator.BW"] = pid_info["cat_cnt"]["operator.UPDATE"] = 0
                SingleLogger().debug("%s - the %d th iteration: FW:%f, BW: %f, Iteration time: %f" % (prefix, len(pid_info["iter_multi_steps"]), pid_info["fw_multi_steps"][-1], pid_info["bw_multi_steps"][-1], pid_info["iter_multi_steps"][-1]))

            ### Statistic the iteration time
            iter_time_multi_steps = np.array(pid_info["iter_multi_steps"])
            iter_time_avg, iter_time_std = np.average(iter_time_multi_steps), np.std(iter_time_multi_steps)
            SingleLogger().debug("<%s> average iter time %f (\u00B1 %f): %s" % (
                prefix, iter_time_avg, iter_time_std, str(pid_info["iter_multi_steps"])))

            fw_time = sum(pid_info["fw_multi_steps"]) / float(len(pid_info["fw_multi_steps"]))
            bw_time = sum(pid_info["bw_multi_steps"]) / float(len(pid_info["bw_multi_steps"]))
            update_time = sum(pid_info["update_multi_steps"]) / float(len(pid_info["update_multi_steps"]))
            iter_list_all.append(pid_info["iter_multi_steps"])
            SingleLogger().info("<%s> fw : %f + bw: %f + update: %f -> time/it = %f (\u00B1 %f) ms" % (prefix,
                    fw_time, bw_time, update_time, iter_time_avg, iter_time_std))
            
            if step_num_upper is None or len(pid_info["iter_multi_steps"]) < step_num_upper:
                ### Different GPUs may have different number of steps, find the smallest one as the step_num_upper
                step_num_upper = len(pid_info["iter_multi_steps"])
    
        ### Step 3: calculate the average iteration time
        # * iter_list_all, shape = (n_GPUs, n_steps) ==> (n_steps)
        iter_list_all = [_list[:step_num_upper] for _list in iter_list_all]
        iter_list_all = np.average(np.array(iter_list_all), axis=0)
        self.iter_time = np.average(iter_list_all)
        _std = np.std(iter_list_all)
        STD_CHECK_THESHOLD = 0.1
        if _std / self.iter_time > STD_CHECK_THESHOLD:
            SingleLogger().info(
                "Std.dev is large compared to Ave. ({:.3f}/{:.3f}), take the median as the iteration time".format(_std, self.iter_time))
            self.iter_time = np.average(iter_list_all[1:])

            self.iter_time = np.median(iter_list_all)

            _std = np.std(iter_list_all[1:])
            self.opt_step = np.argmin(np.abs(iter_list_all - self.iter_time))
            SingleLogger().info("<Overall> step %d is the one closest to average %f (\u00B1 %f) ms - %s" %
                                (self.opt_step, self.iter_time, _std, iter_list_all))
        else:
            self.opt_step = np.argmin(np.abs(iter_list_all - self.iter_time))
            SingleLogger().info("<Overall> step %d is the one closest to average %f (\u00B1 %f) ms - %s" %
                                (self.opt_step, self.iter_time, _std, iter_list_all))

        ### Step 4: calculate the avg of each operator
        for name, statistic in self.name2sta.items():
            ### TODO (huhanpeng), var can be calculated directly with avg list
            statistic["avg"] = sum(statistic["time"]) / statistic["cnt"]
            statistic["median"] = sorted(statistic["time"])[int(statistic["cnt"]/2)]
            statistic["var"] = 0.0

            # assert statistic["time"] != 0
            cat = parse_cat_fine_grained(name)
            if cat in self.cat2sta:
                if statistic["avg"] > self.cat2sta[cat]["max_t"]:
                    self.cat2sta[cat]["max_t"] = statistic["avg"]
                    self.cat2sta[cat]["max_name"] = name
            else:
                self.cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name, "time": 0, "cnt": 0, "op_cnt":0}
            self.cat2sta[cat]["time"] += sum(statistic["time"])
            self.cat2sta[cat]["cnt"] += statistic["cnt"]
            self.cat2sta[cat]["op_cnt"] += 1

        for cat, statistic in self.cat2sta.items():
            statistic["avg"] = statistic["time"] / statistic["cnt"]

        ### Step 4: calculate the variance of each operator
        for idx, event in enumerate(self.traces):
            if self._is_ignore_for_sta(event):
                continue
            unique_name = self.ret_unique_name(event)
            self.name2sta[unique_name]["var"] += pow(event["dur"] / 1000.0 - self.name2sta[unique_name]["avg"], 2)
            
            ### record which steps this operator occurs in
            if "step_ids" not in self.name2sta[unique_name]:
                self.name2sta[unique_name]["step_ids"] = [None] * (self.max_step + 1)
            self.name2sta[unique_name]["step_ids"][event["args"]["step"]] = idx

        for name, statistic in self.name2sta.items():
            statistic["var"] = statistic["var"] / float(statistic["cnt"])
        
        self.all_prefix = list(prefix_dict.keys())

    def print_stat(self, sort=True, line_num=None):
        if sort:
            sort_sta = sorted(self.name2sta.items(), key=lambda x: x[1]["avg"], reverse=True)
        else:
            sort_sta = self.name2sta.items()
        print_str = "{}Profile Statistics.{}\n".format(bcolors.CGREEN, bcolors.ENDC)
        print_str += ("===================\n")
        print_str += ("%-80s\t Total Count\t Min Time (ms)\t Max Time (ms)\t Avg Time (ms)\t Std.dev (ms)\t Median (ms)\n" % ("Name"))
        print_str += ("%-80s\t -----------\t -------------\t -------------\t -------------\t ---------------\t ---------------\n" % ("----"))
        line_cnt = 0
        for name, statistic in sort_sta:
            if (line_num and line_cnt >= line_num):
                break        
            print_str += ("%-80s\t %11d\t %12.4f\t %13.4f\t %13.4f\t %13.4f\t %13.4f\n" %
                    (name,
                    statistic["cnt"],
                    statistic["min_t"],
                    statistic["max_t"],
                    statistic["avg"],
                    math.sqrt(statistic["var"]),
                    statistic.get('median', -1)
                    ))
            line_cnt += 1

        # Group by category
        print_str += ("\n")
        print_str += ("Group by category\n")
        print_str += ("===================\n")
        line_cnt = 0
        for cat, statistic in self.cat2sta.items():
            if (line_num and line_cnt >= line_num):
                    break
            print_str += ("Category: %-10s\t The most time-consuming OP: %-30s -> %13.4f (ms)\n" %
                          (cat, statistic["max_name"], statistic["max_t"] / 1000.0))
            line_cnt += 1

        SingleLogger().info(print_str)

    def lookup_stat(self, comm_backend, wk_prefix, local_rank, name, _field="avg"):
        ''' look up data from the stat info, return average time in ms by default
        Parameters
        __________
        with_prefix: boolean
            if True, name has had prefix, no need to process it
        '''
        if self.dir_level == DirLevel.GPU or self.has_prefix(name):
            unique_name = name
        elif self.dir_level == DirLevel.WORKER:
            unique_name = gen_long_name("rank{}".format(local_rank), name)
        elif self.dir_level == DirLevel.TRIAL:
            unique_name = gen_long_name(gen_pid_name(comm_backend, wk_prefix, local_rank), name)
        else:
            raise RuntimeError("Unsupported DirLevel.")

        if unique_name not in self.name2sta:
            # SingleLogger().warn("Fail to find the trace of %s" % unique_name)
            return 0.0
        else:
            if _field == "avg" and self.name2sta[unique_name]["avg"] > 0 and \
                math.sqrt(self.name2sta[unique_name]["var"]) / self.name2sta[unique_name]["avg"] > 1:
                ### If an operator's execution time is unstable, use median instead
                self.name2sta[unique_name]["median"]
            return self.name2sta[unique_name][_field]

    def has_prefix(self, name):
        return DEL in name

    def export2xlsx(self, _dir, _stats=None, filename=None, sheet_name=None):
        ''' Export the statitic results to an XLSX file

        Parameters
        ----------
        _stats: list
            A list of statitic results
        _dir: str
            The directory to store the XLSX file
        '''
        if _stats is None:
            _stats = [self.name2sta]
        workbook = xlsxwriter.Workbook(os.path.join(_dir, 'statistic.xlsx' if filename is None else filename + ".xlsx"))
        for idx, _stat in enumerate(_stats):
            worksheet = workbook.add_worksheet(sheet_name[idx] if sheet_name is not None else None)
            row = 0
            header = []
            for name, statistic in sorted(_stat.items()):
                if row == 0:
                    # -- Output the header of the sheet
                    col = 0
                    worksheet.write(row, col, "Name")
                    for key in statistic:
                        col += 1
                        header.append(key)
                        worksheet.write(row, col, key)
                row += 1
                col = 0
                worksheet.write(row, col, name)
                for key in header:
                    col += 1
                    worksheet.write(row, col, statistic[key])
        workbook.close()

    def search_by_long_name(self, longname, start_idx=0):
        for idx in range(start_idx, len(self.traces)):
            ### ignore instance events
            if self.traces[idx]["ph"].lower() == "i":
                continue
            std_name = gen_long_name(self.traces[idx]["pid"], self.traces[idx]["args"]["name"])
            if std_name == longname:
                return idx, self.traces[idx]
        return None, None

    def get_iter_time(self):
        ''' print the iteration time and computation time
        *Note* that this function is strongly dependent on how the bps_trace_final.json
        is generated based on the temp.json, i.e., which ops of temp.json are used in 
        the bps_trace_final.json

        Returns
        -------
        iter_time: float, average iteration time, overall
        self.opt_step: int, the index of the step where the iteration 
            time is the cloest to the average performance, used for
            converting dynamic graph into static graph.
        '''
        return self.iter_time, self.opt_step

    def map_name2idxlist(self, name):
        ''' map the trace name to the list of indexes in the traces
        Returns
        -------
        A list of indexs, some elements may be None 

        '''
        assert self.has_prefix(name) or name == "END", name
        if name not in self.name2sta:
            return None
        return self.name2sta[name]["step_ids"]


class BiasRange:
    def __init__(self, _l, _r):
        self.l = _l
        self.r = _r

    def max_min_with_none(self, a, b, is_max=True):
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return max(a, b) if is_max else min(a, b)

    def add_with_none(self, a, b):
        if a is None or b is None:
            return None
        else:
            return a + b

    def __mul__(self, other):
        ### intersection
        nl = self.max_min_with_none(self.l, other.l, is_max=True)
        nr = self.max_min_with_none(self.r, other.r, is_max=False)
        return BiasRange(nl, nr)

    def __add__(self, other):
        nl = self.add_with_none(self.l, other.l)
        nr = self.add_with_none(self.r, other.r)
        return BiasRange(nl, nr)

    def random_gen_value(self):
        INFINITY = 1e6
        if self.l is None and self.r is not None:
            SingleLogger().warn("BiasRange selects a value in random, with a range (-inf, %f]" % self.r)
            return random.uniform(-INFINITY, self.r)
        elif self.l is not None and self.r is None:
            SingleLogger().warn("BiasRange selects a value in random, with a range [%f, inf)" % self.l)
            return random.uniform(self.l, INFINITY)
        elif self.l is None and self.r is None:
            SingleLogger().warn("BiasRange selects a value in random, with a range (-inf, inf)")
            return random.uniform(-INFINITY, INFINITY)
        else:
            return random.uniform(self.l, self.r)

    def display(self):
        print(self.displays())

    def displays(self):
        ls = "(-inf" if self.l is None else "[" + str(self.l)
        rs = "inf)" if self.r is None else str(self.r) + "]"
        return "%s, %s" % (ls, rs)


class PathManager:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir_level = self.get_dir_level(self.path)
        ### get the sub files and directories
        _, self.dirs, self.files = list(os.walk(self.path))[0]
        self.dirs = sorted([_d for _d in self.dirs if not _d.startswith(".")])

    def get_dir_level(self, _dir):
        ''' return the level of the current dir '''
        def recur_look_up(_d):
            root, dirs, files = list(os.walk(_d))[0]
            
            if FileName.COMP.value in files:
                return 0
            else:
                target_dir = None
                for _d in dirs:
                    if not _d.startswith("."):
                        target_dir = _d
                        break
                assert target_dir is not None, "No explicit directory found under {}".format(root)
                return 1 + recur_look_up(os.path.join(root, target_dir))
        try:
            level = recur_look_up(_dir)
        except:
            print(_dir)
            raise
        return DirLevel(level)

    def search_comm(self):
        return self.search(FileName.COMM.value)

    def search(self, target):
        ''' Search the target file, if not exit, return None '''
        if isinstance(target, Enum):
            target = target.value
        if target in self.files:
            return os.path.join(self.path, target)
        if self.dir_level == DirLevel.WORKER:
            for worker_dir in self.dirs:
                gpu_root, gpu_dirs, gpu_files = list(os.walk(os.path.join(self.path, worker_dir)))[0]
                if target in gpu_files:
                    return os.path.join(gpu_root, target)
        elif self.dir_level == DirLevel.TRIAL:
            for _dir in self.dirs:
                worker_root, worker_dirs, worker_files = list(os.walk(os.path.join(self.path, _dir)))[0]
                if target in worker_files:
                    return os.path.join(worker_root, target)
                else:
                    for worker_dir in worker_dirs:
                        gpu_root, gpu_dirs, gpu_files = list(os.walk(os.path.join(worker_root, worker_dir)))[0]
                        if target in gpu_files:
                            return os.path.join(gpu_root, target)
        SingleLogger().warn("Fail to find %s in path %s" % (str(target), self.path))
        return

    def ret_prefix(self):
        ''' Return the host id and rank for DirLevel.GPU
        '''
        if self.dir_level != DirLevel.GPU:
            raise ValueError("Only support DirLevel.GPU now")

        path_split = self.path.split("/")
        local_rank = int(path_split[-1])
        wk_prefix = path_split[-2]
        return wk_prefix, local_rank

    def ret_id_in_trial(self):
        if self.dir_level == DirLevel.GPU:
            wk_prefix, local_rank = self.ret_prefix()
            return wk_prefix + "-" + str(local_rank)
        elif self.dir_level == DirLevel.WORKER:
            return self.path.split("/")[-1]
        elif self.dir_level == DirLevel.TRIAL:
            return self.path
        else:
            raise ValueError()


def painted_timeline(traces, mapping, dump_path):
    ''' Paint a timeline
    * `mapping`: a function map a trace to the new name
    '''
    rst = {
        "traceEvents": [],
        "displayTimeUnit": "ms"
    }
    for trace in traces:
        trace["name"] = mapping(trace)
        rst["traceEvents"].append(trace)
    with open(dump_path, 'w') as f:
        json.dump(rst, f)

