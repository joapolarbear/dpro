import os
import ujson as json
import random
import math
import xlsxwriter
import threading
from enum import Enum
import numpy as np

from logger_utils import Singleton, SingleLogger
from base import bcolors

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
    COMM="comm.json"
    IO="io.json"
    DAG="dag.gml"
    GRAPHDEF = "final_graph.json"
    TRACE="bps_trace_final.json"
    COMP = "trace.json.gz"
    SYMBOL="symbol_debug_str.txt"
    TENSOR_NAME="gradient_name_list.json"
    COMM_DETAIL="comm_detail.json"
    INFO="info.json"
    NCCL_GRAPH="nccl_graph.txt"
    NCCL_RANK_GRAPH="nccl_rank_graph.json"
    TRAIL_DAG="trail_dag.gml"
    LOCAL_DFG="local_dfg.gml"
    STATISTIC="statistic.txt"
    BPS_COMM_DETAIL="comm_timeline.json"
    BPS_SERVER_TRACE="server_timeline.json"
    IP_TO_RANK="ip_to_rank.txt"
    KEY_DICT="key_dict.txt"
    BYTEPS_CACHE="bps_cache.pickle"
    BPS_ALIGNED_TRACE="bps_comm_aligned.json"
    METADATA="metadata.json"    

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

def _is_comm_trace(trace):
    return trace["cat"] == "Comm"

def first_valid_dir(_path):
    for _dir in os.listdir(_path):
        if _dir.startswith('.'):
            continue
        return _dir
    raise ValueError("No valid directory under {}".format(_path))

######################## Delete ########################
def return_stat(traces):
    """ Basic Statistic """
    name2sta = {}
    cat2sta = {}
    for event in traces:
        name = event["name"]
        if _is_comm_trace(event):
            name = event["args"]["name"] + "." + event["name"]
        if name in name2sta:
            name2sta[name]["cnt"] += 1
            name2sta[name]["time"] += event["dur"] / 1000.0
            name2sta[name]["min_t"] = min(name2sta[name]["min_t"], event["dur"] / 1000.0)
            name2sta[name]["max_t"] = max(name2sta[name]["max_t"], event["dur"] / 1000.0)
        else:
            name2sta[name] = {"cnt": 1, "time": event["dur"] / 1000.0, 
                "min_t": event["dur"] / 1000.0, "max_t": event["dur"] / 1000.0,
                # \TODO: add `cat` field for communication traces
                # "cat": event["cat"] 
                "cat": event["name"].split(".")[0]
                }
            
    """calculate the avg """
    for name, statistic in name2sta.items():
        statistic["avg"] = statistic["time"] / statistic["cnt"]
        statistic["var"] = 0.0
        cat = statistic["cat"]
        if cat in cat2sta:
            if statistic["avg"] > cat2sta[cat]["max_t"]:
                cat2sta[cat]["max_t"] = statistic["avg"]
                cat2sta[cat]["max_name"] = name
        else:
            cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name}

    """calculate the variance"""
    for event in traces:
        name = event["name"]
        if _is_comm_trace(event):
            name = event["args"]["name"] + "." + event["name"]
        name2sta[name]["var"] += pow(event["dur"] / 1000.0 - name2sta[name]["avg"], 2)

    for name, statistic in name2sta.items():
        statistic["var"] = statistic["var"] / float(statistic["cnt"])
    return name2sta, cat2sta

######################## Delete ########################


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

def parse_suffix_from_name(name):
    if DDEL in name:
        return name.split(DDEL)[0], name.split(DDEL)[1]
    else:
        return name, None

def parse_pid_from_name(name):
    if DEL not in name:
        return "none"
    else:
        return name.split(DEL)[0]

def parse_rawname(name):
    if DEL not in name:
        return name
    else:
        return name.split(DEL)[1]

def parse_op_name(name):
    if DEL in name:
        name = name.split(DEL)[1]
    if DDEL in name:
        name = name.split(DDEL)[0]
    if "." in name:
        name = name.split(".")[1]
    # name_split = name.split("_")
    # if name_split[-1] in ["gamma", "beta", "weight", "bias"]:
    #     return "_".join(name_split[:-1])
    # else:
    #     return name
    return name

def _parse_tf_layer_names(name):
    layer_names = []
    if "+" in name:
        for _name in name.split("+"):
            layer_names += _parse_tf_layer_names(_name)
        return layer_names
    op_name_split = parse_op_name(name).split("/")
    op_type = "FW"
    idx = 0
    if op_name_split[idx] == "gradients":
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
    
def parse_cat_from_name(name):
    if "I/O" in name:
        return CatName.IO.value
    elif "Comm" in name or "PUSH" in name or "PULL" in name:
        return CatName.COMM.value
    elif "FW" in name or "BW" in name or "COMP" in name or "UPDATE" in name or "OUTPUT" in name:
        return CatName.OPERATOR.value
    elif "COPY_FIRST" in name or "SUM" in name or "COPY_MERGED" in name:
        return CatName.PS_SERVER_OPERATOR.value
    elif name == "END":
        return CatName.DEBUG.value
    else:
        raise ValueError("Can not decide the cat of %s" % name)

def parse_allinfo_from_name(name):
    if DEL not in name:
        std_name = name
        pid = "none"
    else:
        pid, std_name = name.split(DEL)

    suffix = None
    if DDEL in std_name:
        std_name, suffix = std_name.split(DDEL)

    cat = parse_cat_from_name(std_name)
    
    return pid, std_name, cat, suffix

### CATs that will be affected if we change the GPU/CPU rate
COMP_CAT = ["operator.FW", "operator.BW", "operator.UPDATE",
    "operator.OUTPUT", "ServerOp"]
### CATs that will be affected if we change the bandwidth
### TODO (huhanpeng): check whether it is correct for BytePS
COMM_CAT = ["Comm.SEND", "Comm.RECV", "Comm.PUSH_REQ",
            "Comm.PUSH_RES", "Comm.PULL_REQ", "Comm.PULL_RES"]
def parse_cat_fine_grained(name_):
    if "Comm" in name_:
        if "SEND" in name_:
            ret_cat = "Comm.SEND"
        elif "RECV" in name_:
            ret_cat = "Comm.RECV"
        else:
            ret_cat = "Comm.other"
    elif "PUSH_REQ" in name_:
        return "Comm.PUSH_REQ"
    elif "PUSH_RES" in name_:
        return "Comm.PUSH_RES"
    elif "PULL_REQ" in name_:
        return "Comm.PULL_REQ"
    elif "PULL_RES" in name_:
        return "Comm.PULL_RES"
    elif "I/O" in name_:
        ret_cat = "I/O"
    elif "BW" in name_ and "FW" in name_:
        ret_cat = "operator.FW+BW"
    elif "BW" in name_:
        ret_cat = "operator.BW"
    elif "FW" in name_:
        ret_cat = "operator.FW"
    elif "COMM" in name_:
        ret_cat = "operator.COMM"
    elif "COMP" in name_:
        # TODO (delete)
        ret_cat = "operator.COMP"
    elif "UPDATE_" in name_:
        ret_cat = "operator.UPDATE"
    elif "OUTPUT" in name_:
        ret_cat = "operator.OUTPUT"
    elif name_ == "END":
        ret_cat = "virtual"
    elif "COPY_FIRST" in name_ or "SUM" in name_ or "COPY_MERGED" in name_:
        return "ServerOp"
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
        self.ret_stat()

    def dump(self, dir_):
        trace_thread = threading.Thread(target=self._dump, args=(dir_,))
        trace_thread.start()

    def _dump(self, dir_):
        rst_traces = sorted(self.traces, key=lambda x: (x["pid"], x["tid"]))
        with open(os.path.join(dir_, FileName.TRACE.value), 'w') as f:
            json.dump(rst_traces, f)
        str_ = "%d,%d,%d,%f\n"%(self.dir_level.value, self.max_step, self.opt_step, self.iter_time)
        str_ += str(self.name2sta) + "\n"
        str_ += str(self.cat2sta)
        with open(os.path.join(dir_, FileName.STATISTIC.value), 'w') as fp:
            fp.write(str_)

    def load(self, dir_):
        self.traces = read_traces(os.path.join(dir_, FileName.TRACE.value))
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
        
    def _is_comm_trace(self, event):
        return event["cat"] == "Comm"

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
            if unique_name in self.name2sta:
                if MAX_CNT is not None and self.name2sta[unique_name]["cnt"] >= MAX_CNT:
                    event["args"]["cnt"] = -1
                    continue
                self.name2sta[unique_name]["cnt"] += 1
                if cal_median:
                    self.name2sta[unique_name]["time"].append(event["dur"] / 1000.0)
                else:
                    self.name2sta[unique_name]["time"] += event["dur"] / 1000.0
                self.name2sta[unique_name]["min_t"] = min(self.name2sta[unique_name]["min_t"], event["dur"] / 1000.0)
                self.name2sta[unique_name]["max_t"] = max(self.name2sta[unique_name]["max_t"], event["dur"] / 1000.0)
            else:
                self.name2sta[unique_name] = {
                    "cnt": 1, 
                    "time": [event["dur"] / 1000.0] if cal_median else event["dur"] / 1000.0, 
                    "min_t": event["dur"] / 1000.0, 
                    "max_t": event["dur"] / 1000.0,
                    "cat": cat,
                    "id": len(self.name2sta)
                    }
            event["args"]["cnt"] = self.name2sta[unique_name]["cnt"] - 1

            pid_info = prefix_dict[prefix]
            if pid_info["time_base"] is None:
                pid_info["time_base"] = event["ts"]
            ### Add the `step` field
            if "step" not in event["args"]:
                ### TODO (huhanpeng): Can this adapt to MXNet
                if pid_info["time_cursor"] is None:
                    pass
                elif event["ts"] - pid_info["time_cursor"] - pid_info["time_base"] > ITER_GAP_LOWER_BOUND_US and \
                        cat in AS_START_CAT and \
                        pid_info["cat_cnt"]["operator.BW"] > 0 and \
                        pid_info["cat_cnt"]["operator.UPDATE"] > 0:
                    pid_info["step_cnt"] += 1
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
                pid_info["cat_cnt"][cat] += event["dur"] / 1000.0
            self.max_step = max(event["args"]["step"], self.max_step)

            ### Calculate the iteration time
            ### only check the iteration time when current node is FW/BW/UPDATE op
            # * and for byteps traces, there exists pids in the form like server_3_t2....
            #   do not need to calculate iteration time for those pids
            if parse_cat_from_name(event["name"]) != CatName.OPERATOR.value or \
                    not (prefix.startswith("host") or prefix.startswith("traces_")):
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

        ### Aggregate duration of all iterations
        iter_list_all = []
        step_num_upper = None
        for prefix in sorted(prefix_dict.keys()):
            if not (prefix.startswith("host") or prefix.startswith("traces_")):
                continue
            pid_info = prefix_dict[prefix]

            ### Check and statistic the LAST iteration
            if not pid_info["fw_end"] or not pid_info["bw_end"] or not pid_info["bw_start"]:
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
    
        ### calculate the average iteration time
        # * iter_list_all, shape = (n_GPUs, n_steps) ==> (n_steps)
        iter_list_all = [_list[:step_num_upper] for _list in iter_list_all]
        iter_list_all = np.average(np.array(iter_list_all), axis=0)
        self.iter_time = np.average(iter_list_all)
        _std = np.std(iter_list_all)
        STD_CHECK_THESHOLD = 0.1
        if _std / self.iter_time > STD_CHECK_THESHOLD:
            SingleLogger().info(
                "Std.dev is large compared to Ave. ({:.3f}/{:.3f}), remove the first step by default".format(_std, self.iter_time))
            self.iter_time = np.average(iter_list_all[1:])
            _std = np.std(iter_list_all[1:])
            self.opt_step = np.argmin(np.abs(iter_list_all - self.iter_time))
            SingleLogger().info("<Overall> step %d is the one closest to average %f (\u00B1 %f) ms - %s" %
                                (self.opt_step, self.iter_time, _std, iter_list_all))
        else:
            self.opt_step = np.argmin(np.abs(iter_list_all - self.iter_time))
            SingleLogger().info("<Overall> step %d is the one closest to average %f (\u00B1 %f) ms - %s" %
                                (self.opt_step, self.iter_time, _std, iter_list_all))

        """calculate the avg """
        for name, statistic in self.name2sta.items():
            ### TODO (huhanpeng), var can be calculated directly with avg list
            if cal_median:
                statistic["avg"] = sum(statistic["time"]) / statistic["cnt"]
                statistic["median"] = sorted(statistic["time"])[int(statistic["cnt"]/2)]
            else:
                statistic["avg"] = statistic["time"] / statistic["cnt"]
            statistic["var"] = 0.0

            # assert statistic["time"] != 0
            cat = parse_cat_fine_grained(name)
            if cat in self.cat2sta:
                if statistic["avg"] > self.cat2sta[cat]["max_t"]:
                    self.cat2sta[cat]["max_t"] = statistic["avg"]
                    self.cat2sta[cat]["max_name"] = name
            else:
                self.cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name, "time": 0, "cnt": 0, "op_cnt":0}
            self.cat2sta[cat]["time"] += sum(statistic["time"]) if cal_median else statistic["time"]
            self.cat2sta[cat]["cnt"] += statistic["cnt"]
            self.cat2sta[cat]["op_cnt"] += 1

        for cat, statistic in self.cat2sta.items():
            statistic["avg"] = statistic["time"] / statistic["cnt"]

        """calculate the variance"""
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

    def lookup_stat(self, wk_prefix, rank_prefix, name,  _field="avg"):
        ''' look up data from the stat info, return average time in ms by default
        Parameters
        __________
        with_prefix: boolean
            if True, name has had prefix, no need to process it
        '''
        if self.dir_level == DirLevel.GPU or self.has_prefix(name):
            unique_name = name
        elif self.dir_level == DirLevel.WORKER:
            unique_name = gen_long_name(rank_prefix, name)
        elif self.dir_level == DirLevel.TRIAL:
            unique_name = gen_long_name("%s.%s"%(wk_prefix, rank_prefix), name)
        else:
            raise RuntimeError("Unsupported DirLevel.")

        if unique_name not in self.name2sta:
            # SingleLogger().warn("Fail to find the trace of %s" % unique_name)
            return 0.0
        else:
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
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return a + b

    def __mul__(self, other):
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
        rank_prefix = "rank%s"%(path_split[-1])
        wk_prefix = path_split[-2]
        return wk_prefix, rank_prefix

    def ret_id_in_trial(self):
        if self.dir_level == DirLevel.GPU:
            wk_prefix, rank_prefix = self.ret_prefix()
            return wk_prefix + "." + rank_prefix
        elif self.dir_level == DirLevel.WORKER:
            return self.path.split("/")[-1]
        elif self.dir_level == DirLevel.TRIAL:
            return self.path
        else:
            raise ValueError()




