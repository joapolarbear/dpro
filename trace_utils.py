import os
import ujson as json
import random
import xlsxwriter
import traceback
import logger_utils
from enum import Enum
from logger_utils import Singleton, SingleLogger

QUEUETYPE = {
    "NCCL": {
        "fine": [
            "NEGOTIATE_ALLREDUCE_none",
            "QUEUE",
            "NCCL_ALLREDUCE"
            ],
        "coarse": [
            "NEGOTIATE_ALLREDUCE",
            "ALLREDUCE"
            ], 
        }
}

### The delimiter bettwen the pid and the raw name in the standard name format
DEL = "->"
DDEL = "~>"

MAX_CNT = None

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
    TRACE="bps_trace_final.json"
    COMP="temp.json"
    SYMBOL="symbol_debug_str.txt"
    TENSOR_NAME="gradient_name_list.txt"
    COMM_DETAIL="comm_detail.json"
    INFO="info.json"
    NCCL_GRAPH="nccl_graph.txt"
    TRAIL_DAG="trail_dag.gml"
    STATISTIC="statistic.txt"
    BPS_COMM_DETAIL="comm_timeline.json"
    BPS_SERVER_TRACE="server_timeline.json"
    IP_TO_RANK="ip_to_rank.txt"
    KEY_DICT="key_dict.txt"
    BYTEPS_CACHE="bps_cache.pickle"
    BPS_ALIGNED_TRACE="bps_comm_aligned.json"
    TF_METADATA="metadata.json"

class CatName(Enum):
    OPERATOR="operator"
    COMM="Comm"
    IO="I/O"
    DEBUG="virtual"

class PlatformName(Enum):
    MXNET = "MXNET"
    TENSORFLOW = "TENSORFLOW"

GAP_STR_OP2OP = "%sGAP%s"%(CatName.OPERATOR.value, CatName.OPERATOR.value)
GAP_STR_COMM2COMM = "%sGAP%s"%(CatName.COMM.value, CatName.COMM.value)
GAP_STR_OP2COMM = "%sGAP%s"%(CatName.OPERATOR.value, CatName.COMM.value)
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


def return_path_dict(root_path):
    ### TODO : delete
    ''' Map the paths of each file from its name
    Args:
        root_path: the root path for one GPU
    '''
    assert os.path.isdir(root_path)
    root_path = os.path.abspath(root_path)
    __root, _, files = list(os.walk(root_path))[0]
    path_dict = {"root": __root}
    path_dict["local_rank"] = int(__root.split("/")[-1])
    for __file in files:
        cur_path = os.path.join(__root, __file)
        if FileName.TRACE.value in __file:
            path_dict[FileName.TRACE.value] = cur_path
        elif __file == FileName.DAG.value:
            # mygraph = nx.read_gml(cur_path)
            path_dict[FileName.DAG.value] = cur_path
        elif __file == FileName.COMP.value:
            path_dict[FileName.COMP.value] = cur_path
        elif __file == FileName.COMM.value:
            path_dict[FileName.COMM.value] = cur_path
        elif __file == FileName.IO.value:
            path_dict[FileName.IO.value] = cur_path
        elif "loss" in __file:
            idx = int(__file.split("loss")[1].split(".")[0])
            if "loss" not in path_dict:
                path_dict["loss"] = {idx: cur_path}
            else:
                path_dict["loss"][idx] = cur_path
        elif __file == FileName.SYMBOL.value:
            path_dict[FileName.SYMBOL.value] = cur_path
        elif __file == FileName.TENSOR_NAME.value:
            path_dict[FileName.TENSOR_NAME.value] = cur_path
        elif __file == FileName.COMM_DETAIL.value:
            path_dict[FileName.COMM_DETAIL.value] = cur_path
        else:
            pass
    return path_dict
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

def parse_allinfo_from_name(name):
    if DEL not in name:
        raw_name = name
        pid = "none"
    else:
        ns = name.split(DEL)
        pid = ns[0]
        raw_name = ns[1]

    if "I/O" in raw_name:
        return pid, raw_name, CatName.IO.value
    elif "Comm" in raw_name or "PUSH" in raw_name or "PULL" in raw_name:
        return pid, raw_name, CatName.COMM.value
    elif "FW" in raw_name or "BW" in raw_name or "COMP" in raw_name or "UPDATE" in raw_name or "OUTPUT" in raw_name or "COPY_FIRST" in raw_name or "SUM" in raw_name or "COPY_MERGED" in raw_name:
        return pid, raw_name, CatName.OPERATOR.value
    elif raw_name == "END":
        return pid, raw_name, CatName.DEBUG.value
    else:
        raise ValueError("Can not decide the cat of %s" % name)


def parse_pid_from_name(name):
    if DEL not in name:
        return "none"
    else:
        return name.split(DEL)[0]

def parse_rawname_from_name(name):
    if DEL not in name:
        return name
    else:
        return name.split(DEL)[1]

def parse_layer_name(name):
    if DEL in name:
        name = name.split(DEL)[1]
    if DDEL in name:
        name = name.split(DDEL)[0]
    if "." in name:
        name = name.split(".")[1]
    name_split = name.split("_")
    if name_split[-1] in ["gamma", "beta", "weight", "bias"]:
        return "_".join(name_split[:-1])
    else:
        return name

def parse_cat_from_name(name):
    if "I/O" in name:
        return CatName.IO.value
    elif "Comm" in name or "PUSH" in name or "PULL" in name:
        return CatName.COMM.value
    elif "FW" in name or "BW" in name or "COMP" in name or "UPDATE" in name or "OUTPUT" in name or "COPY_FIRST" in name or "SUM" in name or "COPY_MERGED" in name:
        return CatName.OPERATOR.value
    elif name == "END":
        return CatName.DEBUG.value
    else:
        raise ValueError("Can not decide the cat of %s" % name)

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
    elif "FW" in name_:
        ret_cat = "operator.FW"
    elif "BW" in name_:
        ret_cat = "operator.BW"
    elif "COMM" in name_:
        ret_cat = "operator.COMM"
    elif "COMP" in name_:
        ret_cat = "operator.COMP"
    elif "UPDATE_" in name_:
        ret_cat = "operator.UPDATE"
    elif "OUTPUT" in name_:
        ret_cat = "operator.OUTPUT"
    elif name_ == "END":
        ret_cat = "virtual"
    elif "COPY_FIRST" in name_ or "SUM" in name_ or "COPY_MERGED" in name_:
        return "operator.SERVERCOMP"
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

        self.name2sta = None
        self.cat2sta = None
        self.dir_level = dir_level

        self.max_cnt = 0
        self.ret_stat()

    def check_traces(self, traces):
        for trace in traces:
            if trace.get("name", None) is None or trace.get("ts", None) is None:
                print(trace)
                raise
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

    def ret_stat(self):
        """ Basic Statistic """
        self.name2sta = {}
        self.cat2sta = {}
        for event in self.traces:
            if self._is_ignore_for_sta(event):
                continue
            unique_name = self.ret_unique_name(event)
            if unique_name in self.name2sta:
                if MAX_CNT is not None and self.name2sta[unique_name]["cnt"] >= MAX_CNT:
                    event["args"]["cnt"] = -1
                    continue
                self.name2sta[unique_name]["cnt"] += 1
                self.name2sta[unique_name]["time"] += event["dur"] / 1000.0
                self.name2sta[unique_name]["min_t"] = min(self.name2sta[unique_name]["min_t"], event["dur"] / 1000.0)
                self.name2sta[unique_name]["max_t"] = max(self.name2sta[unique_name]["max_t"], event["dur"] / 1000.0)
            else:
                self.name2sta[unique_name] = {
                    "cnt": 1, 
                    "time": event["dur"] / 1000.0, 
                    "min_t": event["dur"] / 1000.0, 
                    "max_t": event["dur"] / 1000.0,
                    # \TODO: add `cat` field for communication traces
                    # "cat": event["cat"] 
                    "cat": event["cat"],
                    "id": len(self.name2sta)
                    }
            event["args"]["cnt"] = self.name2sta[unique_name]["cnt"] - 1
                
        """calculate the avg """
        for name, statistic in self.name2sta.items():
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
            self.cat2sta[cat]["time"] += statistic["time"]
            self.cat2sta[cat]["cnt"] += statistic["cnt"]
            self.cat2sta[cat]["op_cnt"] += 1

        for cat, statistic in self.cat2sta.items():
            statistic["avg"] = statistic["time"] / statistic["cnt"]

        """calculate the variance"""
        for event in self.traces:
            if self._is_ignore_for_sta(event):
                continue
            unique_name = self.ret_unique_name(event)
            self.name2sta[unique_name]["var"] += pow(event["dur"] / 1000.0 - self.name2sta[unique_name]["avg"], 2)
        for name, statistic in self.name2sta.items():
            statistic["var"] = statistic["var"] / float(statistic["cnt"])
            self.max_cnt = max(statistic["cnt"], self.max_cnt)

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
        '''
        assert isinstance(self.traces, list)
        operator_traces_list = self.group_computation_op_by_prefix()

        ret = []
        for prefix in sorted(operator_traces_list.keys()):
            operator_traces = operator_traces_list[prefix] 
            fw_bw_list = []
            iter_list = []
            operator_traces = sorted(operator_traces, key=lambda x: x["ts"])

            iter_cnt = None
            step_start_ts = None
            cur_iter_time = 0
            fw_bw_end = 0
            for event in operator_traces:
                if iter_cnt is None:
                    ### initialization
                    step_start_ts = event['ts']
                    cur_iter_time = event['ts'] + event['dur']
                    iter_cnt = event["args"]["cnt"]
                elif iter_cnt != event["args"]["cnt"]:
                    ### a new iteration
                    assert step_start_ts is not None
                    if iter_cnt == -1:
                        continue
                    if event["args"]["cnt"] < iter_cnt:
                        SingleLogger().warn("Illegal cnt field for this event %s %s %d" % (event["pid"], event["name"], event["args"]["cnt"]))
                        continue
                    iter_list.append((cur_iter_time - step_start_ts) / 1000.0)
                    fw_bw_list.append((fw_bw_end - step_start_ts) / 1000.0)
                    # SingleLogger().info("%s - the %d th iteration: FW+BW: %f, Iteration time: %f" % (prefix, len(iter_list), fw_bw_list[-1], iter_list[-1]))
                    SingleLogger().debug("%s - the %d th iteration: FW+BW: %f, Iteration time: %f" % (prefix, len(iter_list), fw_bw_list[-1], iter_list[-1]))
                    step_start_ts = event['ts']
                    cur_iter_time = event['ts'] + event['dur']
                    iter_cnt = event["args"]["cnt"]
                else:
                    ### during an iteration
                    cur_iter_time = event['ts'] + event['dur']

                    ### TODO (huhanpeng): change after fine-tune update
                    ### here we assume UPDATE is following the last BP op.
                    if "FW" in event["name"] or "BW" in event["name"]:
                        fw_bw_end = cur_iter_time
                    
            ### Needed if there is only one step
            iter_list.append((cur_iter_time - step_start_ts) / 1000.0)
            fw_bw_list.append((fw_bw_end - step_start_ts) / 1000.0)
            SingleLogger().debug("%s - the %d th iteration: FW+BW: %f, Iteration time: %f" % (prefix, len(iter_list), fw_bw_list[-1], iter_list[-1]))

            fw_bw_time = sum(fw_bw_list) / float(len(fw_bw_list))
            iter_time = sum(iter_list) / float(len(iter_list))
            ret.append((prefix, fw_bw_time, iter_time))
            SingleLogger().info("<%s> fw + bw: %f ms -- iteration time: %f ms" % (prefix,
                    fw_bw_time, iter_time))
        return ret

    def group_computation_op_by_prefix(self):
        prefix2traces = {}
        def _get_prefix(e):
            prefix = e["pid"]
            if prefix not in prefix2traces:
                prefix2traces[prefix] = []
            return prefix
        for event in self.traces:
            if event["cat"] == "operator" and not self._is_ignore_for_sta(event):
                prefix2traces[_get_prefix(event)].append(event)
        return prefix2traces

    def dump(self, dir_):
        rst_traces = sorted(self.traces, key=lambda x: (x["pid"], x["tid"]))
        with open(os.path.join(dir_, FileName.TRACE.value), 'w') as f:
            json.dump(rst_traces, f)

        str_ = "%d,%d\n"%(self.dir_level.value, self.max_cnt)
        str_ += str(self.name2sta) + "\n"
        str_ += str(self.cat2sta)
        with open(os.path.join(dir_, FileName.STATISTIC.value), 'w') as fp:
            fp.write(str_)

    def load(self, dir_):
        self.traces = read_traces(os.path.join(dir_, FileName.TRACE.value))

        with open(os.path.join(dir_, FileName.STATISTIC.value), 'r') as fp:
            str_ = fp.read().split("\n")

        dir_level, max_cnt = str_[0].split(",")
        self.dir_level = DirLevel(int(dir_level))
        self.max_cnt = int(max_cnt)

        self.name2sta = eval(str_[1])
        self.cat2sta = eval(str_[2])


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
        self.dirs = sorted(self.dirs)
        ### only for DirLevel.GPU path
        self.path_dict = return_path_dict(self.path) if self.dir_level == DirLevel.GPU else None

    def get_dir_level(self, _dir):
        ''' return the level of the current dir '''
        def recur_look_up(_d):
            root, dirs, files = list(os.walk(_d))[0]
            if "dag.gml" in files:
                return 0
            else:
                return 1 + recur_look_up(os.path.join(root, dirs[0]))
        level = recur_look_up(_dir)
        return DirLevel(level)

    def search_comm(self):
        return self.search(FileName.COMM.value)

    def search(self, target):
        ''' Search the target file, if not exit, return None '''
        if isinstance(target, Enum):
            target = target.value
        
        if self.dir_level == DirLevel.GPU: 
            if target in self.path_dict:
                return self.path_dict[target]
            else:
                ### only allow to traceback to one upper folder
                parent_path = os.path.dirname(self.path)
                root, dirs, files = list(os.walk(parent_path))[0]
                if target in files:
                    return os.path.join(root, target)
                else:
                    SingleLogger().warn("Fail to find %s in path %s" % (str(target), self.path))
                    return
        elif self.dir_level == DirLevel.WORKER:
            if target in self.files:
                return os.path.join(self.path, target)
            else:
                SingleLogger().warn("Fail to find %s in path %s" % (str(target), self.path))
                return
        elif self.dir_level == DirLevel.TRIAL:
            if target in self.files:
                return os.path.join(self.path, target)
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

    """
    def fuzzy_search(self, target):
        '''fuzzy search, return a list whose elements contain target'''
        assert self.dir_level == DirLevel.TRIAL
        ret = []

        def lookup_files(_path, _files):
            for file in _files:
                if target in file:
                    ret.append(os.path.join(_path, file))

        lookup_files(self.path, self.files)
        for _dir in self.dirs:
            worker_root, worker_dirs, worker_files = list(os.walk(os.path.join(self.path, _dir)))[0]
            lookup_files(worker_root, worker_files)
            for worker_dir in worker_dirs:
                gpu_root, gpu_dirs, gpu_files = list(os.walk(os.path.join(worker_root, worker_dir)))[0]
                lookup_files(gpu_root, gpu_files)

        return ret
    """

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




