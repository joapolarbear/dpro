import os
import json
import xlsxwriter
import traceback
import logger_utils
from logger_utils import Singleton, SingleLogger

QUEUETYPE = {
    "NCCL": {
        "fine": [
            "NEGOTIATE_ALLREDUCE",
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

@Singleton
class QueueType:
    def __init__(self, backend, fine_grained=True):
        self.queue_type_list = QUEUETYPE[backend]["fine" if fine_grained else "coarse"]

    def ret_list(self):
        return self.queue_type_list

from enum import Enum
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

class TraceManager:
    def __init__(self, traces, dir_level=None):
        self.traces = traces
        self.traces = sorted(self.traces, key=lambda x: (x["ts"], x["name"]), reverse=False)

        self.name2sta = None
        self.cat2sta = None
        self.dir_level = dir_level

        self.ret_stat()

    def _is_comm_trace(self, event):
        return event["cat"] == "Comm"

    def ret_unique_name(self, event):
        return event["pid"] + DEL + event["name"]

    def ret_stat(self):
        """ Basic Statistic """
        self.name2sta = {}
        self.cat2sta = {}
        for event in self.traces:
            if event["ph"].lower() == "i":
                continue
            unique_name = self.ret_unique_name(event)
            if unique_name in self.name2sta:
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
                    "cat": event["cat"]
                    }
                
        """calculate the avg """
        for name, statistic in self.name2sta.items():
            statistic["avg"] = statistic["time"] / statistic["cnt"]
            statistic["var"] = 0.0
            cat = statistic["cat"]
            if cat in self.cat2sta:
                if statistic["avg"] > self.cat2sta[cat]["max_t"]:
                    self.cat2sta[cat]["max_t"] = statistic["avg"]
                    self.cat2sta[cat]["max_name"] = name
            else:
                self.cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name}

        """calculate the variance"""
        for event in self.traces:
            if event["ph"].lower() == "i":
                continue
            unique_name = self.ret_unique_name(event)
            self.name2sta[unique_name]["var"] += pow(event["dur"] / 1000.0 - self.name2sta[unique_name]["avg"], 2)

        for name, statistic in self.name2sta.items():
            statistic["var"] = statistic["var"] / float(statistic["cnt"])

    def lookup_stat(self, wk_prefix, rank_prefix, name, with_prefix=False,  _field="avg"):
        ''' look up data from the stat info
        Parameters
        __________
        with_prefix: boolean
            if True, name has had prefix, no need to process it
        '''
        if self.dir_level == DirLevel.GPU or with_prefix:
            unique_name = name
        elif self.dir_level == DirLevel.WORKER:
            unique_name = "%s%s%s"%(rank_prefix, DEL, name)
        elif self.dir_level == DirLevel.TRIAL:
            unique_name = "%s.%s%s%s"%(wk_prefix, rank_prefix, DEL, name)

        if unique_name not in self.name2sta:
            # SingleLogger().warn("Fail to find the trace of %s" % unique_name)
            return 0.0
        else:
            return self.name2sta[unique_name][_field]

    def export2xlsx(self, _stats, _dir, filename=None, sheet_name=None):
        ''' Export the statitic results to an XLSX file

        Parameters
        ----------
        _stats: list
            A list of statitic results
        _dir: str
            The directory to store the XLSX file
        '''
        workbook = xlsxwriter.Workbook(os.path.join(_dir, 'statistic.xlsx' if filename is None else filename + ".xlsx"))
        for idx, _stat in enumerate(_stats):
            worksheet = workbook.add_worksheet(sheet_name[idx] if sheet_name is not None else None)
            row = 0
            header = []
            for name, statistic in _stat.items():
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

def split_name(_name):
    try:
        name_split = _name.split(".")
        _local_rank = int(name_split[0].split("rank")[1])
        raw_name = ".".join(name_split[1:])
    except:
        raise ValueError("split_name error: " + _name)
    return _local_rank, raw_name

def lookup_stat(name2sta, _name, _field="avg"):
    ''' look up data from the stat info
    * if name2sta is the stat info of entire worker, _name should contain rank id
    * else, name2sta is the stat info of one GPU
    '''
    if "rank" not in _name:
        return name2sta[_name][_field] if _name in name2sta else 0.0
    _local_rank, _raw_name = split_name(_name)
    return name2sta["traces"][_local_rank][_raw_name][_field]

def return_path_dict(root_path):
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

def parse_cat_from_name(name):
    if "I/O" in name:
        return "I/O"
    elif "Comm" in name:
        return "Comm"
    elif "FW" in name or "BW" in name or "STEP" in name:
        return "operator"
    else:
        raise ValueError("Can not decide the cat of %s" % name)

def group_computation_op_by_prefix(traces, rank=None):
    prefix2traces = {}
    def _get_prefix(e):
        prefix = e["pid"]
        if prefix not in prefix2traces:
            prefix2traces[prefix] = []
        return prefix
    for event in traces:
        if event["cat"] == "operator":
            prefix2traces[_get_prefix(event)].append(event)
    return prefix2traces

def get_iter_time(traces, rank=None):
    ''' print the iteration time and computation time
    *Note* that this function is strongly dependent on how the bps_trace_final.json
    is generated based on the temp.json, i.e., which ops of temp.json are used in 
    the bps_trace_final.json
    '''
    if isinstance(traces, dict):
        traces = traces["traceEvents"]
    else:
        assert isinstance(traces, list)
    operator_traces_list = group_computation_op_by_prefix(traces, rank)

    ret = []
    for prefix in sorted(operator_traces_list.keys()):
        operator_traces = operator_traces_list[prefix]
        start_ts = None
        cur_iter_time = 0
        fw_bw_list = []
        iter_list = []
        operator_traces = sorted(operator_traces, key=lambda x: x["ts"])
        for event in operator_traces:
            if start_ts is None:
                start_ts = event['ts']
            ### here we assume STEP is following the last BP op.
            if "STEP" in event["name"]:
                fw_bw_list.append((cur_iter_time - start_ts) / 1000.0)
            cur_iter_time = event['ts'] + event['dur']
            if "STEP" in event["name"]:
                iter_list.append((cur_iter_time - start_ts) / 1000.0)
                start_ts = None
        fw_bw_time = sum(fw_bw_list) / float(len(fw_bw_list))
        iter_time = sum(iter_list) / float(len(iter_list))
        ret.append((prefix, fw_bw_time, iter_time))
        SingleLogger().info("<%s> fw + bw: %f ms -- iteration time: %f ms" % (prefix,
                fw_bw_time, iter_time))
    return ret

def read_list(path):
    ''' read a list from a file
    '''
    with open(path, 'r') as f:
        l = f.read().split("\n")
    l = l[:-1] if l[-1] == '' else l
    return l

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
            return "TRIAL_ROOT"
        else:
            raise ValueError()

