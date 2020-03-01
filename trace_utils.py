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

def export2xlsx(_stats, _dir, filename=None, sheet_name=None):
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
        else:
            pass
    return path_dict

def combine_traces_of_one_GPU(_traces, _local_rank, _tmp_traces, _comm_filter=None):
    for event in _traces:
        if event["cat"] == "Comm" and _comm_filter is not None and event["args"]["name"] not in _comm_filter:
            #! Only show the communication nodes belonging to comm_filter if comm_filter is set
            continue
        event['pid'] = "rank%d."%_local_rank + str(event['pid'])
        event['name'] = "rank%d."%_local_rank + str(event['name'])
        _tmp_traces.append(event)

def combine_traces_of_one_path(_path, _comm_filter=None):
    ''' Combine all traces in one path, add the rank in the form of 'rank0'
        to their names as prefixes
    '''
    tmp_traces = []
    _path = os.path.abspath(_path)
    if os.path.isdir(_path):
        #! If its a directory of a worker, read all traces of all GPUs
        root, dirs, _ = list(os.walk(_path))[0]
        #! avoid that directory is like `worker/0/`
        if len(dirs) == 0:
            raise ValueError("Given path should be the root directory of a worker traces"
                " or the path of one trace TXT file")
        dirs = sorted(dirs)
        for _dir in dirs:
            path_dict = return_path_dict(os.path.join(root, _dir))
            local_rank = path_dict["local_rank"]
            traces = read_traces(path_dict["trace_path"])
            combine_traces_of_one_GPU(traces, local_rank, tmp_traces, _comm_filter=_comm_filter)
    else:
        #! Or, read just one trace file
        traces = read_traces(_path)
        local_rank = _path.split('/')[-2]
        combine_traces_of_one_GPU(traces, local_rank, tmp_traces, _comm_filter=_comm_filter)
    return tmp_traces

def group_computation_op_by_rank(traces, rank=None):
    ret_dict = {}
    def _get_rank_str(_name):
        if "rank" in event["name"]:
            _rank_str = _name.split(".")[0]
        elif rank is not None:
            _rank_str = "rank" + str(rank)
        else:
            _rank_str = "rank?"

        if _rank_str not in ret_dict:
            ret_dict[_rank_str] = []
        return _rank_str
    for event in traces:
        if event["cat"] == "operator":
            ret_dict[_get_rank_str(event["name"])].append(event)
    return ret_dict

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
    operator_traces_list = group_computation_op_by_rank(traces, rank)

    ret = []
    for _rank, operator_traces in operator_traces_list.items():
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
        ret.append((_rank, fw_bw_time, iter_time))
        SingleLogger().info("<%s> fw + bw: %f ms -- iteration time: %f ms" % (_rank,
                fw_bw_time, iter_time))
    return ret

def get_dir_level(_dir):
    ''' return the level of the current dir '''
    def recur_look_up(_d):
        root, dirs, files = list(os.walk(_d))[0]
        if "dag.gml" in files:
            return 0
        else:
            return 1 + recur_look_up(os.path.join(root, dirs[0]))
    level = recur_look_up(_dir)
    return DirLevel(level)

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
        self.dir_level = get_dir_level(self.path)
        ### get the sub files and directories
        _, self.dirs, self.files = list(os.walk(self.path))[0]
        self.dirs = sorted(self.dirs)
        ### only for DirLevel.GPU path
        self.path_dict = return_path_dict(self.path) if self.dir_level == DirLevel.GPU else None

    def search_comm(self):
        return self.search(FileName.COMM.value)

    def search(self, target):
        if isinstance(target, Enum):
            target = target.value
        ''' Search the target file, if not exit, return None '''
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


