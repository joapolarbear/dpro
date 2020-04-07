import warnings
import os

# append for auto_profiling
import logging
import sys, os
import json
import networkx as nx
import threading
import time
import logger_utils

from trace_utils import *
from dag_utils import * 

class RunningSpan:
    def __init__(self):
        self.reset_span()
        self.disable = False

    def init_start(self, s):
        if self.start is None:
            self.start = s

    def init_end(self, e):
        ### allow to override
        self.end = e

    def if_start(self, t):
        if self.disable:
            return True

        if self.start is not None and t < self.start:
            return False
        else:
            return True

    def if_end(self, t):
        if self.disable:
            return False

        if self.end is not None and t >= self.end:
            return True
        else:
            return False

    def reset_span(self):
        self.start = None
        self.end = None

class ClockAligner:
    def __init__(self):
        self.traces_per_host = {}
        self.marker_per_host = {}
        self.standard = None
        self.ref = None

    def append_traces(self, host_id, traces):
        if host_id not in self.traces_per_host:
            self.traces_per_host[host_id] = []
            self.marker_per_host[host_id] = None
        self.traces_per_host[host_id] += traces

    def mark_ref_time(self, host_id, _t, _override=False):
        if host_id not in self.traces_per_host:
            self.traces_per_host[host_id] = []
            self.marker_per_host[host_id] = None

        if self.marker_per_host[host_id] is None or _override:
            self.marker_per_host[host_id] = _t

    def align(self):
        rst_traces = []
        host_ids = list(self.traces_per_host.keys())
        standard_time = self.marker_per_host[host_ids[0]]
        for host_id, traces in self.traces_per_host.items():
            if host_id == host_ids[0]:
                pass
            else:
                bias = standard_time - self.marker_per_host[host_id]
                for trace in traces:
                    trace["ts"] += bias
            rst_traces += traces
        return rst_traces

class Collector(object):
    #! class used to go through the file system and collect info
    def __init__(self, root_path):
        self.logger = logger_utils.SingleLogger()
        self.pm = PathManager(root_path)
        self.time_dict = None
        self.run_span = {}
        ### Used for clock synchronization when combime traces from multiple machines
        self.clock_aligner = None
        ### TODO (huhanpeng): assume different host use the same dag
        self.dag = None

    def update_final_traces(self, _io=False, _comm=False, _operator=False):
        trace_path = self.pm.search(FileName.TRACE)
        assert os.path.exists(trace_path)
        self.logger.info("Updating " + trace_path) 
        with open(trace_path, 'r') as f:
            self.time_dict = json.load(f)

        if _io is True:
            self.delete_traces_by_cat("I/O")
            self.bpf_collect_io()

        if _comm is True:
            self.delete_traces_by_cat("Comm")
            self.bpf_collect_comm()

        if _operator is True:
            self.delete_traces_by_cat("operator")
            self.bpf_collect_comp()

        get_iter_time(self.time_dict)

        with open(trace_path, 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def delete_traces_by_cat(self, _cat):
        _rst_traces = {"traceEvents": []}
        _rst_traces["traceEvents"] = [_trace for _trace in self.time_dict["traceEvents"] if _trace["cat"] != _cat]
        self.time_dict = _rst_traces

    def re_gen_final_traces(self):
        trace_path = self.pm.search(FileName.TRACE)
        self.logger.info("Recombining " + trace_path)
        self.time_dict = {"traceEvents":[]}
        ### Apply dependencies in dag to the mxnet traces.
        self.bpf_collect_comp()
        ### Collect communication traces, IO traces and STEP traces and apply dependency
        self.bpf_collect_io()
        self.bpf_collect_comm()
        get_iter_time(self.time_dict)
        with open(trace_path, 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def bpf_collect_non_comm(self, _path, pid, host_id=None):
        tmp_pm = PathManager(_path)
        assert tmp_pm.dir_level == DirLevel.GPU
        self.bpf_collect_comp(tmp_pm, pid, host_id)
        self.bpf_collect_io(tmp_pm, pid, host_id)
        self.bpf_collect_comm_detail(tmp_pm, pid, host_id)

    def bpf_collect_comp(self, tmp_pm=None, pid=None, host_id=None):
        '''Apply dependency info to the mxnet trace results

        Parameters
        ----------
        _path : dict
            if _path is not given, use the `pm` of the object
            or, if _path is given, it should be the computation_path.

        Returns
        ----------
        rst_traces : dict
            A dict containing MXNet trace results combined with dependency info.
        '''
        comp_path = self.pm.search(FileName.COMP) if tmp_pm is None else tmp_pm.search(FileName.COMP)
        if self.dag is None:
            dag_path = self.pm.search(FileName.DAG) if tmp_pm is None else tmp_pm.search(FileName.DAG)
            self.dag = nx.read_gml(dag_path)

        if comp_path is None:
            return

        wk_prefix, _ = PathManager("/".join(comp_path.split('/')[:-1])).ret_prefix()
        if wk_prefix not in self.run_span:
            self.run_span[wk_prefix] = RunningSpan()


        ''' Output trace resutls '''
        with open(comp_path, 'r') as f:
            mxnet_traces = json.load(f)
        
        one_pid = None
        rst_traces = {"traceEvents": []}

        ### convert the ph from B/E to X
        ### TODO(huhanpeng): it's more safe to use a stack style method
        index = 0
        traces = []
        while index < len(mxnet_traces["traceEvents"]):
            if "ts" not in mxnet_traces["traceEvents"][index]:
                index += 1
                continue
            trace = mxnet_traces["traceEvents"][index]
            if trace["ph"] == 'B' or trace["ph"] == 'b':
                next_trace = mxnet_traces["traceEvents"][index+1]
                assert trace["name"] == next_trace["name"]
                trace["dur"] = next_trace['ts'] - trace['ts']
                trace["ph"] = "X"
                traces.append(trace)
                index += 2
            else:
                index += 1

        traces = sorted(traces, key=lambda x: x["ts"], reverse=False)

        def standar_name(_name):
            '''Fetch and handle the trace name'''
            #! add for mxnet-gluon case
            if "name=" in _name:
                _name = _name.split("name=")[1].split(";")[0]
            #! backward nodes or forward nodes
            _name = "BW." + _name.split("_backward")[0] if "_backward" in _name else "FW." + _name
            _name = _name.split("_fwd")[0] if "_fwd" in _name else _name
            return _name 

        ### find the last BP op in the timeline
        def real_last_bw_name():
            statue = "init"
            _index = 0
            tmp = None
            while _index < len(traces):
                trace = traces[_index]
                _index += 1
                name = standar_name(trace["name"])
                if name not in self.dag.nodes:
                    continue
                if statue == "init" and "FW" in name:
                    statue = "fw"
                elif statue == "fw" and "BW" in name:
                    statue = "bw"
                    tmp = name
                elif statue == "bw" and "BW" in name:
                    tmp = name
                elif statue == "bw" and "FW" in name:
                    statue = "fw"
                    return tmp
        _real_last_bw_name = real_last_bw_name()

        IGNORE_OP = ["DeleteVariable", "sum", "_plus_scalar", 
                "_copyto_GPU2GPU", "broadcast_add", 
                "Reshape", "Cast", "_arange", "elemwise_add",
                "_ones", "SyncCopyGPU2CPU", "_mul_scalar",
                "CopyGPU2CPU", "CopyCPU2GPU", "SetValueOp"]

        def is_update_op(_trace):
            ### TODO (huhanpeng) !!! change this when model is changed
            if "update" in _trace["name"]:
                return True
            if "operator" != _trace["cat"]:
                return False
            if _trace["name"] in IGNORE_OP:
                return False
            if "backward" in _trace["name"]:
                return False
            return True

        ### collect traces of FW + BP OPs and STEP OPs
        index = 0
        while index < len(traces):
            trace = traces[index]
            index += 1
            name = standar_name(trace["name"])       

            if name not in self.dag.nodes:
                ### Only collect nodes in the dag
                ### TODO (huhanpeng): some trvial nodes may also be useful
                continue

            ### deduplication
            ### TODO (huhanpeng): should be careful, only choose one prosess here
            if one_pid is None:
                one_pid = trace["pid"]
            elif one_pid != trace["pid"]:
                continue

            ### Initialize the start time of the entire running span
            self.run_span[wk_prefix].init_start(trace["ts"])

            innodes = [_n for _n, _ in self.dag.in_edges(name)]
            _args = {"name": name}
            for i, _n in enumerate(innodes):
                _args["input%d"%i] = _n
            trace["name"] = name
            trace["args"] = _args
            if pid is not None:
                trace["pid"] = pid
            rst_traces["traceEvents"].append(trace)

            ### if all STEP-dependent BW nodes have arrived, process traces til FW
            # if len(last_bw_nodes) == 0:
            if name == _real_last_bw_name:
                _step_ts = None
                _step_dur = 0
                while index < len(traces):
                    _trace = traces[index]
                    if one_pid != _trace["pid"]:
                        index += 1
                    else:
                        name = standar_name(_trace["name"])
                        if name in self.dag.nodes:
                            break
                        index += 1
                        if is_update_op(_trace):
                            if _step_ts is None:
                                # print(_trace["name"], _trace["ts"])
                                _step_ts = _trace["ts"]
                            _step_dur = _trace["ts"] + _trace["dur"] - _step_ts
                if _step_ts is not None:
                    if self.clock_aligner is not None and host_id is not None:
                        ### register the start time of first STEP
                        self.clock_aligner.mark_ref_time(host_id, _step_ts)
                    rst_traces["traceEvents"].append({
                        "name": "STEP",
                        "ts": _step_ts,
                        "dur": _step_dur,
                        "ph": "X",
                        "cat": "operator",
                        "pid": pid if pid is not None else one_pid,
                        "args": {
                            "name":"STEP"
                        }
                    })
                    ### Initialize the end time of the entire running span
                    self.run_span[wk_prefix].init_end(_step_ts + _step_dur)

        if host_id is not None:
            self.clock_aligner.append_traces(host_id, rst_traces["traceEvents"])
        else:
            self.time_dict["traceEvents"] += rst_traces["traceEvents"]

    def bpf_collect_io(self, tmp_pm=None, pid=None, host_id=None):
        io_path = self.pm.search(FileName.IO) if tmp_pm is None else tmp_pm.search(FileName.IO)
        if io_path is None:
            return

        wk_prefix, _ = PathManager("/".join(io_path.split('/')[:-1])).ret_prefix()

        rst_traces = []
        with open(io_path, 'r') as f:
            io_traces = json.load(f)

        if isinstance(io_traces, dict):
            io_traces = io_traces["traceEvents"]

        io_traces = sorted(io_traces, key=lambda x: x["ts"], reverse=False)
        for trace in io_traces:
            if "ts" in trace and not self.run_span[wk_prefix].if_start(trace["ts"]):
                continue
            elif "ts" in trace and self.run_span[wk_prefix].if_end(trace["ts"]):
                break
            else:
                if pid is not None:
                    trace["pid"] = pid
                rst_traces.append(trace)

        if host_id is not None:
            self.clock_aligner.append_traces(host_id, rst_traces)
        else:
            self.time_dict["traceEvents"] += rst_traces

    def bpf_collect_comm_detail(self, tmp_pm, pid=None, host_id=None):
        comm_d_path = self.pm.search(FileName.COMM_DETAIL) if tmp_pm is None else tmp_pm.search(FileName.COMM_DETAIL)

        if comm_d_path is None:
            return

        wk_prefix, _ = PathManager("/".join(comm_d_path.split('/')[:-1])).ret_prefix()

        rst_traces = []
        try:
            with open(comm_d_path, 'r') as f:
                traces = json.load(f)
        except json.decoder.JSONDecodeError:
            traceback.print_exc()
            traces = []
            self.logger.warn("Ignore above exception: %s" % p)

        if isinstance(traces, dict):
            traces = traces["traceEvents"]

        traces = sorted(traces, key=lambda x: x["ts"], reverse=False)
        for trace in traces:
            # if host_id is not None and "broadcast" in trace["name"] and trace["ph"] != "i":
            #     self.clock_aligner.mark_ref_time(host_id, trace["ts"])

            if "ts" in trace and not self.run_span[wk_prefix].if_start(trace["ts"]):
                continue
            elif "ts" in trace and self.run_span[wk_prefix].if_end(trace["ts"]):
                break
            else:
                if trace["ph"] == "i" or trace["ph"] == "I":
                    trace["s"] = "p"
            rst_traces.append(trace)

        assert len(rst_traces) > 0

        if host_id is not None:
            self.clock_aligner.append_traces(host_id, rst_traces)
        else:
            self.time_dict["traceEvents"] += rst_traces

    def bpf_collect_comm(self):
        comm_path = self.pm.search(FileName.COMM)
        if comm_path is None:   
            return
        if self.dag is None:
            dag_path = self.pm.search(FileName.DAG)
            self.dag = nx.read_gml(dag_path)
        comm_traces = self.parse_comm_traces(comm_path)
        self.time_dict["traceEvents"] += comm_traces
    
    def parse_comm_traces(self, path):
        self.gradient_name_list = {}

        ### Get the RunningSpan of root host
        run_span_key = sorted(self.run_span.keys())[0]

        #! read communication traces offline
        with open(path, 'r') as f:
            json_str = f.read()

        ### TODO (huhanpeng) delete
        # fix the json file
        if json_str[-1] != ']':
            json_str_lines = json_str.split("\n")
            if json_str_lines[-1] == '':
                json_str_lines = json_str_lines[:-1]
            if json_str_lines[-1][-1] == ',':
                json_str_lines[-1] = json_str_lines[-1][:-1]+']'
            json_str = "\n".join(json_str_lines)
        comm_traces = json.loads(json_str)

        ret = []
        for trace in comm_traces:
            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in self.gradient_name_list
                    _split_name = trace["args"]["name"].split(".")
                    # ignore the traces whose names end with purly digits
                    if str.isdigit(_split_name[-1]):
                        continue
                    raw_name = ".".join(_split_name[1:])
                    prefix = _split_name[0]
                    if "horovod_" not in prefix:
                        raise ValueError("comm.json format error, "
                            "trace args name should start with "
                            "horovod_broadcast or horovod_allreduce: %s" % trace["args"]["name"])
                    process_name = "Comm." + raw_name
                    self.gradient_name_list[trace["pid"]] = {
                            "process_name": process_name,
                            "tid": prefix,
                            "list": []
                            }
                else:
                    pass
            elif "ts" in trace and not self.run_span[run_span_key].if_start(trace["ts"]):
                continue
            elif "ts" in trace and self.run_span[run_span_key].if_end(trace["ts"]):
                break
            elif trace["pid"] in self.gradient_name_list and trace["ph"] == "B":
                cur_pid = self.gradient_name_list[trace["pid"]]
                cur_pid["list"].append((trace["name"], trace["ts"]))
            elif trace["pid"] in self.gradient_name_list and trace["ph"] == "E":
                cur_pid = self.gradient_name_list[trace["pid"]]
                if len(cur_pid["list"]) == 0:
                    continue
                name, ts = cur_pid["list"].pop()
                dur = trace["ts"] - ts
                process_name = cur_pid["process_name"]
                input_nodes = [u for u, _ in self.dag.in_edges(process_name)]
                if len(input_nodes) == 1:
                    input0 = list(input_nodes)[0]
                elif len(input_nodes) == 0:
                    input0 = None
                    # self.logger.warn("%s has no in edges" % process_name)
                else:
                    raise ValueError("Each communication node can not "
                        "have more than 1 in-edge nodes: %s" % process_name)
                ret.append(
                    {
                        "name": name,
                        "ts": ts,
                        "dur": dur,
                        "ph": "X",
                        "pid": process_name,
                        "tid": cur_pid["tid"],
                        "cat": "Comm",
                        "args":{
                            "name": process_name,
                            "input0": input0
                        }
                    })
            else:
                pass
        return ret

    def bpf_collect_update(self):
        raise NotImplementedError()
        with open(self.path_dict["update"], 'r') as f:
            rst_traces = json.load(f)
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]

    ### TODO (huhanpeng): delete
    def loop_collect(self, sub_option):
        if self.pm.dir_level == DirLevel.GPU:
            if sub_option == "operator":
                self.update_final_traces(_operator=True)
            else:
                self.re_gen_final_traces()

    def iter_combine(self):
        rst_traces = {"traceEvents": []}
        if self.pm.dir_level == DirLevel.GPU:
            self.time_dict = {"traceEvents":[]}
            self.bpf_collect_comp()
            self.bpf_collect_io()
            self.bpf_collect_comm()
            rst_traces["traceEvents"] += self.time_dict["traceEvents"]
        elif self.pm.dir_level == DirLevel.WORKER:
            ### collect computation traces and IO traces
            for _dir in self.pm.dirs:
                self.time_dict = {"traceEvents":[]} 
                gpu_path = os.path.join(self.pm.path, _dir)
                ### All GPUs on all host machines share the communication traces
                self.bpf_collect_non_comm(_path=gpu_path, pid="rank%s"%_dir)
                rst_traces["traceEvents"] += self.time_dict["traceEvents"]
            self.time_dict = {"traceEvents":[]} 
            self.bpf_collect_comm()
            rst_traces["traceEvents"] += self.time_dict["traceEvents"]

        elif self.pm.dir_level == DirLevel.TRIAL:
            self.clock_aligner = ClockAligner()
            for _dir in self.pm.dirs:
                worker_traces = []
                worker_path = os.path.join(self.pm.path, _dir)
                worker_root, worker_dirs, _ = list(os.walk(worker_path))[0]
                worker_dirs = sorted(worker_dirs)
                for __dir in worker_dirs:
                    self.time_dict = {"traceEvents":[]} 
                    gpu_path = os.path.join(worker_root, __dir)
                    ### All GPUs on all host machines share the communication traces
                    self.bpf_collect_non_comm(_path=gpu_path, pid=str(_dir)+".rank%s"%__dir, host_id=_dir)
                    worker_traces += self.time_dict["traceEvents"]

            ### align the time
            rst_traces["traceEvents"] += self.clock_aligner.align()
            self.clock_aligner = None

            ### only read comm.json once
            self.time_dict = {"traceEvents":[]} 
            self.bpf_collect_comm()
            rst_traces["traceEvents"] += self.time_dict["traceEvents"]


        with open(os.path.join(self.pm.path, FileName.TRACE.value), 'w') as f:
                json.dump(rst_traces, f, indent=4)

        return rst_traces["traceEvents"]

    def collect_traces(self):
        trace_path = self.pm.search(FileName.TRACE)
        if trace_path is not None:
            traces = read_traces(trace_path)
            return TraceManager(traces, self.pm.dir_level)
        else:
            self.logger.info("Generating %s" % (FileName.TRACE.value))
            return TraceManager(self.iter_combine(), self.pm.dir_level)

    def iter_time(self):
        traceM = self.collect_traces()
        get_iter_time(traceM.traces)

    def collect_dag(self, args):
        assert self.pm.dir_level == DirLevel.TRIAL
        critical_path = []
        worker_dag_list = []   
        traceM = self.collect_traces()
        for _dir in self.pm.dirs:
            worker_path = os.path.join(self.pm.path, _dir)
            worker_root, worker_dirs, _ = list(os.walk(worker_path))[0]
            for worker_dir in worker_dirs:
                gpu_path = os.path.join(worker_root, worker_dir)
                self.logger.info("## Collect DAG in %s" % (gpu_path))
                dagmanager = DAGManager(gpu_path, traceM)
                max_para_degree, _critical_path = dagmanager.gen_gpu_dag(_pretty=args.pretty)
                worker_dag_list.append(dagmanager.gpu_dag)
                if _critical_path is not None:
                    critical_path += _critical_path

        ### Combine all worker_dag_list on one worker, build the dependency
        return nx.compose_all(worker_dag_list)
        
    def all_prefix_list(self):
        ''' Return all prefixes under the dirctory.
            * For DirLevel.WORKER, it is a list of rank<id>
            * For DIrLevel.TRIAL, it is a list of host<id>.rank<id>
        '''
        prefixL = []
        if self.pm.dir_level == DirLevel.TRIAL:
            for _dir in self.pm.dirs:
                worker_path = os.path.join(self.pm.path, _dir)
                worker_root, worker_dirs, _ = list(os.walk(worker_path))[0]
                for worker_dir in worker_dirs:
                    prefixL.append("%s.rank%s"%(_dir, worker_dir))
        elif self.pm.dir_level == DirLevel.WORKER:
            for _dir in self.pm.dirs:
                prefixL.append("%s"%(_dir))
        else:
            raise ValueError("Do not support DirLevel.GPU")
        return prefixL

        
    
