from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from trace_utils import return_path_dict, get_iter_time

class Collector(object):
    #! class used to collect trace info
    def __init__(self, _path_dict=None):
        self.logger = logger_utils.SingleLogger()
        self.time_dict = {"traceEvents":[]}
        if _path_dict is not None:
            self.path_dict = _path_dict
        else:
            self.path_dict = {}
        
        # dag is necessary.
        if "gml_path" in self.path_dict:
            self.dag = nx.read_gml(self.path_dict["gml_path"])

    def bpf_collect_io(self):
        if "io" not in self.path_dict or not os.path.exists(self.path_dict["io"]):
            self.logger.warn("'io.json' not exists.")
            return

        with open(self.path_dict["io"], 'r') as f:
            rst_traces = json.load(f)
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]

    def bpf_collect_comm(self):
        if "comm" not in self.path_dict or not os.path.exists(self.path_dict["comm"]):   
            _p = os.path.join(os.path.dirname(self.path_dict["root"]), "comm.json")
            if os.path.exists(_p):
                self.logger.warn("find 'comm.json' at %s instead" % _p)
                self.path_dict["comm"] = _p
            else:
                self.logger.warn("'comm.json' not exists.")
                return
        comm_traces = self.parse_comm_traces(self.path_dict["comm"])
        self.time_dict["traceEvents"] += comm_traces

    def parse_comm_traces(self, path):
        self.gradient_name_list = {}

        #! read communication traces offline
        with open(path, 'r') as f:
            json_str = f.read()
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

            elif trace["pid"] in self.gradient_name_list and trace["ph"] == "B":
                cur_pid = self.gradient_name_list[trace["pid"]]
                cur_pid["list"].append((trace["name"], trace["ts"]))
            elif trace["pid"] in self.gradient_name_list and trace["ph"] == "E":
                cur_pid = self.gradient_name_list[trace["pid"]]
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

    def update_final_traces(self, _io=False, _comm=False, _operator=False):
        self.logger.info("Updating " + self.path_dict["trace_path"])
        assert os.path.exists(self.path_dict["trace_path"])
        with open(self.path_dict["trace_path"], 'r') as f:
            self.time_dict = json.load(f)

        if _io is True:
            self.delete_traces("I/O")
            self.bpf_collect_io()

        if _comm is True:
            self.delete_traces("Comm")
            self.bpf_collect_comm()

        if _operator is True:
            self.delete_traces("operator")
            self.bpf_collect_computation()

        get_iter_time(self.time_dict, self.logger)

        with open(self.path_dict["trace_path"], 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def delete_traces(self, _cat):
        _rst_traces = {"traceEvents": []}
        _rst_traces["traceEvents"] = [_trace for _trace in self.time_dict["traceEvents"] if _trace["cat"] != _cat]
        self.time_dict = _rst_traces

    def re_gen_final_traces(self):
        self.logger.info("Recombining " + self.path_dict["trace_path"])
        self.time_dict = {"traceEvents":[]}
        ### Apply dependencies in self.dag to the mxnet traces.
        self.bpf_collect_computation()
        ### Collect communication traces, IO traces and STEP traces and apply dependency
        self.bpf_collect_io()
        self.bpf_collect_comm()
        get_iter_time(self.time_dict, self.logger)
        with open(self.path_dict["trace_path"], 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def re_gen_comp_io_traces(self):
        self.time_dict = {"traceEvents":[]}
        ### Apply dependencies in self.dag to the mxnet traces.
        self.bpf_collect_computation()
        ### Collect communication traces, IO traces and STEP traces and apply dependency
        self.bpf_collect_io()
        return self.time_dict

    def bpf_collect_computation(self):
        '''Apply dependency info to the mxnet trace results

        Parameters
        ----------
        mxnet_traces : dict
            A dict containing MXNet trace results.

        Returns
        ----------
        rst_traces : dict
            A dict containing MXNet trace results combined with dependency info.
        '''
        
        if "temp" not in self.path_dict or not os.path.exists(self.path_dict["temp"]):
            self.logger.warn("'temp.json' not exists.")
            return

        ''' Output trace resutls '''
        with open(self.path_dict["temp"], 'r') as f:
            mxnet_traces = json.load(f)
        
        pid = None
        rst_traces = {"traceEvents": []}

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

        def _preprocess(_name):
            '''Fetch and handle the trace name'''
            #! add for mxnet-gluon case
            if "name=" in _name:
                _name = _name.split("name=")[1].split(";")[0]
            #! backward nodes or forward nodes
            _name = "BW." + _name.split("_backward")[0] if "_backward" in _name else "FW." + _name
            _name = _name.split("_fwd")[0] if "_fwd" in _name else _name
            return _name 

        IGNORE_OP = ["DeleteVariable", "sum", "_plus_scalar", 
                "_copyto_GPU2GPU", "broadcast_add", 
                "Reshape", "Cast", "_arange", "elemwise_add",
                "_ones", "SyncCopyGPU2CPU", "_mul_scalar",
                "CopyGPU2CPU", "CopyCPU2GPU"]

        def real_last_bw_name():
            statue = "init"
            _index = 0
            tmp = None
            while _index < len(traces):
                trace = traces[_index]
                _index += 1
                name = _preprocess(trace["name"])
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

        index = 0
        iteration_time = {"ts": None, "fw_bw": [], "iteration": []}
        while index < len(traces):
            trace = traces[index]
            index += 1
            name = _preprocess(trace["name"])       

            if name not in self.dag.nodes:
                #! Only collect nodes in the dag
                #! TODO: some trvial nodes may also be useful
                continue

            #! deduplication
            #! TODO: should be careful, only choose one prosess here
            if pid is None:
                pid = trace["pid"]
            elif pid != trace["pid"]:
                continue

            innodes = [_n for _n, _ in self.dag.in_edges(name)]
            args = {"name": name}
            for i, _n in enumerate(innodes):
                args["input%d"%i] = _n
            trace["name"] = name
            trace["args"] = args
            if iteration_time["ts"] is None:
                iteration_time["ts"] = trace["ts"]
            rst_traces["traceEvents"].append(trace)

            #! if all STEP-dependent BW nodes have arrived, process traces til FW
            # if len(last_bw_nodes) == 0:
            if name == _real_last_bw_name:
                iteration_time["fw_bw"].append((trace["ts"] + trace["dur"]- iteration_time["ts"]) / 1000.0)
                _step_ts = None
                _step_dur = 0
                while index < len(traces):
                    _trace = traces[index]
                    if pid != _trace["pid"]:
                        index += 1
                    else:
                        name = _preprocess(_trace["name"])
                        if name in self.dag.nodes:
                            break
                        index += 1
                        if _trace["name"] in IGNORE_OP or "operator" != _trace["cat"]:
                            pass
                        else:
                            if _step_ts is None:
                                # print(_trace["name"], _trace["ts"])
                                _step_ts = _trace["ts"]
                            _step_dur = _trace["ts"] + _trace["dur"] - _step_ts
                if _step_ts is not None:
                    rst_traces["traceEvents"].append({
                        "name": "STEP",
                        "ts": _step_ts,
                        "dur": _step_dur,
                        "ph": "X",
                        "cat": "operator",
                        "pid": pid,
                        "args": {
                            "name":"STEP"
                        }
                    })

                iteration_time["iteration"].append((_step_ts + _step_dur - iteration_time["ts"]) / 1000.0)
                iteration_time["ts"] = None

        self.time_dict["traceEvents"] += rst_traces["traceEvents"]



