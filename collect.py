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

from trace_utils import return_path_dict

class Collector(object):
    #! class used to collect trace info
    def __init__(self, _logger, _trace_dir=None, _path_dict=None):
        self.logger = _logger
        self.time_dict = {"traceEvents":[]}
        if _path_dict is not None:
            self.path_dict = _path_dict
        else:
            assert _trace_dir is not None
            self.trace_dir = _trace_dir
            self.path_dict = return_path_dict(self.trace_dir)
        assert "trace_path" in self.path_dict
        assert "gml_path" in self.path_dict
        assert "temp" in self.path_dict
        assert "comm" in self.path_dict
        assert "io" in self.path_dict
        assert "loss" in self.path_dict
        assert "symbol_debug_str" in self.path_dict
        self.dag = nx.read_gml(self.path_dict["gml_path"])

        self.gradient_name_list = None # !!!! need to handle this          

    def byteps_collect_io(self):
        with open(self.path_dict["io"], 'r') as f:
            rst_traces = json.load(f)
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]

    def byteps_collect_comm(self):
        #! read communication traces offline
        with open(self.path_dict["comm"], 'r') as f:
            rst_traces = json.load(f)
        for trace in rst_traces["traceEvents"]:
            if "byteps.gradient_" not in trace["args"]["name"]:
                continue
            para_index = int(trace["args"]["name"].split("_")[-1])
            para_name = self.gradient_name_list[para_index]
            if trace["name"] != trace["args"]["name"]:
                #! subtask
                trace["name"] = "Comm." + para_name + "." + trace["name"].split(".")[-1]
            else:
                #! main task
                trace["name"] = "Comm." + para_name
            trace["pid"] = "Comm." + para_name
            trace["args"]["name"] = "Comm." + para_name
            input_nodes = [u for u, _ in self.dag.in_edges("Comm." + para_name)]
            assert len(input_nodes) == 1
            trace["args"]["input0"] = list(input_nodes)[0]
            self.time_dict["traceEvents"].append(trace)

    def byteps_collect_update(self):
        raise NotImplementedError()
        with open(self.path_dict["update"], 'r') as f:
            rst_traces = json.load(f)
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]

    def update_final_traces(self, _io=False, _comm=False, _operator=False):
        self.logger.info("Updating " + self.path_dict["trace_path"])
        with open(self.path_dict["trace_path"], 'r') as f:
            self.time_dict = json.load(f)

        if _io is True:
            self.delete_traces("I/O")
            self.byteps_collect_io()

        if _comm is True:
            raise NotImplementedError("gradient_name_list has not be recorded.")
            self.delete_traces("Comm")
            self.byteps_collect_comm()

        if _operator is True:
            self.delete_traces("operator")
            self.byteps_collect_computation()

        with open(self.path_dict["trace_path"], 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def delete_traces(self, _cat):
        _rst_traces = {"traceEvents": []}
        _rst_traces["traceEvents"] = [_trace for _trace in self.time_dict["traceEvents"] if _trace["cat"] != _cat]
        self.time_dict = _rst_traces

    def save_trace(self):
        #! Apply dependencies in self.dag to the mxnet traces.
        self.byteps_collect_computation()
        #! Collect communication traces, IO traces and STEP traces and apply dependency
        self.byteps_collect_io()
        self.byteps_collect_comm()
        self.byteps_collect_update() 
        with open(self.path_dict["trace_path"], 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def byteps_collect_computation(self):
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
                "_ones", "SyncCopyGPU2CPU", "_mul_scalar"]

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

                iteration_time["iteration"].append((_trace["ts"] + _trace["dur"] - iteration_time["ts"]) / 1000.0)
                iteration_time["ts"] = None

        self.logger.info("fw + bw: %f ms -- iteration time: %f ms" % (
                sum(iteration_time["fw_bw"]) / float(len(iteration_time["fw_bw"])), 
                sum(iteration_time["iteration"]) / float(len(iteration_time["iteration"]))
                ))
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]



