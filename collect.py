import warnings
import os

# append for auto_profiling
import sys, os
import ujson as json
import networkx as nx
import threading
import time
import logger_utils
import arg_utils
import debug_utils

from trace_utils import *
from dag_utils import * 
from horovod.graph import *
from parameter import *
from bps_helper.graph import *

args_ = arg_utils.SingleArg().args

GAP_THRESHOLD = 20

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
    def __init__(self, byteps_graph = None):
        self.traces_per_host = {}
        self.marker_per_host = {}
        self.standard = None
        self.ref = None
        self.byteps_graph = byteps_graph

    def append_traces(self, host_id, traces):
        if host_id not in self.traces_per_host:
            self.traces_per_host[host_id] = []
            self.marker_per_host[host_id] = (None, None)
        self.traces_per_host[host_id] += traces

    def mark_ref_time(self, host_id, _t, ref_name, _override=False):
        if host_id not in self.traces_per_host:
            self.traces_per_host[host_id] = []
            self.marker_per_host[host_id] = (None, None)

        if self.marker_per_host[host_id][0] is None or _override:
            self.marker_per_host[host_id] = (_t, ref_name)

    def align(self):
        rst_traces = []
        host_ids = list(self.traces_per_host.keys())

        def host_id_to_rank(host_id):
            return int(host_id.split("_")[-1])

        standard_time, ref_name = self.marker_per_host[host_ids[0]]
        if self.byteps_graph is not None:
            host_ranks = [host_id_to_rank(fn) for fn in host_ids]
            base_host_name = host_ids[host_ranks.index(self.byteps_graph.master_host_id)]
        else:
            base_host_name = host_ids[0]
        if standard_time is None and self.byteps_graph is None:
            SingleLogger().warn("Have not set the align standard, fail to do clock synchronization")
        for host_id, traces in self.traces_per_host.items():
            if host_id == base_host_name or (standard_time is None and self.byteps_graph is None):
                pass
            else:
                # if n != ref_name:
                #     print("ref_name: %s, name: %s" % (ref_name, n))
                #     raise
                if self.byteps_graph is None:
                    t, n = self.marker_per_host[host_id]
                    bias = standard_time - t
                else:
                    bias = self.byteps_graph.time_drift[host_id_to_rank(host_id)]
                SingleLogger().info("Align - add {} us to {}".format(bias, host_id))
                for trace in traces:
                    trace["ts"] += bias
            rst_traces += traces
        return rst_traces

class Collector(object):
    #! class used to go through the file system and collect info
    def __init__(self, root_path, comm_backend = "NCCL", platform = "TENSORFLOW"):
        self.logger = logger_utils.SingleLogger()
        self.pm = PathManager(root_path)
        self.traceM = None
        self.comm_backend = comm_backend.upper()
        self.platform = platform.upper()
        if self.comm_backend not in ["NCCL", "BYTEPS"]:
            raise RuntimeError("Unsupported communication backend {}. Must use NCCL or BytePS.".format(self.comm_backend))
        if self.platform not in ["MXNET", "TENSORFLOW"]:
            raise RuntimeError("Unsupported platform {}. Must be one of MXNET or TENSORFLOW.".format(self.platform))
        self.nccl_graph = None
        self.byteps_graph = None
        if self.comm_backend == "NCCL":
            self.nccl_graph = ncclGraph()
        else:
            # BYTEPS
            self.byteps_graph = bytepsGraph()
        self.trail_dag = None

        self.time_dict = None
        self.run_span = {}
        ### Used for clock synchronization when combime traces from multiple machines
        self.clock_aligner = None
        self.para_dict = None

        ### TODO (huhanpeng): assume different host use the same dag, original dag
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

        TraceManager(self.time_dict, self.pm.dir_level).get_iter_time()

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
        ### Collect communication traces, IO traces and UPDATE traces and apply dependency
        self.bpf_collect_io()
        self.bpf_collect_comm()
        TraceManager(self.time_dict, self.pm.dir_level).get_iter_time()
        with open(trace_path, 'w') as f:
            json.dump(self.time_dict, f, indent=4)

    def bpf_collect_for_rank(self, _path, pid, host_id=None):
        tmp_pm = PathManager(_path)
        assert tmp_pm.dir_level == DirLevel.GPU
        self.bpf_collect_comp(tmp_pm, pid, host_id)
        self.bpf_collect_io(tmp_pm, pid, host_id)
        self.bpf_collect_comm_detail(tmp_pm, pid, host_id)
        self.bpf_collect_comm(tmp_pm, pid, host_id)

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
        debug_utils.DebugRecorder().debug_event_start()
        comp_path = self.pm.search(FileName.COMP) if tmp_pm is None else tmp_pm.search(FileName.COMP)
        if self.dag is None:
            dag_path = self.pm.search(FileName.DAG) if tmp_pm is None else tmp_pm.search(FileName.DAG)
            debug_utils.DebugRecorder().debug_event_start()
            self.dag = nx.read_gml(dag_path)
            debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.read_dag", "Collct", "0")

        if comp_path is None:
            return

        wk_prefix, _ = PathManager("/".join(comp_path.split('/')[:-1])).ret_prefix()
        if wk_prefix not in self.run_span:
            self.run_span[wk_prefix] = RunningSpan()

        debug_utils.DebugRecorder().debug_event_start()
        ''' Output trace resutls '''
        with open(comp_path, 'r') as f:
            raw_traces = json.load(f)
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.jsonload", "Collct", "0")

        one_pid = None
        kernel_pid = None
        rst_traces = {"traceEvents": []}
        kernel_times = {}

        ### convert the ph from B/E to X
        ### TODO(huhanpeng): it's more safe to use a stack style method
        index = 0
        traces = []
        while index < len(raw_traces["traceEvents"]):
            if self.platform == "TENSORFLOW" and one_pid is None:
                for trace in raw_traces["traceEvents"]:
                    if trace["ph"] == "M" and trace["name"] == "process_name":
                        if "args" in trace and "name" in trace["args"]:
                            if "device:GPU" in trace["args"]["name"] and "Compute" in trace["args"]["name"] and "replica" in trace["args"]["name"]:
                                one_pid = trace["pid"]
                            elif "device:GPU" in trace["args"]["name"] and "stream:all Compute" in trace["args"]["name"]:
                                kernel_pid = trace["pid"]

            if "ts" not in raw_traces["traceEvents"][index]:
                index += 1
                continue
            trace = raw_traces["traceEvents"][index]
            if trace["cat"] == "Op":
                trace["cat"] = "operator"
            if trace["ph"] == 'B' or trace["ph"] == 'b':
                next_trace = raw_traces["traceEvents"][index+1]
                assert trace["name"] == next_trace["name"]
                trace["dur"] = next_trace['ts'] - trace['ts']
                trace["ph"] = "X"
                traces.append(trace)
                index += 2
            elif trace["ph"] == "X":
                traces.append(trace)
                index += 1
            else:
                index += 1

        debug_utils.DebugRecorder().debug_event_start()
        traces = sorted(traces, key=lambda x: x["ts"], reverse=False)
        SingleLogger().info("Comp traces length: {}.".format(len(traces)))
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.sorted", "Collct", "0")

        ### find the last BP op in the timeline
        if self.platform == "MXNET":
            def real_last_bw_name():
                statue = "init"
                _index = 0
                last_bw = None
                last_fw = None
                first_bw = None
                while _index < len(traces):
                    trace = traces[_index]
                    _index += 1
                    name = standard_name(trace["name"], platform=self.platform)
                    if name not in self.dag.nodes:
                        continue
                    if statue == "init" and "FW" in name:
                        statue = "fw"
                        last_fw = name
                    elif statue == "fw" and "FW" in name:
                        last_fw = name
                    elif statue == "fw" and "BW" in name:
                        statue = "bw"
                        first_bw = name
                        last_bw = name
                    elif statue == "bw" and "BW" in name:
                        last_bw = name
                    elif statue == "bw" and "FW" in name:
                        statue = "fw"
                        return last_fw, first_bw, last_bw
            debug_utils.DebugRecorder().debug_event_start()          
            last_fw, first_bw, last_bw = real_last_bw_name()
            debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.real_last_bw_name", "Collct", "0")

            def is_update_op(_trace):
                ### TODO (huhanpeng) !!! change this when model is changed
                if "update" in _trace["name"].lower():
                    return True
                else:
                    return False

            def is_cal_op(_trace):
                if "dot" == _trace["name"] or "add_n" == _trace["name"]:
                    return True
                else:
                    return False

        ### collect traces of FW + BP OPs and UPDATE OPs
        tid_hub = []
        def _ret_operator_tid(tid_):
            if tid_ in tid_hub:
                return "operator%d"%(tid_hub.index(tid_))
            else:
                tid_hub.append(tid_)
                return "operator%d"%(len(tid_hub) - 1)

        cnt = 0
        index = 0
        dag_node_names_std = set([standard_name(n, platform=self.platform) for n in self.dag.nodes])
        while index < len(traces):
            trace = traces[index]
            index += 1
            if self.platform == "MXNET":
                raw_name = trace["name"]
            elif self.platform == "TENSORFLOW":
                raw_name = trace["args"]["name"]
            name = standard_name(raw_name, platform=self.platform)       

            ### deduplication
            ### TODO (huhanpeng): should be careful, only choose one prosess here
            if one_pid is None:
                one_pid = trace["pid"]
            elif kernel_pid and kernel_pid == trace["pid"]:
                if name not in kernel_times:
                    kernel_times[name] = []
                kernel_times[name].append((trace['ts'], trace['dur']))
                continue
            elif one_pid != trace["pid"]:
                continue
            
            if name not in dag_node_names_std:
                ### Only collect nodes in the dag
                ### TODO (huhanpeng): some trvial nodes may also be useful
                if args_.trace_level == "debug":
                    trace["name"] = "%s.%d"%(trace["name"], index)
                    trace["tid"] = trace["cat"] = "debug"
                    if pid is not None:
                        trace["pid"] = pid
                    rst_traces["traceEvents"].append(trace)
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
            trace["tid"] = _ret_operator_tid(trace["tid"])
            rst_traces["traceEvents"].append(trace)

            ### Handle OUTPUT
            if self.platform == "MXNET":
                if name == last_fw:
                    output_ts = None
                    output_dur = None
                    output_tid = None
                    while index < len(traces):
                        _trace = traces[index]
                        if one_pid != _trace["pid"]:
                            index += 1
                        else:
                            name = standard_name(_trace["name"], platform=self.platform)
                            if name == first_bw or name in self.dag.nodes:
                                break
                            output_ts = _trace["ts"] if output_ts is None else output_ts
                            output_dur = _trace["ts"] + _trace["dur"] - output_ts
                            output_tid = _trace["tid"] if output_tid is None else output_tid
                            index += 1
                    if output_ts is not None and output_dur is not None:
                        rst_traces["traceEvents"].append({
                            "name": "OUTPUT0",
                            "ts": output_ts,
                            "dur": output_dur,
                            "ph": "X",
                            "cat": "operator",
                            "pid": pid if pid is not None else one_pid,
                            "tid": _ret_operator_tid(output_tid),
                            "args": {
                                "name":"OUTPUT0"
                            }
                        })

                ### if all UPDATE-dependent BW nodes have arrived, process traces til FW
                # if len(last_bw_nodes) == 0:
                elif name == last_bw:
                    _update_ts = None
                    _cal_ts = None
                    _cal_tid = None
                    _duration = 0
                    _cnt = 0
                    while index < len(traces):
                        _trace = traces[index]
                        if one_pid != _trace["pid"]:
                            index += 1
                        else:
                            name = standard_name(_trace["name"], platform=self.platform)
                            if name in self.dag.nodes:
                                break
                            index += 1
                            if is_cal_op(_trace):
                                if _cal_ts is None:
                                    _cal_ts = _trace["ts"]
                                    _cal_tid = _trace["tid"]
                                _duration = _trace["ts"] + _trace["dur"] - _cal_ts
                            if is_update_op(_trace):
                                if _update_ts is None:
                                    _update_ts = _trace["ts"]
                                    ### Add UPDATE_CAL node
                                    rst_traces["traceEvents"].append({
                                        "name": "UPDATE_CAL",
                                        "ts": _cal_ts,
                                        "dur": _duration,
                                        "ph": "X",
                                        "cat": "operator",
                                        "pid": pid if pid is not None else one_pid,
                                        "tid": _ret_operator_tid(_cal_tid),
                                        "args": {
                                            "name":"UPDATE_CAL"
                                        }
                                    })
                                _duration = _trace["ts"] + _trace["dur"] - _update_ts
                                rst_traces["traceEvents"].append({
                                    "name": "UPDATE_%d"%_cnt,
                                    "ts": _trace["ts"],
                                    "dur": _trace["dur"],
                                    "ph": "X",
                                    "cat": "operator",
                                    "pid": pid if pid is not None else one_pid,
                                    "tid": _ret_operator_tid(_trace["tid"]),
                                    "args": {
                                        "name":"UPDATE_%d"%_cnt
                                    }
                                })
                                _cnt += 1
                    if _update_ts is not None:
                        ### Initialize the end time of the entire running span
                        self.run_span[wk_prefix].init_end(_update_ts + _duration)
            elif self.platform == "TENSORFLOW":
                _trace = traces[index]
                if "args" in _trace and "input0"in _trace["args"] and _trace["args"]["input0"] == "global_step":
                    self.run_span[wk_prefix].init_end(_trace["ts"])
            else:
                raise NotImplementedError("Unsupported platform {}.".format(self.platform))
        
        occurence_counter = {}
        for trace in rst_traces["traceEvents"]:
            if "ph" in trace and trace["ph"] == "X":
                if trace["name"] in kernel_times:
                    if trace["name"] not in occurence_counter:
                        occurence_counter[trace["name"]] = 0
                    try:
                        kernel_ts, kernel_dur = kernel_times[trace["name"]][occurence_counter[trace["name"]]]
                    except:
                        # SingleLogger.warn("Length mismatch between kernel and op launch traces for op {}".format(trace["name"]))
                        occurence_counter[trace["name"]] += 1
                        continue
                    trace["ts"] = kernel_ts
                    trace["dur"] = kernel_dur
                    occurence_counter[trace["name"]] += 1
        self.clock_aligner.append_traces(host_id, rst_traces["traceEvents"])
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp", "Collct", "0")

    def bpf_collect_io(self, tmp_pm=None, pid=None, host_id=None):
        debug_utils.DebugRecorder().debug_event_start()
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
                if "tid" not in trace:
                    trace["tid"] = "I/O"
                rst_traces.append(trace)

        self.clock_aligner.append_traces(host_id, rst_traces)
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_io", "Collct", "0")

    def bpf_collect_comm_detail(self, tmp_pm, pid=None, host_id=None):
        if self.comm_backend != "NCCL":
            return

        debug_utils.DebugRecorder().debug_event_start()
        comm_d_path = self.pm.search(FileName.COMM_DETAIL) if tmp_pm is None else tmp_pm.search(FileName.COMM_DETAIL)

        if comm_d_path is None:
            return

        wk_prefix, _ = PathManager("/".join(comm_d_path.split('/')[:-1])).ret_prefix()

        rst_traces = []
        debug_utils.DebugRecorder().debug_event_start()
        try:
            with open(comm_d_path, 'r') as f:
                traces = json.load(f)
        except json.decoder.JSONDecodeError:
            traceback.print_exc()
            traces = []
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comm_detail.jsonload", "Collct", "0")

        algo = args_.nccl_algo
        if algo is None:
            raise ValueError("--nccl_algo must be given")
        elif algo.lower() == "tree":
            self.nccl_graph.parse_tree_topo(traces["Tree"], map_to=pid)
        elif algo.lower() == "ring":
            self.nccl_graph.parse_ring_topo(traces["RealRing"], map_to=pid)  
            # self.nccl_graph.parse_connect_topo(traces["Ring"], map_to=pid)  
        if isinstance(traces, dict):    
            traces = traces["traceEvents"]

        traces = sorted(traces, key=lambda x: x["ts"], reverse=False)
        first_op = None
        for trace in traces:
            ### ignore digit
            if trace["name"].split(".")[-2].isdigit():
                continue

            if first_op is None:
                first_op = trace["args"]["name"]

            ### Only check the time range for the first communication operator, 
            ### then the following communication traces should be added to the final traces
            if "ts" in trace and not self.run_span[wk_prefix].if_start(trace["ts"]):
                continue
            elif first_op == trace["args"]["name"] and "ts" in trace and self.run_span[wk_prefix].if_end(trace["ts"]):
                break

            if trace["ph"].lower() == "i":
                if args_.trace_level != "debug":
                    continue
                trace["s"] = "p"

            trace["name"] = "Comm." + trace["name"].split("horovod_allreduce.")[1]
            trace["args"]["name"] = gen_long_name(None, trace["name"], suffix=("%d_%d_%d_%d"%(
                                    int(trace["args"]["loopId"]),
                                    int(trace["args"]["channelId"]),
                                    int(trace["args"]["chunkId"]), 
                                    int(trace["args"]["sliceId"]))))
            if pid is not None:
                trace["tid"] = trace["pid"]
                trace["pid"] = pid
            rst_traces.append(trace)

        assert len(rst_traces) > 0

        self.nccl_graph.parse_traces(rst_traces)
        self.clock_aligner.append_traces(host_id, rst_traces)
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comm_detail", "Collct", "0")

    def bpf_collect_comm(self, tmp_pm=None, pid=None, host_id=None):
        debug_utils.DebugRecorder().debug_event_start()
        comm_path = self.pm.search(FileName.COMM) if tmp_pm is None else tmp_pm.search(FileName.COMM)
        if comm_path is None:   
            return
        if self.dag is None:
            dag_path = self.pm.search(FileName.DAG)
            self.dag = nx.read_gml(dag_path)
        comm_traces = self.parse_comm_traces(comm_path, pid=pid, host_id=host_id)
        if host_id is None:
            self.time_dict["traceEvents"] += comm_traces
        else:
            self.clock_aligner.append_traces(host_id, comm_traces)
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comm", "Collct", "0")
    
    def parse_comm_traces(self, path, pid=None, host_id=None):
        self.gradient_name_table = {}

        ### **NOTE** that this requires the computation traces have been collected
        wk_prefix, _ = PathManager("/".join(path.split('/')[:-1])).ret_prefix()

        #! read communication traces offline
        debug_utils.DebugRecorder().debug_event_start()
        with open(path, 'r') as f:
            json_str = f.read()
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comm.read_traces", "Collct", "0")

        ### TODO (huhanpeng) delete
        ''' Fix the json file
            For Horovod, the timeline_ outputs traces as soon as a new trace is appended to the queue
            Making the trace file ends abnormally.
        '''
        debug_utils.DebugRecorder().debug_event_start()
        if json_str[-1] != ']':
            json_str_lines = json_str.split("\n")
            if json_str_lines[-1] == '':
                json_str_lines = json_str_lines[:-1]
            if json_str_lines[-1][-1] == ',':
                json_str_lines[-1] = json_str_lines[-1][:-1]+']'
            json_str = "\n".join(json_str_lines)
        comm_traces = json.loads(json_str)
        if isinstance(comm_traces, dict):
            comm_traces = comm_traces["traceEvents"]
        debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comm.load_string", "Collct", "0")

        ret = []
        for trace in comm_traces:
            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in self.gradient_name_table
                    if trace["args"]["name"] == "":
                        continue
                    _split_name = trace["args"]["name"].split(".")
                    ### Ignore the traces whose names end with purly digits
                    if str.isdigit(_split_name[-1]):
                        continue

                    if "Sync" in trace["args"]["name"]:
                        raw_name = "Sync"
                        prefix = "Sync"
                    else:
                        raw_name = ".".join(_split_name[1:])
                        prefix = _split_name[0]
                        ### For Horovod
                        if "horovod_" not in prefix:
                            raise ValueError("comm.json format error, "
                                "trace args name should start with "
                                "horovod_broadcast or horovod_allreduce: %s" % trace["args"]["name"])

                    process_name = "Comm." + raw_name
                    self.gradient_name_table[trace["pid"]] = {
                            "process_name": process_name,
                            "tid": prefix,
                            "list": []
                            }
                else:
                    pass
            elif "ts" in trace and not self.run_span[wk_prefix].if_start(trace["ts"]):
                continue
            elif "ts" in trace and self.run_span[wk_prefix].if_end(trace["ts"]):
                break
            elif trace["pid"] in self.gradient_name_table and trace["ph"] == "B":
                cur_pid = self.gradient_name_table[trace["pid"]]
                cur_pid["list"].append((trace["name"], trace["ts"]))
            elif trace["pid"] in self.gradient_name_table and trace["ph"] == "E":
                cur_pid = self.gradient_name_table[trace["pid"]]
                if len(cur_pid["list"]) == 0:
                    continue
                op_name, ts = cur_pid["list"].pop()
                dur = trace["ts"] - ts
                process_name = cur_pid["process_name"]

                if "Sync" in process_name and "none" not in op_name:
                    if self.clock_aligner is not None and host_id is not None:
                            ### register the start time of first UPDATE
                            self.clock_aligner.mark_ref_time(host_id, ts, "%s.%s"%(process_name, op_name))

                if "Sync" in process_name and args_.trace_level != "debug":
                    continue

                ### TODO (huhanpeng): Sync node only used for debug currently
                cat = "Comm" if "Sync" not in process_name else "debug"

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
                        "name": "%s.%s"%(process_name, op_name),
                        "ts": ts,
                        "dur": dur,
                        "ph": "X",
                        "pid": process_name if pid is None else pid,
                        "tid": cur_pid["tid"] if pid is None else process_name,
                        "cat": cat,
                        "args":{
                            "name": "%s.%s"%(process_name, op_name),
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
        ts_ = time.time()
        rst_traces = {"traceEvents": []}
        if self.pm.dir_level == DirLevel.GPU:
            raise NotImplementedError()
            self.time_dict = {"traceEvents":[]}
            self.bpf_collect_comp()
            self.bpf_collect_io()
            self.bpf_collect_comm()
            rst_traces["traceEvents"] += self.time_dict["traceEvents"]
        elif self.pm.dir_level == DirLevel.WORKER:
            raise NotImplementedError()
            ### collect computation traces and IO traces
            for _dir in self.pm.dirs:
                self.time_dict = {"traceEvents":[]} 
                gpu_path = os.path.join(self.pm.path, _dir)
                ### All GPUs on all host machines share the communication traces
                self.bpf_collect_for_rank(_path=gpu_path, pid="rank%s"%_dir)
                rst_traces["traceEvents"] += self.time_dict["traceEvents"]
            self.time_dict = {"traceEvents":[]} 
            self.bpf_collect_comm()
            rst_traces["traceEvents"] += self.time_dict["traceEvents"]

        elif self.pm.dir_level == DirLevel.TRIAL:
            self.clock_aligner = ClockAligner(byteps_graph= self.byteps_graph if self.comm_backend == "BYTEPS" else None)
            if self.comm_backend == "NCCL":
                self.nccl_graph.map_host_prefix_id(self.pm.dirs)
            for _dir in self.pm.dirs:
                worker_path = os.path.join(self.pm.path, _dir)
                worker_root, worker_dirs, _ = list(os.walk(worker_path))[0]
                worker_dirs = sorted(worker_dirs)
                for __dir in worker_dirs:
                    self.time_dict = {"traceEvents":[]} 
                    gpu_path = os.path.join(worker_root, __dir)
                    ### All GPUs on all host machines share the communication traces
                    self.bpf_collect_for_rank(_path=gpu_path, pid=str(_dir)+".rank%s"%__dir, host_id=_dir)

            ### align the time
            rst_traces["traceEvents"] += self.clock_aligner.align()
            self.clock_aligner = None

            if self.comm_backend == "NCCL" and not args_.pretty:
                self.nccl_graph.print_graph()

            # ### only read comm.json once
            # self.time_dict = {"traceEvents":[]} 
            # self.bpf_collect_comm()
            # rst_traces["traceEvents"] += self.time_dict["traceEvents"]
        if self.comm_backend == "BYTEPS":
            rst_traces["traceEvents"] += self.byteps_graph.gen_compatible_trace(dump_path=os.path.join(self.pm.path, FileName.BPS_ALIGNED_TRACE.value))

        self.logger.info("Take %f s to combine all traces" % (time.time() - ts_))
        return rst_traces["traceEvents"]

    def dump_traces(self):
        if rst_traces is None:
            rst_traces = self.traceM.traces
        rst_traces = sorted(rst_traces, key=lambda x: (x["pid"], x["tid"]))
        with open(os.path.join(self.pm.path, FileName.TRACE.value), 'w') as f:
            json.dump(rst_traces, f)

    def collect_traces(self, force_=False):
        self.logger.info("# Collecting Traces")
        self.logger.info("Generating %s" % (FileName.TRACE.value))
        self.traceM = TraceManager(self.iter_combine(), self.pm.dir_level)

    def iter_time(self):
        if self.traceM is None:
            self.collect_traces()
        self.logger.info("Original Iteration Time")
        return self.traceM.get_iter_time()

    def collect_update_dict(self):
        ### Map tensor name to its update index
        info_path = self.pm.search(FileName.INFO)
        para_path = self.pm.search(FileName.TENSOR_NAME)
        if info_path is None:
            self.logger.warn("Fail to map_tensors_to_update")
            aggregate_num = 0
        else:
            with open(info_path, 'r') as fp:
                aggregate_num = json.load(fp)["opt_aggregate_num"]
        self.para_dict = ParameterDict(para_path)
        return self.para_dict.map_tensors_to_update(aggregate_num)

    def collect_trail_dag(self):
        assert self.pm.dir_level == DirLevel.TRIAL
        self.logger.info("# Collecting DAG")
        critical_path = []
        worker_dag_list = []

        if self.platform == "MXNET":
            update_dict = self.collect_update_dict()
        else:
            update_dict = None
        for _dir in self.pm.dirs:
            worker_path = os.path.join(self.pm.path, _dir)
            worker_root, worker_dirs, _ = list(os.walk(worker_path))[0]
            for worker_dir in worker_dirs:
                gpu_path = os.path.join(worker_root, worker_dir)
                self.logger.info("## Collect DAG in %s" % (gpu_path))
                dagmanager = DAGManager(gpu_path, self.traceM, self.nccl_graph, self.byteps_graph)
                max_para_degree, _critical_path = dagmanager.gen_gpu_dag(_pretty=args_.pretty, update_dict=update_dict)
                worker_dag_list.append(dagmanager.gpu_dag)
                if _critical_path is not None:
                    critical_path += _critical_path

        ### Combine all worker_dag_list on one worker, build the dependency
        composed_dag = nx.compose_all(worker_dag_list)
        if self.comm_backend == "BYTEPS":
            composed_dag = nx.compose(composed_dag, self.byteps_graph.get_comm_graph())

        self.trail_dag = composed_dag

    def init(self, force_=False):
        trace_path = self.pm.search(FileName.TRACE)
        if self.comm_backend == "NCCL":
            nccl_graph_path = self.pm.search(FileName.NCCL_GRAPH)
        else:
            nccl_graph_path = None
            byteps_cache_path = self.pm.search(FileName.BYTEPS_CACHE)
            if byteps_cache_path is not None:
                SingleLogger().info("Inited BytePS graph helper from cache.")
                self.byteps_graph.init_from_cache(byteps_cache_path)
            else:
                SingleLogger().info("Unable to find BytePS cache file.")
                # read or generate BPS comm_trace
                byteps_comm_detail_path = self.pm.search(FileName.BPS_COMM_DETAIL)
                if byteps_comm_detail_path is None or force_:
                    # need to run preprocessing
                    if args_.pcap_file_path is None:
                        SingleLogger().error("Cannot find BytePS comm trace or pcap files.")
                        exit(1)
                    pcap_fns = [fn for fn in os.listdir(args_.pcap_file_path) if (os.path.isfile(os.path.join(args_.pcap_file_path,fn)) and fn.endswith(".pcap"))]
                    pcap_paths = [os.path.join(args_.pcap_file_path, fn) for fn in pcap_fns]
                    process_names = [fn.split(".pcap")[0] for fn in pcap_fns]
                    ip_to_rank_path = self.pm.search(FileName.IP_TO_RANK)
                    ip_to_rank_dict = {}
                    try:
                        with open(ip_to_rank_path, "r") as f:
                            for line in f:
                                ip, rank = line.strip().split(":")
                                ip = ip.strip()
                                rank = rank.strip()
                                ip_to_rank_dict[ip] = rank
                    except:
                        SingleLogger().error("Failed to read ip to rank mapping.")
                        exit(1)
                    if self.platform == "MXNET":
                        gradient_name_list_path = self.pm.search(FileName.TENSOR_NAME)
                    else:
                        gradient_name_list_path = None
                    key_dict_path = self.pm.search(FileName.KEY_DICT)
                    SingleLogger().info("Preprocessing pcap files: {}.".format(pcap_paths))
                    byteps_comm_detail_path = preprocess_pcap(pcap_paths, process_names, ip_to_rank_dict, key_dict_path, gradient_name_list_path=gradient_name_list_path, platform=self.platform)
                else:
                    SingleLogger().info("Found BytePS comm trace file in {}.".format(byteps_comm_detail_path))
                # read or generate BPS server trace
                byteps_server_trace_path = self.pm.search(FileName.BPS_SERVER_TRACE)
                if byteps_server_trace_path is None or force_:
                    # need to run preprocessing
                    if args_.server_log_path is None:
                        SingleLogger().error("Cannot find BytePS server trace or raw log files.")
                        exit(1)
                    log_fns = [fn for fn in os.listdir(args_.server_log_path) if os.path.isfile(os.path.join(args_.server_log_path,fn)) and fn.endswith(".txt")]
                    log_paths = [os.path.join(args_.server_log_path, fn) for fn in log_fns]
                    node_ranks = [int(fn.split(".txt")[0].split("_")[-1]) for fn in log_fns]
                    if self.platform == "MXNET":
                        gradient_name_list_path = self.pm.search(FileName.TENSOR_NAME)
                    else:
                        gradient_name_list_path = None
                    key_dict_path = self.pm.search(FileName.KEY_DICT)
                    SingleLogger().info("Parsing server log files: {}.".format(log_paths))
                    byteps_server_trace_path = parse_server_logs(log_paths, node_ranks, key_dict_path, gradient_name_list_path=gradient_name_list_path, platform=self.platform)
                else:
                    SingleLogger().info("Found BytePS server trace file in {}".format(byteps_server_trace_path))
                # initialize BytePS graph helper
                self.byteps_graph.init(byteps_comm_detail_path, byteps_server_trace_path)

        trail_dag_path = self.pm.search(FileName.TRAIL_DAG)
        if force_ or trace_path is None or (self.comm_backend == "NCCL" and nccl_graph_path is None) or trail_dag_path is None:
            self.collect_traces()
            self.collect_trail_dag()
            self.fine_tune_trace_dag()
            ### Cache these info
            self.traceM.dump(self.pm.path)
            if self.comm_backend == "NCCL":
                self.nccl_graph.dump(os.path.join(self.pm.path, FileName.NCCL_GRAPH.value))
            nx.write_gml(self.trail_dag, os.path.join(self.pm.path, FileName.TRAIL_DAG.value), lambda x: str(x))
        else:
            self.traceM = TraceManager()
            self.traceM.load(self.pm.path)
            if self.comm_backend == "NCCL":
                self.nccl_graph.load(nccl_graph_path)
            self.trail_dag = nx.read_gml(trail_dag_path)
        
        # for (u,v,c) in self.trail_dag.edges.data():
        #     if "exec_edges" in c:
        #         print(u, v, c)
        #         exit(0)

        ### TODO (huhanpeng) dump it or not
        if self.platform == "MXNET":
            if self.para_dict is None:
                self.collect_update_dict()

        return self.iter_time()    
        
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

    def fine_tune_trace_dag(self):
        ### Fine tune the traces and dependency graph
        if self.comm_backend == "BYTEPS":
            self.byteps_graph.calc_bw_to_comm_delay(self.traceM.traces, self.trail_dag)
        self.add_gap_to_nodes()
        ### TODO (huhanpeng): does this adapt to BytePS ???
        self.add_avg_to_nodes()
        if self.comm_backend == "NCCL":
            self.clip_recv_events()

    def search_trace_with_cnt(self, longname, send_idx):
        ### should not cansider suffix
        return self.traceM.search_by_long_name(longname, send_idx + 1)

    def re_align_traces(self, dag):
        ''' Re-align traces according to the dependency info in dag.
        TODO (huhanpeng): 
        1. currently lower priority, since clock synchronization 
            will not bring much replay accuracy improvement
        2. Cost large amount of time for a large model, e.g. Bert
        '''
        raise NotImplementedError("Do not support clock synchronization currently.")

        if self.nccl_graph.algo == NCCL_ALGO.TREE:
            SingleLogger().warn("Trace name has not be solved, can not look up")
            return
        ### bias based on each other
        align_table = dict([(host_id, {}) for host_id in self.nccl_graph.host_id2prefix.keys()])
        def display_align_table():
            for host_id, _dict in align_table.items():
                print("For host %d" % host_id)
                for _id, _range in _dict.items():
                    print("     based on %d: %s" % (_id, _range.displays()))
                    

        ### bias based on host0
        align_list = [None for host_id in self.nccl_graph.host_id2prefix.keys()]
        for u, v in dag.edges:
            if "Comm" in u and "SEND" in u:
                assert "Comm" in v and "RECV" in v
                send_host = self.nccl_graph.ret_hostid(u)
                recv_host = self.nccl_graph.ret_hostid(v)

                ### For the edge SEND->RECV in one host, just ignore
                if send_host == recv_host:
                    continue

                ### Find the corresponding traces
                ### TODO (huhanpeng): only find one pair of traces
                send_idx = recv_idx = -1
                cnt = 0
                while True:
                    send_idx, send_trace = self.search_trace_with_cnt(u, send_idx)
                    recv_idx, recv_trace = self.search_trace_with_cnt(v, recv_idx)
                    if send_idx is None:
                        break
                    ### Find send trace and recv trace
                    assert send_trace["args"]["cnt"] == recv_trace["args"]["cnt"]
                    cnt += 1
                    send_end_t = send_trace["ts"] + send_trace["dur"]
                    recv_end_t = recv_trace["ts"] + recv_trace["dur"]
                    if send_host > recv_host:
                        if recv_host not in align_table[send_host]:
                            align_table[send_host][recv_host] = BiasRange(None, None)
                        ### (hostid=send_host)'s bias based on (rankid=recv_host)
                        align_table[send_host][recv_host] *= BiasRange(None, recv_end_t - send_end_t)  
                    else:
                        ### send_host < recv_host:
                        if send_host not in align_table[recv_host]:
                            align_table[recv_host][send_host] = BiasRange(None, None)
                        ### (hostid=send_host)'s bias based on (rankid=recv_host)
                        align_table[recv_host][send_host] *= BiasRange(send_end_t - recv_end_t, None)
                    # print(send_trace)
                    # print(recv_trace)
                    # display_align_table()
                    break

        ### tidy up align table, calculate bias for all hostid based hostid=0
        def ret_bias_range_to_host0(_hostid):
            if _hostid == 0:
                return BiasRange(0, 0)
            range2host0 = BiasRange(None, None)
            for base_id, _range in align_table[_hostid].items():
                ### Assume the bias of base_id is correct
                range2host0 *= (_range + ret_bias_range_to_host0(base_id))
            return range2host0
        
        for host_id in sorted(align_table.keys()):
            bias_range = ret_bias_range_to_host0(host_id)
            SingleLogger().info("%d's bias range: %s" % (host_id, bias_range.displays()))
            align_list[host_id] = bias_range.random_gen_value()

        ### Apply these bias
        for trace in self.traceM.traces:
            ### For the original Horovod Communication traces, no need to align
            ### TODO (huhanpeng): delte these traces
            if "Comm." in trace["pid"]:
                continue

            host_id = self.nccl_graph.ret_hostid(trace["pid"])
            trace["ts"] += align_list[host_id]

    def add_gap_to_nodes(self):
        ''' Add gaps for each node '''
        self.logger.info("Add gap to dependency DAG nodes...")
        prev_events_dict = {}
        self.name2idxlist = {}
        for idx, event in enumerate(self.traceM.traces):
            if self.traceM._is_ignore_for_sta(event):
                continue

            ### Map the trace name to its indexs in all traces, 
            ### e.g. trace_name --> [idx0, idx1, None, ...], traces[idx0][name] = trace_name
            unique_name = self.traceM.ret_unique_name(event)
            if unique_name not in self.trail_dag.nodes:
                continue
            if unique_name not in self.name2idxlist:
                self.name2idxlist[unique_name] = [None] * self.traceM.max_cnt
            if event["args"]["cnt"] != -1:
                self.name2idxlist[unique_name][event["args"]["cnt"]] = idx

            ### Get previous event for this pid
            if event["pid"] not in prev_events_dict:
                prev_events_dict[event["pid"]] = {}
            cur_pid_dict = prev_events_dict[event["pid"]]

            ### Handle the gap among FW, BW and OUTPUT nodes (exclude BW->UPDATE)
            cat_ = parse_cat_from_name(event["name"])
            if self.comm_backend == "NCCL":
                ### TODO(huhanpeng): Only calculate gaps for operator nodes now
                is_need_gap = (cat_ in cur_pid_dict and cat_ == CatName.OPERATOR.value)
            else:
                ### BYTEPS
                is_need_gap = (cat_ in cur_pid_dict and (cat_ == CatName.OPERATOR.value or cat_ == CatName.COMM.value))
            if is_need_gap:
                ### There are some prev events with the same pid and find-grained cat
                prev_e = cur_pid_dict[cat_]
                gap_string = GAP_STR_OP2OP if cat_ == CatName.OPERATOR.value else GAP_STR_COMM2COMM
                if not "UPDATE_CAL" in prev_e["name"] and not "UPDATE_CAL" in event["name"] and not ("BW" in prev_e["name"] \
                    and "UPDATE_" in event["name"]) and not ("UPDATE_" in prev_e["name"] and "FW_" in event["name"]) \
                    and "local_num_masks" not in prev_e["name"]:
                    gap = event["ts"] - (prev_e["ts"] + prev_e["dur"])
                    ### TODO (huhanpeng): test whether to use this threshold
                    if gap < 0:
                        continue
                    if gap < GAP_THRESHOLD or self.comm_backend == "NCCL":
                        prev_name = self.traceM.ret_unique_name(prev_e)
                        if prev_name not in self.trail_dag.nodes:
                            continue
                        if gap_string not in self.trail_dag.nodes[prev_name]:
                            self.trail_dag.nodes[prev_name][gap_string] = gap
                            self.trail_dag.nodes[prev_name]["cnt"] = 1
                        else:
                            self.trail_dag.nodes[prev_name][gap_string] += gap
                            self.trail_dag.nodes[prev_name]["cnt"] += 1
            cur_pid_dict[cat_] = event

        queue_type_ = QueueType().ret_list()[0]
        for node_ in self.trail_dag.nodes:
            if GAP_STR_OP2OP in self.trail_dag.nodes[node_]:
                self.trail_dag.nodes[node_][GAP_STR_OP2OP] /= self.trail_dag.nodes[node_]["cnt"]
            if GAP_STR_COMM2COMM in self.trail_dag.nodes[node_]:
                self.trail_dag.nodes[node_][GAP_STR_COMM2COMM] /= self.trail_dag.nodes[node_]["cnt"]

            if self.comm_backend == "NCCL":
                ### Handle the gap between BW and the first Comm node
                if queue_type_ in node_:
                    bw_node, _ = list(self.trail_dag.in_edges(node_))[0]
                    prefix_ = parse_pid_from_name(node_)
                    comm_node = None
                    for succ_ in self.trail_dag.successors(node_):
                        if parse_pid_from_name(succ_) == prefix_:
                            comm_node = succ_
                            break
                    try:
                        u_idx_l, v_idx_l = self.name2idxlist[bw_node], self.name2idxlist[comm_node] 
                    except KeyError:
                        ### Some rank does not have SEND nodes
                        continue
                    gap = 0
                    cnt = 0
                    for cnt_ in range(self.traceM.max_cnt):
                        u_idx, v_idx = u_idx_l[cnt_], v_idx_l[cnt_]
                        if u_idx is None or v_idx is None:
                            continue
                        u_event = self.traceM.traces[u_idx]
                        v_event = self.traceM.traces[v_idx]
                        gap += v_event["ts"] - (u_event["ts"] + u_event["dur"])
                        cnt += 1
                    self.trail_dag.nodes[bw_node][GAP_STR_OP2COMM] = gap / cnt if cnt != 0 else 0
            else:
                ## Add BW -> Comm delays using byteps_graph
                ## inter node delays are directly added in replayer
                if "BW" in node_:
                    pid = parse_pid_from_name(node_)
                    node_rank = pid.split(".")[0].split("_")[-1]
                    gap = self.byteps_graph.bw_delays["worker_"+node_rank]
                    self.trail_dag.nodes[node_][GAP_STR_OP2COMM] = gap

    def clip_recv_events(self):
        self.logger.info("Clip RECV events...")
        for u, v in self.trail_dag.edges:
            if "I/O" in u:
                continue

            u_idx_l = self.name2idxlist[u] if u in self.name2idxlist else None
            v_idx_l = self.name2idxlist[v] if v in self.name2idxlist else None
            if u_idx_l is None or v_idx_l is None:
                ### some dag nodes do not appear in the traces
                continue

            gap = 0
            n = 0
            if not args_.disable_revise and "SEND" in u and "RECV" in v:
                ### Revise RECV events according to SEND-RECV dependents
                ### TODO (huhanpeng): cat2sta's maximum has not be updated
                recv_dict = None
                for cnt_ in range(self.traceM.max_cnt):
                    u_idx, v_idx = u_idx_l[cnt_], v_idx_l[cnt_]
                    if u_idx is None or v_idx is None:
                        continue
                    ### if RECV.start_t() < SEND.start_t(), revise
                    ### RECV.dur = RECV.dur - (SEND.ts - RECV.ts)
                    ### RECV.ts = SEND.ts
                    send_event = self.traceM.traces[u_idx]
                    recv_event = self.traceM.traces[v_idx]
                    how_much_less = 0
                    if send_event["ts"] > recv_event["ts"]:
                        temp_dur = recv_event["dur"]
                        recv_event["dur"] = max(recv_event["dur"] - (send_event["ts"] - recv_event["ts"]), 0)
                        recv_event["ts"] = send_event["ts"]
                        how_much_less = temp_dur - recv_event["dur"]
                    if recv_dict is None:
                        recv_dict = {
                            "unique_name": self.traceM.ret_unique_name(recv_event),
                            "durs": [recv_event["dur"]],
                            "less": how_much_less
                        }
                    else:
                        recv_dict["durs"].append(recv_event["dur"])
                        recv_dict["less"] += how_much_less
                if recv_dict is not None:
                    name_ = recv_dict["unique_name"]
                    ### Update statistical information: name2sta
                    avg = sum(recv_dict["durs"]) / len(recv_dict["durs"]) / 1000.0
                    self.traceM.name2sta[name_]["avg"] = avg 
                    var_l = [pow(_d / 1000.0 - avg, 2) for _d in recv_dict["durs"]]
                    self.traceM.name2sta[recv_dict["unique_name"]]["var"] = sum(var_l) / len(var_l)
                    ### Update statistical information: cat2sta
                    if recv_dict["less"] != 0:
                        cat = parse_cat_fine_grained(name_)
                        self.traceM.cat2sta[cat]["time"] -= recv_dict["less"]
                        self.traceM.cat2sta[cat]["avg"] = self.traceM.cat2sta[cat]["time"] / self.traceM.cat2sta[cat]["cnt"]
                    ### Update DAG information
                    for next_ in self.trail_dag.successors(name_):
                        self.trail_dag.edges[name_, next_]["weight"] = avg

    def add_avg_to_nodes(self):
        for node_ in self.trail_dag.nodes:
            ### Add duration to the node as an attribute
            self.trail_dag.nodes[node_]["avg"] = self.traceM.lookup_stat(None, None, node_)

            if parse_cat_from_name(node_) == CatName.COMM.value:
                if "avg" in self.trail_dag.nodes[node_]:
                    self.trail_dag.nodes[node_]["avg"] /= 10

    def add_gaps_clip_events(self):
        ''' According to the traces and DAG, add a 'gap' field for each edge (u, v)
        which denotes the gap between two events u and v.
        '''
        name2idxlist = {}
        for idx, event in enumerate(self.traceM.traces):
            if self.traceM._is_ignore_for_sta(event):
                continue

            unique_name = self.traceM.ret_unique_name(event)
            if unique_name not in name2idxlist:
                name2idxlist[unique_name] = [None] * self.traceM.max_cnt

            if event["args"]["cnt"] != -1:
                name2idxlist[unique_name][event["args"]["cnt"]] = idx

        ### Calculate the average gap for each edge
        for u, v in self.trail_dag.edges:
            # if "I/O" in u or ("BW" not in u and "UPDATE_" in v):
            if "I/O" in u:
                self.trail_dag.edges[u, v]["gap"] = 0
                self.trail_dag.edges[u, v]["cost"] = self.trail_dag.edges[u, v]["weight"]
                continue

            u_idx_l = name2idxlist[u] if u in name2idxlist else None
            v_idx_l = name2idxlist[v] if v in name2idxlist else None
            if u_idx_l is None or v_idx_l is None:
                ### some dag nodes do not appear in the traces
                if u_idx_l is not None:
                    self.trail_dag.edges[u, v]["cost"] = self.trail_dag.edges[u, v]["weight"]
                continue

            gap = 0
            n = 0
            if not args_.disable_revise and "SEND" in u and "RECV" in v:
                ### Revise RECV events according to SEND-RECV dependents
                ### TODO (huhanpeng): cat2sta's maximum has not be updated
                recv_dict = None
                for cnt_ in range(self.traceM.max_cnt):
                    u_idx, v_idx = u_idx_l[cnt_], v_idx_l[cnt_]
                    if u_idx is None or v_idx is None:
                        continue
                    ### if RECV.start_t() < SEND.start_t(), revise
                    ### RECV.dur = RECV.dur - (SEND.ts - RECV.ts)
                    ### RECV.ts = SEND.ts
                    send_event = self.traceM.traces[u_idx]
                    recv_event = self.traceM.traces[v_idx]
                    how_much_less = 0
                    if send_event["ts"] > recv_event["ts"]:
                        temp_dur = recv_event["dur"]
                        recv_event["dur"] = max(recv_event["dur"] - (send_event["ts"] - recv_event["ts"]), 0)
                        recv_event["ts"] = send_event["ts"]
                        how_much_less = temp_dur - recv_event["dur"]
                    if recv_dict is None:
                        recv_dict = {
                            "unique_name": self.traceM.ret_unique_name(recv_event),
                            "durs": [recv_event["dur"]],
                            "less": how_much_less
                        }
                    else:
                        recv_dict["durs"].append(recv_event["dur"])
                        recv_dict["less"] += how_much_less
                if recv_dict is not None:
                    name_ = recv_dict["unique_name"]
                    ### Update statistical information: name2sta
                    avg = sum(recv_dict["durs"]) / len(recv_dict["durs"]) / 1000.0
                    self.traceM.name2sta[name_]["avg"] = avg 
                    var_l = [pow(_d / 1000.0 - avg, 2) for _d in recv_dict["durs"]]
                    self.traceM.name2sta[recv_dict["unique_name"]]["var"] = sum(var_l) / len(var_l)
                    ### Update statistical information: cat2sta
                    if recv_dict["less"] != 0:
                        cat = parse_cat_fine_grained(name_)
                        self.traceM.cat2sta[cat]["time"] -= recv_dict["less"]
                        self.traceM.cat2sta[cat]["avg"] = self.traceM.cat2sta[cat]["time"] / self.traceM.cat2sta[cat]["cnt"]
                    ### Update DAG information
                    for next_ in self.trail_dag.successors(name_):
                        self.trail_dag.edges[name_, next_]["weight"] = avg
            else:
                for cnt_ in range(self.traceM.max_cnt):
                    u_idx, v_idx = u_idx_l[cnt_], v_idx_l[cnt_]
                    if u_idx is None or v_idx is None:
                        continue
                    gap += (self.traceM.traces[v_idx]["ts"] - (self.traceM.traces[u_idx]["ts"] + self.traceM.traces[u_idx]["dur"]))
                    n += 1
            if gap < 0 and not ("SEND" in u and "RECV" in v):
                self.logger.warn("The gap < 0 between %s and %s" % (u, v))
                # raise
            gap = 0 if n == 0 else gap / float(n)
            self.trail_dag.edges[u, v]["gap"] = gap
            self.trail_dag.edges[u, v]["cost"] = gap / 1000 + self.trail_dag.edges[u, v]["weight"]

    def detect_straggler1(self):
        prefix2traces = {}
        def _get_prefix(e):
            prefix = e["pid"]
            if prefix not in prefix2traces:
                prefix2traces[prefix] = []
            return prefix
        for event in self.traceM.traces:
            if not self.traceM._is_ignore_for_sta(event):
                prefix2traces[_get_prefix(event)].append(event)
        name2sta_list = []
        sheet_name = []
        for key_ in sorted(prefix2traces.keys()):
            prefix2traces[key_] = TraceManager(prefix2traces[key_], DirLevel.GPU)
            print("\n%s" % key_)
            for cat_, sta_ in sorted(prefix2traces[key_].cat2sta.items()):
                print("Cat: %-20s: Avg %-12.4f ms, Op Count %-10d" % (cat_, sta_["avg"], sta_["op_cnt"]))
            name2sta_list.append(prefix2traces[key_].name2sta)
            sheet_name.append(key_)
        
        self.traceM.export2xlsx(name2sta_list, self.pm.path, filename="diagnosis", sheet_name=sheet_name)

    def detect_bottleneck1(self):
        # critical_path = dag_longest_path(self.trail_dag, self.pm, weight="cost", default_weight=0, _debug_level=2)
        critical_path = dag_longest_path(self.trail_dag, self.pm, weight="weight", default_weight=0, _debug_level=2)
        return critical_path


