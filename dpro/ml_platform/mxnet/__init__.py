import json
from ...trace_utils import PathManager, RunningSpan, FileName, SingleLogger

def collect_rank_comp_mx(tmp_pm=None, pid=None, host_id=None,
        update_clip_overlapping=False,
        name_hook_fn=lambda x: x,
        run_span=None, dag=None,
        trace_level="info"):
    '''Collect Computation Traces

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
    # debug_utils.DebugRecorder().debug_event_start()
    comp_path = tmp_pm.search(FileName.COMP)
    if comp_path is None:
        return
    with open(comp_path, 'r') as fp:
        raw_traces = json.load(fp)["traceEvents"]

    dag_node_names_std = list(dag.nodes)

    wk_prefix, _ = PathManager("/".join(comp_path.split('/')[:-1])).ret_prefix()
    if wk_prefix not in run_span:
        run_span[wk_prefix] = RunningSpan()

    one_pid = None
    # if self.platform == "TENSORFLOW":
    kernel_pid = None
    kernel_times = {}

    ### collect traces of FW + BP OPs and UPDATE OPs
    tid_hub = []
    def _ret_operator_tid(tid_):
        if tid_ in tid_hub:
            return "operator%d"%(tid_hub.index(tid_))
        else:
            tid_hub.append(tid_)
            return "operator%d"%(len(tid_hub) - 1)

    ### convert the ph from B/E to X
    ### TODO(huhanpeng): delete this, since it is only for Tensorflow ???
    index = 0
    traces = []
    while index < len(raw_traces["traceEvents"]):
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

    ### At this point, traces are unsorted
    # debug_utils.DebugRecorder().debug_event_start()
    traces = sorted(traces, key=lambda x: x["ts"], reverse=False)
    SingleLogger().debug("Original Comp traces length: {}.".format(len(traces)))
    # debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.sorted", "Collct", "0")

    ### find the last BP op in the timeline
    def real_last_bw_name():
        statue = "init"
        _index = 0
        last_bw = None
        last_fw = None
        first_bw = None
        while _index < len(traces):
            trace = traces[_index]
            _index += 1
            name = name_hook_fn(trace["name"])
            if name not in dag.nodes:
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
    # debug_utils.DebugRecorder().debug_event_start()          
    last_fw, first_bw, last_bw = real_last_bw_name()
    # debug_utils.DebugRecorder().debug_event_end("collect_" + pid+"_comp.real_last_bw_name", "Collct", "0")

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

    index = 0
    rst_traces = []
    pre_event = None
    while index < len(traces):
        trace = traces[index]
        index += 1
        raw_name = trace["name"]
        name = name_hook_fn(raw_name)

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

        if name not in dag_node_names_std or "Comm" in name:
            ### Only collect nodes in the dag
            ### TODO (huhanpeng): some trvial nodes may also be useful
            if trace_level == "debug":
                trace["name"] = "%s.%d"%(trace["name"], index)
                trace["tid"] = trace["cat"] = "debug"
                if pid is not None:
                    trace["pid"] = pid
                rst_traces.append(trace)
            # print(trace)
            continue
        ### TODO (huhanpeng): TF profiles CUDA kernels and lables kernels in the same 
        ### operator with the same operator name. To make trace name unique, we combine
        ### kernel-level traces to operator level traces
        if pre_event is not None and "args" in pre_event and pre_event["args"]["name"] == name:
            pre_event["dur"] = trace["ts"] + trace["dur"] - pre_event["ts"]
        else:
            innodes = [_n for _n, _ in dag.in_edges(name)]
            _args = {"name": name}
            for i, _n in enumerate(innodes):
                _args["input%d"%i] = _n
            trace["name"] = name
            trace["args"] = _args
            if pid is not None:
                trace["pid"] = pid
            trace["tid"] = _ret_operator_tid(trace["tid"])
            rst_traces.append(trace)
            ### Initialize the start time of the entire running span
            run_span[wk_prefix].init_start(trace["ts"])
            pre_event = trace

        ### Handle OUTPUT
        if name == last_fw: # type: ignore
            output_ts = None
            output_dur = None
            output_tid = None
            while index < len(traces):
                _trace = traces[index]
                if one_pid != _trace["pid"]:
                    index += 1
                else:
                    name = name_hook_fn(_trace["name"])
                    if name == first_bw or name in dag.nodes: # type: ignore
                        break
                    output_ts = _trace["ts"] if output_ts is None else output_ts
                    output_dur = _trace["ts"] + _trace["dur"] - output_ts
                    output_tid = _trace["tid"] if output_tid is None else output_tid
                    index += 1
            if output_ts is not None and output_dur is not None:
                rst_traces.append({
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
        elif name == last_bw: # type: ignore
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
                    name = name_hook_fn(_trace["name"])
                    if name in dag.nodes:
                        break
                    index += 1
                    if is_cal_op(_trace): # type: ignore
                        if _cal_ts is None:
                            _cal_ts = _trace["ts"]
                            _cal_tid = _trace["tid"]
                        _duration = _trace["ts"] + _trace["dur"] - _cal_ts
                    if is_update_op(_trace): # type: ignore
                        if _update_ts is None:
                            _update_ts = _trace["ts"]
                            ### Add UPDATE_CAL node
                            rst_traces.append({
                                "name": "UPDATE_CAL",
                                "ts": _cal_ts if _cal_ts is not None else _update_ts,
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
                        rst_traces.append({
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
                run_span[wk_prefix].init_end(_update_ts + _duration)
    
    if update_clip_overlapping:
        # trim the overlapping parts of update nodes
        # assume the end time of each event is accurate
        update_nodes = []
        for ev in rst_traces:
            if "UPDATE" in ev["name"]:
                update_nodes.append(ev)
        end_times = [ev["ts"] + ev["dur"] for ev in update_nodes]

        sorted_end_times, sorted_update_nodes = zip(*sorted(zip(end_times, update_nodes), key=lambda x: x[0]))
        for idx, ev in enumerate(sorted_update_nodes):
            if idx == 0:
                continue
            start_time = ev["ts"]
            if start_time < sorted_end_times[idx - 1]:
                # needs clipping
                ev["ts"] = sorted_end_times[idx - 1]
                ev["dur"] = sorted_end_times[idx] - ev["ts"]

    SingleLogger().debug("Comp traces length: {}.".format(len(rst_traces)))
    return rst_traces