import os
import json
from ...trace_utils import PathManager, RunningSpan, FileName, SingleLogger

from . import memory_lists

def collect_rank_comp_tf(tmp_pm=None, pid=None, host_id=None,
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
    if comp_path.endswith(".gz"):
        os.system("gzip -fdk {}".format(comp_path))
        json_path = comp_path.replace(".gz", "")
        with open(json_path, 'r') as fp:
            raw_traces = json.load(fp)["traceEvents"]
        os.system("rm {}".format(json_path))
    else:
        with open(comp_path, 'r') as fp:
            raw_traces = json.load(fp)["traceEvents"]

    ### Consider mulptiprocessing, each GPU will read its own dag
    dag_node_names_std = list(dag.nodes)

    wk_prefix, _ = PathManager("/".join(comp_path.split('/')[:-1])).ret_prefix()
    if wk_prefix not in run_span:
        run_span[wk_prefix] = RunningSpan()

    ### collect traces of FW + BP OPs and UPDATE OPs
    tid_hub = []
    def _ret_operator_tid(tid_):
        if tid_ in tid_hub:
            return "operator%d"%(tid_hub.index(tid_))
        else:
            tid_hub.append(tid_)
            return "operator%d"%(len(tid_hub) - 1)

    rst_traces = []
    pid_dict = {}
    for trace in raw_traces:
        if "ph" not in trace:
            continue
        if trace["ph"] == "M":
            if trace["name"] == "process_name" and trace["pid"] not in pid_dict:
                pid_dict[trace["pid"]] = {"name": trace["args"]["name"]}
            if trace["name"] == "thread_name" and trace["tid"] not in pid_dict[trace["pid"]]:
                pid_dict[trace["pid"]][trace["tid"]] = trace["args"]["name"]
        elif trace["ph"] == "X":
            pid_name = pid_dict[trace["pid"]]["name"]
            tid_name = pid_dict[trace["pid"]][trace["tid"]]
            if "/device:GPU" in pid_name:
                if "TensorFlow Ops" in tid_name:
                    try:
                        raw_name = trace["args"]["long_name"].split(":")[0]
                        raw_name = raw_name.split("StatefulPartitionedCall/")[1] if raw_name.startswith("StatefulPartitionedCall") else raw_name
                    except IndexError:
                        print(trace)
                        continue
                    except:
                        print(trace)
                        raise

                    # op_type = trace["name"]
                    name = name_hook_fn(raw_name)
                    
                    ### Only collect nodes in the dag
                    ### TODO (huhanpeng): some trvial nodes may also be useful
                    if name not in dag_node_names_std or "Comm" in name:
                        if trace_level == "debug":
                            trace["name"] = "%s" % (trace["name"])
                            trace["tid"] = trace["cat"] = "debug"
                            if pid is not None:
                                trace["pid"] = pid
                            rst_traces.append(trace)
                        continue
                    
                    ### Record dependency info to traces
                    innodes = [_n for _n, _ in dag.in_edges(name)]
                    try:
                        _args = {
                            "name": name,
                            "step": trace["args"]["group_id"]
                        }
                    except:
                        _args = {
                            "name": name
                        }
                    for i, _n in enumerate(innodes):
                        _args["input%d"%i] = _n

                    rst_traces.append({
                        "name": name,
                        "ph": "X",
                        "ts": trace["ts"],
                        "dur": trace["dur"],
                        "pid": pid,
                        "tid": _ret_operator_tid(tid_name),
                        "cat": "operator",
                        "args": _args
                    })

                    run_span[wk_prefix].init_start(trace["ts"])
                    run_span[wk_prefix].init_end(trace["ts"] + trace["dur"])

    rst_traces = sorted(rst_traces, key=lambda x: x["ts"], reverse=False)
    
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

    SingleLogger().debug("Comp traces length: {}".format(len(rst_traces)))
    return rst_traces


    
    
    