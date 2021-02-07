from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse as parse_protobuf_json
import tensorflow as tf
import os
import threading
import multiprocessing
import itertools
import time
import copy
# Try to support both tf2 and tf1
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef

from tensorflow.python.client import timeline
import json
import networkx as nx
try:
    import byteps.tensorflow as bps
except:
    pass
try:
    import horovod.tensorflow as hvd
except:
    pass

class TimelineSession:
    def __init__(self, sess):
        self.sess = sess
        self.graph = sess.graph
        self.step_cnt = 0
        self.feed_dict_meta = {}

        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(hvd.local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            return
        self._end_trace = False
        self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")   

        ### Timeline configuratoin
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
        self.run_metadata = tf.RunMetadata()
        self.traces = {"traceEvents":[]}
        self.trace_meta = {}
        self.next_unique_pid = 0

        self.shape_dict = {}

        self.dag = None

    def run(self, *args_, **kwargs_):
        if self._end_trace:
            ret = self.sess.run(*args_, **kwargs_)
        elif not self._end_trace and self.step_cnt < self.start_step:
            ret = self.sess.run(*args_, **kwargs_)
            self.step_cnt += 1
        elif not self._end_trace and self.step_cnt < self.end_step:
            ret = self.sess.run(*args_, options=self.run_options, run_metadata=self.run_metadata, **kwargs_)
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(self.run_metadata.step_stats)
            ctf = json.loads(tl.generate_chrome_trace_format())
            filtered_trace_events = []
            pid_mapping_in_this_step = {}
            for event in ctf["traceEvents"]:
                if "ph" in event and event["ph"] == "M":
                    if "name" in event and event["name"] == "process_name":
                        if "args" in event:
                            pid_mapping_in_this_step[event["pid"]] = event["args"]["name"]
                            if event["args"]["name"] not in self.trace_meta:
                                self.trace_meta[event["args"]["name"]] = self.next_unique_pid
                                event["pid"] = self.next_unique_pid
                                self.traces["traceEvents"].append(event)
                                self.next_unique_pid += 1
            for event in ctf["traceEvents"]:
                if "ph" in event and event["ph"] != "M":
                    if "pid" in event:
                        event["pid"] = self.trace_meta[pid_mapping_in_this_step[event["pid"]]]
                    filtered_trace_events.append(event)

            self.traces["traceEvents"] += filtered_trace_events
            print("Add the {}th step of traces".format(self.step_cnt))
            self.step_cnt += 1

            ### Create the DAG
            if self.dag is None:
                self.dag = nx.DiGraph()
                for trace in filtered_trace_events:
                    if trace["ph"] == "M" or "args" not in trace:
                        continue
                    op = trace["args"]["op"]
                    name = trace["args"]["name"]

                    ### Add nodes to the DAG
                    if name not in self.dag.nodes:
                        self.dag.add_node(name)

                    ### Add dependency info
                    for k, v in trace["args"].items():
                        if "input" in k:
                            self.dag.add_edge(v, name)

            try:
                not_found = False
                nx.find_cycle(self.dag.cycle)
            except:
                not_found = True
            assert not_found

            def flatten_fetch_list(fetch_list):
                if not isinstance(fetch_list, (list, tuple)):
                    return [fetch_list]
                else:
                    result_list = []
                    for op in fetch_list:
                        result_list += flatten_fetch_list(op)
                    return result_list

            # get shapes from step_stats
            for dev_stats in self.run_metadata.step_stats.dev_stats:
                for node_stats in dev_stats.node_stats:
                    for node_outputs in node_stats.output:
                        slot = node_outputs.slot
                        dtype = node_outputs.tensor_description.dtype
                        shape = []
                        if node_outputs.tensor_description.shape.unknown_rank:
                            shape.append("Unknown")
                        else:
                            for shape_in_dim in node_outputs.tensor_description.shape.dim:
                                shape.append(shape_in_dim.size)
                        if node_stats.node_name+":{}".format(slot) not in self.shape_dict:
                            self.shape_dict[node_stats.node_name+":{}".format(slot)] = {}
                        self.shape_dict[node_stats.node_name+":{}".format(slot)]["shape"] = shape
                        self.shape_dict[node_stats.node_name+":{}".format(slot)]["dtype"] = dtype

            ### Output traces
            if self.step_cnt == self.end_step:
                for idx in range(len(self.run_metadata.partition_graphs)):
                    graph_def = self.run_metadata.partition_graphs[idx]
                    graph_str = json.loads(MessageToJson(graph_def))
                    with open(os.path.join(self.trace_dir, "partition_def_{}.json".format(idx)), "w") as f:
                        json.dump(graph_str, f, indent=4)
                fd = kwargs_.get("feed_dict")
                # collect feed dict meta
                self.fetches = [tensor.name for tensor in flatten_fetch_list(args_[0])]
                for key, tensor in fd.items():
                    shape_as_list = [int(dim) for dim in tensor.shape]
                    dtype_as_str = (str(tensor.dtype).split("\'")[1] if "\'" in str(tensor.dtype) else str(tensor.dtype)).split("_ref")[0]
                    self.feed_dict_meta[key.op.name] = {"shape": shape_as_list, 
                                                    "dtype": dtype_as_str}
                self._end_trace = True
                self.output_traces()
        else:
            ret = self.sess.run(*args_, **kwargs_)

        ### Return all fetches
        return ret
    
    def output_traces(self):
        # https://stackoverflow.com/questions/57557564/tensorflow-reload-mode-to-session-from-graph-def
        var_ops = [op for op in self.sess.graph.get_operations() if op.type == 'VariableV2']
        # Get the values
        var_shapes = {}
        for v in var_ops:
            try:
                shape_as_list = [int(dim) for dim in v.outputs[0].shape]
                dtype_as_str = (str(v.outputs[0].dtype).split("\'")[1] if "\'" in str(v.outputs[0].dtype) else str(v.outputs[0].dtype)).split("_ref")[0]
                var_shapes[v.name] = {"shape": shape_as_list, "dtype": str(v.outputs[0].dtype).split("\'")[1].split("_ref")[0]}
            except tf.errors.FailedPreconditionError:
                # Uninitialized variables are ignored
                pass
        with open(os.path.join(self.trace_dir, "variables_meta.json"), "w") as f:
            json.dump(var_shapes, f, indent=4)

        # with open(os.path.join(self.trace_dir, "temp.json"), "w") as f:
        #     json.dump(self.traces, f, indent=4)
        
        with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
            json.dump(self.shape_dict, f, indent=4)
        
        with open(os.path.join(self.trace_dir, "run_meta.json"), "w") as f:
            json.dump({"fetches":self.fetches, "feed_dict": self.feed_dict_meta}, f, indent=4)

        ## collect graph info
        graphdef = tf.get_default_graph().as_graph_def(add_shapes=True)
        graph_str = json.loads(MessageToJson(graphdef))
        with open(os.path.join(self.trace_dir, "final_graph.json"), "w") as f:
            json.dump(graph_str, f, indent=4)

        # nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        # print("Stop tracing, output trace: %s" % self.trace_dir)

    def should_stop(self):
        return self.sess.should_stop()

class _SecondOrStepTimer(tf.train.SecondOrStepTimer):
    def __init__(self, every_secs=None, every_steps=None, step_bound=None):
        if step_bound is not None:
            if not (isinstance(step_bound, list) or isinstance(step_bound, tuple)):
                raise ValueError("step bound must be a list or a tuple, but {} is given".format(step_bound))
            self._start_step = step_bound[0]
            self._end_step = step_bound[1]
            if self._start_step > self._end_step:
                raise ValueError("Profiling start step must be smaller than the end step.")
        else:
            self._start_step = self._end_step = None

        super(_SecondOrStepTimer, self).__init__(every_secs, every_steps)

    def should_trigger_for_step(self, step):
        if self._start_step is not None:
            if step < self._start_step or step > self._end_step:
                return False

        return super(_SecondOrStepTimer, self).should_trigger_for_step(step)

class TimelineHook(tf.train.ProfilerHook):
    def __init__(self, _summary=False, batch_size=None):
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(bps.local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)

        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            self.start_step = self.end_step = 0
        else:
            self._end_trace = False
            self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
            self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        
        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")
        
        print("TimelineHook enable: {}  start_step: {} end_step: {}".format(not self._end_trace, self.start_step, self.end_step))
        self.dag = None
        self.has_data = False

        self.shape_dict = {}
        self.run_metadata = None
        self.partition_dag = None
        self.step_stats = []

        self._output_file = os.path.join(self.trace_dir, "timeline-{}.json")
        self._file_writer = tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
        self._show_dataflow = True
        self._show_memory = False
        self._timer = _SecondOrStepTimer(
            every_secs=None, every_steps=1, step_bound=(self.start_step, self.end_step))
        self.batch_size = batch_size
        assert self.batch_size is not None

    def before_run(self, run_context):
        t = time.time()
        if not self._end_trace:
            self._request_summary = (
                self._next_step is not None and
                self._timer.should_trigger_for_step(self._next_step))
            
            if self._request_summary and not self.has_data:
                ### the first step to collect traces, self.has_data tells there are data that need outputing
                self.has_data = True
            if self.has_data and not self._request_summary:
                ### the step after the last trace step, output data
                self._end_trace = True
                partition_graphs = []
                for idx in range(len(self.run_metadata.partition_graphs)):
                    graph_def = self.run_metadata.partition_graphs[idx]
                    partition_graphs.append(graph_def)
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(), partition_graphs))
                _t.start()
        else:
            self._request_summary = False
                
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
            if self._request_summary else None)

        t = time.time() - t
        print("Before run takes: {} seconds".format(t))
        return tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        t = time.time()
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
        # Update the timer so that it does not activate until N steps or seconds
        # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            self.run_metadata = run_values.run_metadata
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            # _t = multiprocessing.Process(target=self._save, args=(global_step, self._output_file.format(global_step),
            #          run_values.run_metadata.step_stats))
            # _t.start()
            self.step_stats.append(copy.deepcopy(run_values.run_metadata.step_stats))
            # self._save(global_step, self._output_file.format(global_step),
            #         run_values.run_metadata.step_stats)
            # get shapes from step_stats
            if bps.rank() == 0 and bps.local_rank() == 0:
                if not self.shape_dict:
                    for dev_stats in run_values.run_metadata.step_stats.dev_stats:
                        for node_stats in dev_stats.node_stats:
                            for node_outputs in node_stats.output:
                                slot = node_outputs.slot
                                dtype = node_outputs.tensor_description.dtype
                                shape = []
                                if node_outputs.tensor_description.shape.unknown_rank:
                                    shape.append("Unknown")
                                else:
                                    for shape_in_dim in node_outputs.tensor_description.shape.dim:
                                        shape.append(shape_in_dim.size)
                                if node_stats.node_name+":{}".format(slot) not in self.shape_dict:
                                    self.shape_dict[node_stats.node_name+":{}".format(slot)] = {}
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["shape"] = shape
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["dtype"] = dtype
            if self._file_writer is not None:
                self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)
        self._next_step = global_step + 1
        t = time.time() - t
        print("After run takes: {} seconds".format(t))

    def output_traces(self, ops, partition_graphs):
        self.traces = {"traceEvents":[]}
        ### the ProfilerHook of tensorflow will output the timeline to self.trace_dir/timeline-{global_step}.json
        # for file in sorted(os.listdir(self.trace_dir)):
        #     if file.startswith('timeline-'):
        #         with open(os.path.join(self.trace_dir, file), 'r') as fp:
        #             ctf = json.load(fp)
        #         convert_traces = self.chome_trace_MBE2X(ctf["traceEvents"])
        #         self.traces["traceEvents"] += convert_traces 

        for step_stats in self.step_stats:
            trace = timeline.Timeline(step_stats)
            events_str = trace.generate_chrome_trace_format(
                    show_dataflow=self._show_dataflow, show_memory=self._show_memory)
            events = json.loads(events_str)
            self.traces["traceEvents"] += self.chome_trace_MBE2X(events["traceEvents"])
        
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as fp:
            json.dump(self.traces, fp, indent=4)

        if os.getenv("BYTEPS_PURE_TF_TRACE", '1') == '1':
            ### delete all intermediate redults
            _output_files = os.path.join(self.trace_dir, "timeline-*.json")
            os.system('rm {}'.format(_output_files))

        def serialize_tensor(t):
            _shape = t.shape.as_list() if t.shape.dims is not None else []
            if len(_shape) > 0 and _shape[0] is None:
                _shape[0] = self.batch_size
            return {
                "name": t.name,
                "shape": _shape,
                "dtype": t.dtype.name
            }

        for idx, graph_def in enumerate(partition_graphs):
            graph_json = json.loads(MessageToJson(graph_def))
            with open(os.path.join(self.trace_dir, "partition_def_{}.json".format(idx)), "w") as f:
                json.dump(graph_json, f, indent=4)
            
            if idx == 0:
                # generate dag
                self.partition_dag = nx.DiGraph()
                # clean node names in graph def
                pruned_node = set()
                all_node_names = set([node["name"] if node["name"][0] != "_" else node["name"][1:] \
                                                                    for node in graph_json["node"]])
                for node in graph_json["node"]:
                    if node["name"][0] == "_":
                        node["name"] = node["name"][1:]
                    last_slash_pos = node["name"].rfind("/")
                    if last_slash_pos != -1 and last_slash_pos < len(node["name"])-1 \
                                            and node["name"][last_slash_pos+1] == "_":
                        if node["name"][:last_slash_pos] in all_node_names:
                            pruned_node.add(node["name"])
                            continue
                        else:
                            node["name"] = node["name"][:last_slash_pos]
                    if "input" in node:
                        for idx, input_node in enumerate(node["input"]):
                            if input_node[0] == "_":
                                node["input"][idx] = input_node[1:]
                                input_node = input_node[1:]
                            last_slash_pos = input_node.rfind("/")
                            if last_slash_pos != -1 and last_slash_pos < len(input_node)-1 \
                                                    and input_node[last_slash_pos+1] == "_":
                                node["input"][idx] = input_node[:last_slash_pos]
                            self.partition_dag.add_edge(node["input"][idx].split(":")[0], node["name"])

        if bps.rank() == 0:
            ### Only dump these info for rank 0   
            op_dict = {}
            for op in ops:
                op_dict[op.name] = {
                    "output":[serialize_tensor(e) for e in op.outputs],
                    "input": [serialize_tensor(e) for e in op.inputs._inputs],
                    "op": op.type
                }
            with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
                json.dump(op_dict, f, indent=4)

            if self.partition_dag is not None:
                nx.write_gml(self.partition_dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
            
            with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
                json.dump(self.shape_dict, f, indent=4)

        print("Stop tracing, output trace at %s" % self.trace_dir)

    def chome_trace_MBE2X(self, raw_traces):
        ret = []
        pid_table = {}
        if self.dag is None:
            _dag = nx.DiGraph()
        for trace in raw_traces:
            ### Create the DAG
            if self.dag is None:
                if trace["ph"] == "M" or "args" not in trace:
                    continue
                op = trace["args"]["op"]
                name = trace["args"]["name"]
                if name.startswith("^"):
                    name = name[1:]
                ### Add dependency info
                for k, v in trace["args"].items():
                    if "input" in k:
                        if v.startswith("^"):
                            v = v[1:]
                        _dag.add_edge(v, name)
                    
            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in pid_table
                    if trace["args"]["name"] == "":
                        continue
                    process_name = trace["args"]["name"]
                    if "stream:all Compute" in process_name and "device:GPU" in process_name:
                        pid_table[trace["pid"]] = {"process_name": process_name}
                else:
                    pass
            elif trace["ph"] == "i":
                trace["pid"] = trace["tid"] = "mark"
                ret.append(trace)
            elif trace["pid"] in pid_table and trace["ph"] == "X":
                cur_pid = pid_table[trace["pid"]]
                trace["pid"] = cur_pid["process_name"]
                ret.append(trace)
            else:
                pass
        if self.dag is None:
            self.dag = _dag
        return ret

def load_graph_def_from_json(graph_def_json_path):
    with open(graph_def_json_path, "r") as f:
        graph_def = parse_protobuf_json(f.read(), GraphDef())
    return graph_def

def dump_computation_graph(trace_dir):
    graphdef = tf.compat.v1.get_default_graph().as_graph_def()
    graph_str = json.loads(MessageToJson(graphdef))
    with open(os.path.join(trace_dir, "graph.json"), "w") as f:
        json.dump(graph_str, f, indent=4)

def add_infer_shape_ops(graph=None):
    # add output_shape ops
    if graph is None:
        graph = tf.compat.v1.get_default_graph()
    # collect tensor shapes
    all_ops = graph.get_operations()
    tensor_shape_ops = []
    tensor_names = []
    with graph.as_default():
        for op in all_ops:
            for output in op.outputs:
                tensor_names.append(output.name)
                name, idx = output.name.split(":")
                tensor_shape_ops.append(tf.shape(output, name="final_shape/"+name+"_"+idx))
    return (tensor_names, tensor_shape_ops)
