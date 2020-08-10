from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse as parse_protobuf_json
import tensorflow as tf
import itertools
# Try to support both tf2 and tf1
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef

from tensorflow.python.client import timeline
import json
import networkx as nx
class TimelineSession:
    def __init__(self, sess, tensor_shape_ops=None):
        self.sess = sess
        self.graph = sess.graph
        self.step_cnt = 0
        self.feed_dict_meta = {}
        self.tensor_shape_ops = tensor_shape_ops

        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(bps.local_rank()))
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
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.traces = {"traceEvents":[]}

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
            self.traces["traceEvents"] += ctf["traceEvents"]
            print("Add the {}th step of traces".format(self.step_cnt))
            self.step_cnt += 1

            ### Create the DAG
            if self.dag is None:
                self.dag = nx.DiGraph()
                for trace in ctf["traceEvents"]:
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

            ### Output traces
            if self.step_cnt == self.end_step:
                fd = kwargs_.get("feed_dict")
                tensor_names, tensor_shape_ops = self.tensor_shape_ops
                out_shapes = self.sess.run(tensor_shape_ops, feed_dict=fd)
                self.tensor_shapes = {}
                for name, shape in zip(tensor_names, out_shapes):
                    self.tensor_shapes[name] = [int(s) for s in list(shape)]
                # collect feed dict meta
                self.fetches = [tensor.name for tensor in flatten_fetch_list(args_[0])]
                for key, tensor in fd.items():
                    shape_as_list = [int(dim) for dim in tensor.shape]
                    dtype_as_str = (str(tensor.dtype).split("\'")[1] if "\'" in str(tensor.dtype) else str(tensor.dtype)).split("_ref")[0]
                    self.feed_dict_meta[key.op.name] = {"shape": shape_as_list, 
                                                    "dtype": dtype_as_str}
                self._end_trace = True
                self.output_traces()

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

        with open(os.path.join(self.trace_dir, "temp.json"), "w") as f:
            json.dump(self.traces, f, indent=4)
        
        with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
            json.dump(self.tensor_shapes, f, indent=4)
        
        with open(os.path.join(self.trace_dir, "run_meta.json"), "w") as f:
            json.dump({"fetches":self.fetches, "feed_dict": self.feed_dict_meta}, f, indent=4)

        ## collect graph info
        graphdef = tf.get_default_graph().as_graph_def(add_shapes=True)
        graph_str = json.loads(MessageToJson(graphdef))
        with open(os.path.join(self.trace_dir, "final_graph.json"), "w") as f:
            json.dump(graph_str, f, indent=4)

        nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        print("Stop tracing, output trace: %s" % self.trace_dir)

    def should_stop(self):
        return self.sess.should_stop()

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