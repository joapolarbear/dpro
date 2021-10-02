from argparse import ArgumentError
from enum import Enum
import numpy as np
import networkx as nx
from intervaltree import IntervalTree
import hashlib
import json
import pickle

from logger_utils import *
from trace_utils import *
import arg_utils
from bps_helper.preprocess import parse_server_logs #,preprocess_pcap

import cvxpy as cp

args_ = arg_utils.SingleArg().args

class PS_COMM_OPS():
    PUSH_REQ = "PUSH_REQ"
    PULL_REQ = "PULL_REQ"
    PUSH_RES = "PUSH_RES"
    PULL_RES = "PULL_RES"

class PS_COMP_OPS():
    COPY_FIRST = "COPY_FIRST"
    SUM = "SUM"
    COPY_MERGED = "COPY_MERGED"

COMM_DEL = "::"
PART_DEL = "~PART"

PS_COMM_OPS_SETS = set([PS_COMM_OPS.PUSH_REQ, PS_COMM_OPS.PULL_REQ, PS_COMM_OPS.PUSH_RES, PS_COMM_OPS.PULL_RES])

PS_COMP_OPS_SETS = set([PS_COMP_OPS.COPY_FIRST, PS_COMP_OPS.SUM, PS_COMP_OPS.COPY_MERGED])

class ServerOpCounter(object):
    def __init__(self, byteps_graph):
        self.byteps_graph = byteps_graph
        self.num_workers = byteps_graph.get_num_workers()
        self.num_servers = byteps_graph.get_num_servers()
        self.server_op_counter = {}
        for server in byteps_graph.get_servers():
            self.server_op_counter[server] = {}
    
    def get_next_op(self, name):
        raw_name = parse_rawname(name)
        source, target, tensor_name, sub_op, part_id = self.byteps_graph.parse_comm_event_name(raw_name)
        tensor_name_with_part_id = tensor_name + part_id
        if target in self.server_op_counter:
            # is a push or pull request to servers
            if sub_op == PS_COMM_OPS.PUSH_REQ:
                if tensor_name_with_part_id not in self.server_op_counter[target]:
                    self.server_op_counter[target][tensor_name_with_part_id] = 0
                if self.server_op_counter[target][tensor_name_with_part_id] == 0:
                    # first push, a copy_first sub_op
                    comp_key = (target, tensor_name, PS_COMP_OPS.COPY_FIRST,
                        self.byteps_graph.comp_ops_tid[(target, tensor_name, PS_COMP_OPS.COPY_FIRST, part_id)],
                        part_id)
                    res_name = self.byteps_graph.gen_comp_full_name(comp_key)
                else:
                    comp_key = (target, tensor_name, PS_COMP_OPS.SUM,
                        self.byteps_graph.comp_ops_tid[(target, tensor_name, PS_COMP_OPS.SUM, part_id)],
                        part_id)
                    res_name = self.byteps_graph.gen_comp_full_name(comp_key, sum_index=self.server_op_counter[target][tensor_name_with_part_id]-1)
                self.server_op_counter[target][tensor_name_with_part_id] += 1
                self.server_op_counter[target][tensor_name_with_part_id] = self.server_op_counter[target][tensor_name_with_part_id] % self.num_workers
            else:
                res_name = None
        else:
            res_name = None
        return res_name

def optimize_time_shift(shift_constraints: dict):
    # shift_constraints contains node-level time shift constraints
    # mapping: (node_id_0: int, node_id_1: int) -> value: float
    # denotes contraint: S_node_id_0 - S_node_id_1 <= value
    # base node is chosen as the node with smallest node_id
    node_ids = set()
    for (src_id, dst_id) in shift_constraints:
        node_ids.add(src_id)
        node_ids.add(dst_id)
    node_ids = sorted(list(node_ids))
    basis_id = node_ids[0]
    var_map = {}
    variables = [cp.Variable() for _ in range(len(node_ids) - 1)]
    for (var_id, node_id) in enumerate(node_ids[1:]):
        var_map[node_id] = variables[var_id]

    obj = cp.sum_squares(cp.hstack(variables))
    
    constraints = []
    for (src_id, dst_id), value in shift_constraints.items():
        if src_id == basis_id:
            constraints.append(var_map[dst_id] >= -value)
        elif dst_id == basis_id:
            constraints.append(var_map[src_id] <= value)
        else:
            constraints.append(var_map[src_id] - var_map[dst_id] <= value)
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    
    shift_map = {}
    if prob.status not in ["infeasible", "unbounded"]:
        shift_map[basis_id] = 0
        for node_id in node_ids[1:]:
            shift_map[node_id] = float(var_map[node_id].value)
        return basis_id, shift_map
    else:
        print("[BPS ALIGN] Problem is {}".format(prob.status))
        return basis_id, shift_map
        exit(-1)

class bytepsGraph:
    """ Helper class for processing the detailed communication trace of BytePS
    """
    def __init__(self):
        if args_.profile_start_step is None:
            self.PROFILE_ITER_START = 10
            SingleLogger().warn("[BYTEPS] profile_start_step UNSET. USING DEFAULT VALUE {}.".format(self.PROFILE_ITER_START))
        else:
            self.PROFILE_ITER_START = args_.profile_start_step + 1
            SingleLogger().info("[BYTEPS] Using profile_start_step = {}.".format(self.PROFILE_ITER_START-1))
        if args_.profile_duration is None:
            self.PROFILE_ITER_DURATION = 10
            SingleLogger().warn("[BYTEPS] profile_duration UNSET. USING DEFAULT VALUE {}.".format(self.PROFILE_ITER_DURATION))
        else:
            self.PROFILE_ITER_DURATION = args_.profile_duration
            SingleLogger().info("[BYTEPS] Using profile_duration = {}.".format(self.PROFILE_ITER_DURATION))
        self.tensor_part2server = {}
        self.pid_to_target = {}
        self.pid_to_server = {}
        self.comm_ops_dict = {}
        self.comp_ops_dict = {}
        self.comm_ops_stats = {}
        self.comp_ops_stats = {}
        self.comm_durations = {}
        self.comp_durations = {}
        self.servers_threads = {}
        self.partition_dict = {}
        self.tensor_id2grp = {"grp_names": [], "tensor2grpID": {}}
        self.bw_delays = None
        self.comp_ops_tid = {}
        self.workers = set()
        self.graph = nx.DiGraph()
        self.master_host_id = 0
        self.time_drift = None
        self.comm_delays = None
        self._manual_delay = 0
        self._ignored_tensors = set()
        self._inited = False

        self.grp_part_id2server = None

    def init(self, comm_trace_path, server_trace_path, van_type="ZMQ", align_trace=True):
        if van_type not in ["ZMQ", "RDMA"]:
            raise ArgumentError("Unknown van type: {}".format(van_type))
        self.van_type = van_type
        SingleLogger().info("Reading comm trace from {}".format(comm_trace_path))
        with open(comm_trace_path, "r") as f:
            comm_trace = json.load(f)
        SingleLogger().info("Reading server trace from {}".format(server_trace_path))
        with open(server_trace_path, "r") as f:
            server_trace = json.load(f)
        self.comm_trace_ = comm_trace
        self.server_trace_ = server_trace
        if isinstance(comm_trace, dict):
            self.comm_trace_content_ = comm_trace["traceEvents"]
        elif isinstance(comm_trace, list):
            self.comm_trace_content_ = comm_trace
        else:
            raise RuntimeError("Cannot parse BytePS comm trace.")
        if isinstance(server_trace, dict):
            self.server_trace_content_ = server_trace["traceEvents"]
        elif isinstance(server_trace, list):
            self.server_trace_content_ = server_trace
        else:
            raise RuntimeError("Cannot parse BytePS server trace.")
        # parse tensor assignment information
        self._parse_trace()
        self._build_comm_graph()
        if align_trace:
            if van_type == "ZMQ":
                self._align_traces_zmq()
            else:
                self._align_traces_rdma()
        self._calc_comm_delays()
        self._dump_cache()
        self._inited = True
    
    def init_from_cache(self, cache_path):
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        (self.tensor_part2server, 
         self.comm_durations, self.comp_durations,
         self.servers_threads, self.partition_dict,
         self.comp_ops_tid, self.workers, self.graph,
         self.master_host_id, self.time_drift, 
         self.comm_delays, self.bw_delays, self.van_type,
         self.pid_to_server, self.pid_to_target, self.tensor_id2grp) = state
        self._inited = True
    
    def _dump_cache(self, cache_path=None):
        if cache_path is None:
            cache_path = os.path.join(args_.path, "bps_cache.pickle")
        state = (self.tensor_part2server, 
                 self.comm_durations, self.comp_durations,
                 self.servers_threads, self.partition_dict,
                 self.comp_ops_tid, self.workers, self.graph,
                 self.master_host_id, self.time_drift, 
                 self.comm_delays, self.bw_delays, self.van_type,
                 self.pid_to_server, self.pid_to_target, self.tensor_id2grp)
        with open(cache_path, "wb") as f:
            pickle.dump(state, f)

    def _check_inited(self):
        if self._inited == False:
            raise RuntimeError("Must init byteps graph before use.")
    
    def get_master_host_id(self):
        return self.master_host_id

    def _add_prefix(self, device_name, tensor_name):
        return device_name + DEL + tensor_name
    
    def _add_suffix(self, event_name, suffix):
        return event_name + DDEL + suffix

    def get_num_servers(self):
        return len(self.servers_threads)

    def get_servers(self):
        return list(self.servers_threads.keys())
    
    def get_num_workers(self):
        return len(self.workers)

    def get_workers(self):
        return list(self.workers)
    
    def _parse_source_target(self, process_name):
        source, target = [s.strip() for s in process_name.split("->")]
        return source, target

    def _parse_event_name(self, event_name):
        split = event_name.split(".")
        if len(split) == 2:
            tensor_name, op = split
        else:
            _, tensor_name, op = split
        return tensor_name, op
    
    def _parse_partition(self, tensor_name):
        if PART_DEL in tensor_name:
            tensor_name_wo_part, part_id = self._parse_partition_name(tensor_name)
            if tensor_name_wo_part in self.partition_dict:
                self.partition_dict[tensor_name_wo_part].add(part_id)
            else:
                self.partition_dict[tensor_name_wo_part] = set([part_id])
        else:
            tensor_name_wo_part, part_id = tensor_name, '0'
        if tensor_name_wo_part not in self.tensor_id2grp["grp_names"]:
            for each_tensor_name in tensor_name_wo_part.split("+"):
                self.tensor_id2grp["tensor2grpID"][each_tensor_name] = len(self.tensor_id2grp["grp_names"])
            self.tensor_id2grp["grp_names"].append(tensor_name_wo_part)
        return tensor_name_wo_part, part_id
    
    def parse_tensor_grp_name(self, tensor_name):
        return self.tensor_id2grp["grp_names"][self.tensor_id2grp["tensor2grpID"][tensor_name]]

    def _gen_partitioned_name(self, tensor_name, part_id):
        return tensor_name + PART_DEL + str(part_id)
    
    def _parse_partition_name(self, tensor_name_w_partition):
        '''
        tensor_name_w_partition is in the format of 
            <tensor_name>PART_DEL<part_id>
        '''
        return tensor_name_w_partition.split(PART_DEL)

    def _parse_trace(self):
        # server trace
        for event in self.server_trace_content_:
            if event["ph"] == "M":
                if event["name"] == "process_name":
                    server = event["args"]["name"].strip()
                    self.pid_to_server[event["pid"]] = server

        for event in self.server_trace_content_:
            if event["ph"] != "M":
                server = self.pid_to_server[event["pid"]]
                if server not in self.servers_threads:
                    self.servers_threads[server] = 0
                tensor_name, sub_op = self._parse_event_name(event["name"])
                tensor_name, part_id = self._parse_partition(tensor_name)
                tensor_name_w_part = self._gen_partitioned_name(tensor_name, part_id)
                if tensor_name_w_part in self.tensor_part2server:
                    assert self.tensor_part2server[tensor_name_w_part] == server
                else:
                    self.tensor_part2server[tensor_name_w_part] = server
                self.servers_threads[server] = max(self.servers_threads[server], event["tid"])
                _key = (server, tensor_name, sub_op, part_id)
                if _key not in self.comp_ops_tid:
                    self.comp_ops_tid[_key] = event["tid"]
                _key = (server, tensor_name, sub_op, event["tid"], part_id)
                if _key not in self.comp_ops_dict:
                    self.comp_ops_dict[_key] = []
                self.comp_ops_dict[_key].append((event["ph"], event["ts"]))

        # COMM trace
        for event in self.comm_trace_content_:
            if event["ph"] == "M":
                if event["name"] == "process_name":
                    process_name = event["args"]["name"]
                    source, target = self._parse_source_target(process_name)
                    self.pid_to_target[event["pid"]] = (source, target)
        len_count = {}
        
        for event in self.comm_trace_content_:
            if event["ph"] != "M":
                source, target = self.pid_to_target[event["pid"]]
                tensor_name, sub_op = self._parse_event_name(event["name"])
                tensor_name, part_id = self._parse_partition(tensor_name)
                tensor_name_w_part = self._gen_partitioned_name(tensor_name, part_id)
                if "server" in source:
                    self.workers.add(target)
                    sub_op += "_RES"
                    if tensor_name_w_part in self.tensor_part2server:
                        if self.tensor_part2server[tensor_name_w_part] != source:
                            continue
                else:
                    self.workers.add(source)
                    sub_op += "_REQ"
                    if tensor_name_w_part in self.tensor_part2server:
                        if self.tensor_part2server[tensor_name_w_part] != target:
                            continue  
                _key = (source, target, tensor_name, sub_op, part_id)
                if _key not in self.comm_ops_dict:
                    self.comm_ops_dict[_key] = []
                self.comm_ops_dict[_key].append((event["ph"], event["ts"]))
            
        for key, events in self.comm_ops_dict.items():
            # durations = []
            start_ts = []
            end_ts = []
            for ev in events:
                if ev[0] == "B":
                    start_ts.append(ev[1])
                elif ev[0] == "E":
                    end_ts.append(ev[1])
                else:
                    raise RuntimeError("Cannot parse event ph: ".format(ev[0]))
            start_ts = sorted(start_ts)
            end_ts = sorted(end_ts)
            if key[3] == "PUSH_REQ" or key[3] == "PUSH_RES":
                start_ts = start_ts[self.PROFILE_ITER_START + 1:self.PROFILE_ITER_START+ 1 + self.PROFILE_ITER_DURATION]
                end_ts = end_ts[self.PROFILE_ITER_START + 1:self.PROFILE_ITER_START+ 1 + self.PROFILE_ITER_DURATION]
            else:
                start_ts = start_ts[self.PROFILE_ITER_START:self.PROFILE_ITER_START+self.PROFILE_ITER_DURATION]
                end_ts = end_ts[self.PROFILE_ITER_START:self.PROFILE_ITER_START+self.PROFILE_ITER_DURATION]

            assert len(start_ts) == len(end_ts)
            # if key[0] == "server_2" and key[1] == "worker_0" and key[2] == "DistributedGradientDescentOptimizer_Push_Pull/BytePSPushPull_gradients_resnet50_conv2_block2_1_conv_BiasAdd_grad_tuple_control_dependency_1_0" and key[-1] == "PULL_RES":
            #     import code
            #     code.interact(local=locals())
            durations = list(zip(start_ts, end_ts))
            # last_start = -1
            # for index, ev in enumerate(events):
            #     if ev[0] == "B":
            #         last_start = ev[1]
            #     else:
            #         if last_start != -1:
            #             durations.append((last_start, ev[1]))
            #             last_start = -1
            if key not in self.comm_durations:
                self.comm_durations[key] = {}
            self.comm_durations[key] = durations
        
        for key, events in self.comp_ops_dict.items():
            durations = []
            start_ts = []
            end_ts = []

            for ev in events:
                if ev[0] == "B":
                    start_ts.append(ev[1])
                elif ev[0] == "E":
                    end_ts.append(ev[1])
                else:
                    raise RuntimeError("Cannot parse event ph: ".format(ev[0]))
            assert len(start_ts) == len(end_ts)
            durations = list(zip(start_ts, end_ts))
            # last_start = -1
            # for index, ev in enumerate(events):
            #     if ev[0] == "B":
            #         last_start = ev[1]
            #     else:
            #         if last_start != -1:
            #             durations.append((last_start, ev[1]))
            #             last_start = -1
            if key not in self.comp_durations:
                self.comp_durations[key] = {}
            self.comp_durations[key] = durations

        for key, durations in self.comm_durations.items():
            sub_op = key[3]
            if sub_op == PS_COMM_OPS.PUSH_REQ or sub_op == PS_COMM_OPS.PUSH_RES:
                my_count = len(durations) - 1
            else:
                my_count = len(durations)
            if my_count not in len_count:
                len_count[my_count] = 0
            len_count[my_count] += 1

        mode_len_count = -1
        mode_len = 0
        for key, count in len_count.items():
            if count > mode_len_count:
                mode_len = key
                mode_len_count = count

        for key, durations in self.comm_durations.items():
            sub_op = key[3]
            if sub_op == PS_COMM_OPS.PUSH_REQ or sub_op == PS_COMM_OPS.PUSH_RES:
                # if len(durations) != mode_len + 1:
                if abs(len(durations) - mode_len) > 2:
                    self._ignored_tensors.add(key)
                # chopped_durations = durations[self.PROFILE_ITER_START + 1:self.PROFILE_ITER_START+ 1 + self.PROFILE_ITER_DURATION]
            else:
                # if len(durations) != mode_len:
                if abs(len(durations) - mode_len) > 1:
                    self._ignored_tensors.add(key)
                # chopped_durations = durations[self.PROFILE_ITER_START:self.PROFILE_ITER_START+self.PROFILE_ITER_DURATION]
            # self.comm_durations[key] = chopped_durations
            self.comm_durations[key] = durations

        for key, durations in self.comp_durations.items():
            if len(durations) != mode_len:
                self._ignored_tensors.add(key)
            chopped_durations = durations[self.PROFILE_ITER_START:self.PROFILE_ITER_START+self.PROFILE_ITER_DURATION]
            self.comp_durations[key] = chopped_durations

    def _calc_stats(self):
        self._check_inited()
        comm_avg_std = []
        for key, durations in self.comm_durations.items():
            self.comm_ops_stats[key] = (np.average(durations), np.std(durations) / np.average(durations))
            print(key, self.comm_ops_stats[key])
            comm_avg_std.append(np.std(durations) / np.average(durations))
        print("\nComm Avg:")
        print(np.average(comm_avg_std))
        print()

        comp_avg_std = []
        for key, durations in self.comp_durations.items():
            self.comp_ops_stats[key] = (np.average(durations), np.std(durations) / np.average(durations))
            print(key, self.comp_ops_stats[key])
            comp_avg_std.append(np.std(durations) / np.average(durations))
        print("\nComp Avg:")
        print(np.average(comp_avg_std))

    def gen_comm_device_name(self, comm_key):
        # comm_key is in format (source, target, tensor_name, op, part_id)
        source, target, _, _, _ = comm_key
        return source + COMM_DEL + target

    def gen_comm_event_name(self, comm_key):
        # comm_key is in format (source, target, tensor_name, op, part_id)
        _, _, tensor_name, sub_op, part_id = comm_key
        std_name = "Comm." + tensor_name + "." + sub_op
        return self._add_suffix(std_name, self.gen_comm_device_name(comm_key) + COMM_DEL + part_id) 

    def gen_comm_full_name(self, comm_key):
        return self._add_prefix(self.gen_comm_device_name(comm_key), self.gen_comm_event_name(comm_key))

    def parse_comm_event_name(self, comm_event_name):
        # comm_name, op = comm_event_name.split(".")
        tensor_name, sub_op, suffix = parse_allinfo_from_name_v2(comm_event_name)
        source, target, part_id = self._parse_comm_name(suffix)
        return source, target, tensor_name, sub_op, part_id

    def _parse_comm_name(self, comm_name_suffix):
        source, target, part_id = comm_name_suffix.split(COMM_DEL)
        return source, target, part_id

    def gen_comm_unique_pid(self, comm_key):
        # comm_key is in format (source, target, tensor_name, op, part_id)
        # return self._str_hashing(self.gen_comm_device_name(comm_key))
        return self.gen_comm_device_name(comm_key)
    
    def gen_comp_device_name(self, comp_key):
        # comp_key is in format (server, tensor_name, op, tid)
        server, _, _, tid, _ = comp_key
        return server + "_t" + str(tid)

    def gen_comp_event_name(self, comp_key, sum_index = None):
        # comp_key is in format (server, tensor_name, op, tid, part_id)
        _, tensor_name, sub_op, _, part_id = comp_key
        std_name = "Comm." + tensor_name + "." + sub_op
        if sum_index is None:
            return self._add_suffix(std_name, self.gen_comp_device_name(comp_key) + COMM_DEL + PART_DEL + part_id) 
        else:
            return self._add_suffix(std_name, self.gen_comp_device_name(comp_key) + COMM_DEL + str(sum_index) + COMM_DEL + PART_DEL + part_id)
    
    def gen_comp_full_name(self, comp_key, sum_index=None):
        return self._add_prefix(self.gen_comp_device_name(comp_key), self.gen_comp_event_name(comp_key, sum_index=sum_index))
    
    def parse_comp_name(self, comp_name):
        tensor_name, sub_op, suffix = parse_allinfo_from_name_v2(comp_name)
        suffix_splits = suffix.split(COMM_DEL)
        if len(suffix_splits) == 3:
            server_tid, sum_index, part_id = suffix_splits
        elif len(suffix_splits) == 2:
            server_tid, part_id = suffix_splits
            sum_index = None
        server_id, tid = server_tid.split("_t")
        part_id = part_id.split(PART_DEL)[1]
        return server_id, tid, tensor_name, sub_op, part_id, sum_index
    
    def gen_comp_unique_pid(self, comp_key):
        # comp_key is in format (server, tensor_name, op, tid)
        # return self._str_hashing(self.gen_comp_device_name(comp_key))
        return self.gen_comp_device_name(comp_key)
    
    def _str_hashing(self, s):
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

    def _get_node_from_dev_name(self, s):
        return int(s.split("_")[-1])
    
    def _apply_shift_on_trace(self, master_node, trace_shifts, van_type="ZMQ"):
        SingleLogger().info("# Aligning BPS traces")
        SingleLogger().info("Aligning time based on node {}".format(master_node))
        # for key, item in send_delays.items():
        for key, item in trace_shifts.items():
            SingleLogger().info("Shifting traces of node {} {} by {} us.".format(key, "forward" if item >= 0 else "backward", np.abs(item)))
        
        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            if van_type == "ZMQ":
                if target in trace_shifts:
                    delay = trace_shifts[target]
                    new_durations = []
                    for st, ed in durations:
                        new_durations.append((st+delay, ed+delay))
                    self.comm_durations[key] = new_durations
            else:
                source_delay = 0
                target_delay = 0
                if source in trace_shifts:
                    source_delay = trace_shifts[source]
                if target in trace_shifts:
                    target_delay = trace_shifts[target]
                new_durations = []
                for st, ed in durations:
                    assert st+source_delay <= ed+target_delay
                    new_durations.append((st+source_delay, ed+target_delay))
                self.comm_durations[key] = new_durations

        for key, durations in self.comp_durations.items():
            server, tensor_name, op, tid, part_id = key
            if self._get_node_from_dev_name(server) in trace_shifts:
                delay = trace_shifts[self._get_node_from_dev_name(server)]
                new_durations = []
                for st, ed in durations:
                    new_durations.append((st+delay, ed+delay))
                self.comp_durations[key] = new_durations
        
        self.time_drift = trace_shifts
        self.master_host_id = master_node
    
    def _align_traces_zmq(self):
        # comm_key is in format (source, target, tensor_name, op, part_id)
        worker_ranks = set()
        source_ranks = set()
        server_ranks = set()
        durations_dict = {}
        unique_tensors = set()
        # push_req_ops = {}
        intervals = {}

        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            unique_tensors.add(tensor_name)
            source_rank = self._get_node_from_dev_name(source)
            if source.startswith("server"):
                server_ranks.add(source)
            else:
                worker_ranks.add(source)
            source_ranks.add(source)

            if source not in durations_dict:
                durations_dict[source] = {}
            if (source, target) not in intervals:
                intervals[(source, target)] = IntervalTree()
            for st, ed in durations:
                if st != ed:
                    intervals[(source, target)][st:ed] = True
            durations_dict[source][key] = durations

        send_delays = {}
        send_delay_keys = {}

        for r0 in server_ranks:
            for r1 in worker_ranks:
                send_delays[(r0, r1)] = float('inf')
                send_delays[(r1, r0)] = float('inf')

        trace_shifts = {}
        for source_rank, key_dict in durations_dict.items():
            for key, durations in key_dict.items():
                source, target, tensor_name, op, part_id = key
                if target.startswith("worker") and op == PS_COMM_OPS.PUSH_RES:
                    # an op that can be used to align traces
                    if key in self._ignored_tensors:
                        SingleLogger().warn(
                                "[BPS ALIGN]: Length mismatch between {} and {} on tensor {}".format(
                                    source, target, tensor_name))
                        continue
                    # target_node_id = self._get_node_from_dev_name(target)
                    if source_rank == target:
                        continue
                    for index in range(len(durations)):
                        # find the corresponding push_req
                        push_res_st, push_res_ed = durations[index]
                        _, push_req_ed = durations_dict[target] \
                                                    [(target, source, tensor_name, PS_COMM_OPS.PUSH_REQ, part_id)] \
                                                    [index]
                        # if push_response not queued on server->worker
                        if not intervals[(source, target)].overlap(push_res_st-500, push_res_st-1):
                            if push_res_st - push_req_ed < send_delays[(source_rank, target)]:
                                send_delay_keys[(source_rank, target)] = (key, index)
                                send_delays[(source_rank, target)] = push_res_st - push_req_ed
                        # find the corresponding pull_req
                        pull_req_st, pull_req_ed = durations_dict[target] \
                                                    [(target, source, tensor_name, PS_COMM_OPS.PULL_REQ, part_id)] \
                                                    [index]
                        # if pull_req not queued on worker->server
                        if not intervals[(target, source)].overlap(pull_req_st - 500, pull_req_st-1):
                            if pull_req_st - push_res_ed < send_delays[(target, source_rank)]:
                                send_delay_keys[(target, source_rank)] = (key, index)
                                send_delays[(target, source_rank)] = pull_req_st - push_res_ed

        master_node, trace_shifts = optimize_time_shift(send_delays)

        self._apply_shift_on_trace(master_node, trace_shifts, van_type="ZMQ")

    def _align_traces_rdma(self):
        # comm_key is in format (source, target, tensor_name, op)
        worker_ranks = set()
        server_ranks = set()
        unique_tensors = set()
        
        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            unique_tensors.add(tensor_name)
            # source_rank = self._get_node_from_dev_name(source)
            if source.startswith("server"):
                server_ranks.add(source)
            else:
                worker_ranks.add(source)
        
        node_delays = {}

        for r0 in server_ranks:
            for r1 in worker_ranks:
                node_delays[(r0, r1)] = float('inf')
                node_delays[(r1, r0)] = float('inf')

        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            for (st, ed) in durations:
                node_delays[(source, target)] = min(node_delays[(source, target)], ed - st)

        master_node, trace_shifts = optimize_time_shift(node_delays)
        self._apply_shift_on_trace(master_node, trace_shifts, van_type="RDMA")

    def _calc_comm_delays(self):
        intervals = {}
        network_delays = {}

        push_req_ops = {}
        push_res_ops = {}
        pull_req_ops = {}

        # copy_first_ops = {}
        # sum_ops = {}

        source_ranks = set()

        unique_tensors = set()

        min_start_time = float('inf')
        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            if (source, target) not in intervals:
                intervals[(source, target)] = IntervalTree()
            unique_tensors.add(tensor_name)
            source_rank = self._get_node_from_dev_name(source)
            source_ranks.add(source_rank)
            for st, ed in durations:
                min_start_time = min(min_start_time, st)
                if st != ed:
                    if st > ed:
                        # SingleLogger().warn("Tensor {}::{}->{}_{}, st - ed = {}".format(
                        #     source, target, tensor_name, op, st - ed))
                        continue
                    intervals[(source, target)][st:ed] = (tensor_name, op)
            if op == PS_COMM_OPS.PUSH_REQ:
                if (source, target) not in push_req_ops:
                    push_req_ops[(source, target)] = {}
                push_req_ops[(source, target)][key] = durations
            elif op == PS_COMM_OPS.PUSH_RES:
                if (source, target) not in push_res_ops:
                    push_res_ops[(source, target)] = {}
                push_res_ops[(source, target)][key] = durations
            elif op == PS_COMM_OPS.PULL_REQ:
                if (source, target) not in pull_req_ops:
                    pull_req_ops[(source, target)] = {}
                pull_req_ops[(source, target)][key] = durations

        def _is_first_push(key, index):
            source, target, tensor_name, _ = key
            min_ts = []
            my_end = push_req_ops[(source, target)][key][index][1]
            for (s, t) in push_req_ops.keys():
                for key, durations in push_req_ops[(s, t)].items():
                    try:
                        if durations[index][1] < my_end:
                            return False
                    except:
                        return False
            return True

        for (source, target) in push_req_ops.keys():
            # source: worker, target: server
            # for each push req, get delay for its corresponding copy first or sum
            for key, durations in push_req_ops[(source, target)].items():
                _, _, tensor_name, _, part_id = key
                if key in self._ignored_tensors:
                    continue
                # copy_first_op_durations, tid_cp = copy_first_ops[target][tensor_name]
                # sum_op_durations, tid_sum = sum_ops[target][tensor_name]
                push_res_op_durations = push_res_ops[(target, source)][(target, source, tensor_name, PS_COMM_OPS.PUSH_RES, part_id)]
                for index, (st, ed) in enumerate(durations):
                    pres_st, pres_ed = push_res_op_durations[index]
                    # if not queued in server->worker communication queue
                    if not intervals[(target, source)].overlap(pres_st - 500, pres_st - 5):
                        latency = pres_st - ed
                        # add to worker -> server network delay
                        if (source, target) not in network_delays:
                            network_delays[(source, target)] = []
                        network_delays[(source, target)].append(latency)

        for (source, target) in push_res_ops.keys():
            # source: server, target: worker
            for key, durations in push_res_ops[(source, target)].items():
                _, _, tensor_name, _, part_id = key
                if key in self._ignored_tensors:
                    continue
                pull_req_op_durations = pull_req_ops[(target, source)][(target, source, tensor_name, PS_COMM_OPS.PULL_REQ, part_id)]
                for index, (st, ed) in enumerate(durations):
                    # get pull request
                    rq_st, rq_ed = pull_req_op_durations[index]
                    # if not queued in worker -> server communication queue
                    if not intervals[(target, source)].overlap(rq_st - 500, rq_st - 5):
                        latency = rq_st - ed
                        # add to server -> worker network delay
                        if (source, target) not in network_delays:
                            network_delays[(source, target)] = []
                        network_delays[(source, target)].append(latency)

        for key, items in network_delays.items():
            if items:
                avg_delay = np.average(items)
                network_delays[key] = avg_delay
                SingleLogger().info("Comm delay for {} is {} us.".format(key, avg_delay))
            else:
                network_delays[key] = 0
                SingleLogger().info("Cannot determine comm delay for {}.".format(key))

        self.comm_delays = network_delays

    def gen_compatible_trace(self, dump_path=None):
        self._check_inited()
        trace = []
        comm_pids = set()
        for key, durations in self.comm_durations.items():
            for index, (st, ed) in enumerate(durations):
                json_event = {}
                json_event["name"] = self.gen_comm_event_name(key)
                json_event["ph"] = "X"
                json_event["ts"] = st
                comm_uid = self.gen_comm_unique_pid(key)
                comm_pids.add(comm_uid)
                json_event["pid"] = comm_uid
                json_event["tid"] = 0
                json_event["dur"] = ed - st
                json_event["cat"] = "Comm"
                json_event["args"] = {}
                trace.append(json_event)

        for key, durations in self.comp_durations.items():
            _, _, op, _, part_id = key
            for index, (st, ed) in enumerate(durations):
                if op == PS_COMP_OPS.SUM:
                    for i in range(len(self.workers)-1):
                        json_event = {}
                        json_event["name"] = self.gen_comp_event_name(key, i)
                        json_event["ph"] = "X"
                        json_event["ts"] = st
                        json_event["pid"] = self.gen_comp_unique_pid(key)
                        json_event["tid"] = 0
                        json_event["dur"] = ed - st
                        json_event["cat"] = CatName.PS_SERVER_OPERATOR.value
                        json_event["args"] = {}
                        trace.append(json_event)
                else:
                    json_event = {}
                    json_event["name"] = self.gen_comp_event_name(key)
                    json_event["ph"] = "X"
                    json_event["ts"] = st
                    json_event["pid"] = self.gen_comp_unique_pid(key)
                    json_event["tid"] = 0
                    json_event["dur"] = ed - st
                    json_event["cat"] = CatName.PS_SERVER_OPERATOR.value
                    json_event["args"] = {}
                    trace.append(json_event)
        
        # clip comm events for RDMA
        if self.van_type == "RDMA":
            pid2events = {}
            for ev in trace:
                if ev["pid"] in comm_pids:
                    if ev["pid"] not in pid2events:
                        pid2events[ev["pid"]] = []
                    pid2events[ev["pid"]].append(ev)
            for pid, events in pid2events.items():
                sorted_events = sorted(events, key=lambda x: x["ts"]+x["dur"])
                for idx, ev in enumerate(sorted_events):
                    if idx == 0:
                        continue
                    ev_end_time = ev["ts"] + ev["dur"]
                    prev_ev = sorted_events[idx-1]
                    if ev["ts"] < prev_ev["ts"] + prev_ev["dur"]:
                        ev["ts"] = prev_ev["ts"] + prev_ev["dur"]
                        ev["dur"] = ev_end_time - ev["ts"]

        if dump_path is not None:
            try:
                with open(dump_path, "w") as f:
                    json.dump(trace, f)
                SingleLogger().info("Aligned BPS trace dumped to {}".format(dump_path))
            except:
                SingleLogger().warn("Cannot open path {}. Aligned trace not dumped.".format(dump_path))

        return trace
    
    def get_push_req_node(self, source_id, tensor_name):
        self._check_inited()
        partitions = self.partition_dict.get(tensor_name, ['0'])
        full_names = []
        for part_id in partitions:
            # (source, target, tensor_name, op, partition_id)
            partitioned_name = self._gen_partitioned_name(tensor_name, part_id)
            key = ("worker_"+str(source_id), self.tensor_part2server[partitioned_name], 
                tensor_name, PS_COMM_OPS.PUSH_REQ, part_id)
            full_names.append(self.gen_comm_full_name(key))
        return full_names

    def get_pull_res_node(self, source_id, tensor_name):
        self._check_inited()
        partitions = self.partition_dict.get(tensor_name, ['0'])
        full_names = []
        for part_id in partitions:
            # (source, target, tensor_name, op, partition_id)
            partitioned_name = self._gen_partitioned_name(tensor_name, part_id)
            key = (self.tensor_part2server[partitioned_name], "worker_"+str(source_id),
                tensor_name, PS_COMM_OPS.PULL_RES, part_id)
            full_names.append(self.gen_comm_full_name(key))
        return full_names

    def _build_comm_graph(self):
        edges_to_add = []
        for tensor_name_w_part, assigned_server in self.tensor_part2server.items():
            tensor_name, part_id = self._parse_partition_name(tensor_name_w_part)

            # push req -> push res -> pull req
            for worker_name in self.workers:
                edges_to_add.append(
                    (self.gen_comm_full_name(
                        (worker_name, assigned_server, tensor_name, PS_COMM_OPS.PUSH_REQ, part_id)
                    ),
                    self.gen_comm_full_name(
                        (assigned_server, worker_name, tensor_name, PS_COMM_OPS.PUSH_RES, part_id)
                    ))
                )
                edges_to_add.append(
                    (self.gen_comm_full_name(
                        (assigned_server, worker_name, tensor_name, PS_COMM_OPS.PUSH_RES, part_id)
                    ),
                    self.gen_comm_full_name(
                        (worker_name, assigned_server, tensor_name, PS_COMM_OPS.PULL_REQ, part_id)
                    ))
                )

            # copy_first -> sum * N-1 -> copy_merged
            edges_to_add.append(
                (self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    PS_COMP_OPS.COPY_FIRST, 
                    self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.COPY_FIRST, part_id)],
                    part_id)
                    ),
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    PS_COMP_OPS.SUM, 
                    self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.SUM, part_id)],
                    part_id),
                    sum_index=0
                    ))
            )
            for i in range(1, len(self.workers)-1):
                edges_to_add.append(
                    (self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        PS_COMP_OPS.SUM, 
                        self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.SUM, part_id)],
                        part_id),
                        sum_index=i-1
                        ),
                    self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        PS_COMP_OPS.SUM, 
                        self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.SUM, part_id)],
                        part_id),
                        sum_index=i
                        ))
                )
            edges_to_add.append(
                (self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    PS_COMP_OPS.SUM, 
                    self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.SUM, part_id)],
                    part_id),
                    sum_index=len(self.workers)-2
                    ),
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    PS_COMP_OPS.COPY_MERGED, 
                    self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.COPY_MERGED, part_id)],
                    part_id)
                    ))
            )

            ### NOTE: there should be some edges from byteps comm to byteps comp
            #   DUE to the connectivity is dynamic, depending on the arriving order
            #   of workers' push reqesuts.

            # pull req -> pull res, copy_merged -> pull_res
            for worker_name in self.workers:
                edges_to_add.append(
                    (self.gen_comm_full_name((worker_name, assigned_server, tensor_name, PS_COMM_OPS.PULL_REQ, part_id)),
                    self.gen_comm_full_name((assigned_server, worker_name, tensor_name, PS_COMM_OPS.PULL_RES, part_id)))
                )
                edges_to_add.append(
                    (self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        PS_COMP_OPS.COPY_MERGED, 
                        self.comp_ops_tid[(assigned_server, tensor_name, PS_COMP_OPS.COPY_MERGED, part_id)],
                        part_id)
                        ),
                    self.gen_comm_full_name((assigned_server, worker_name, tensor_name, PS_COMM_OPS.PULL_RES, part_id)))
                )
        
        self.graph.add_edges_from(edges_to_add)
        ### check the dag
        
    def is_server_comp(self, name):
        for op in [PS_COMP_OPS.COPY_FIRST, PS_COMP_OPS.SUM]:
            if op in name:
                return True
        return False

    def get_comm_graph(self):
        self._check_inited()
        return self.graph
    
    def calc_bw_to_comm_delay(self, comp_traces, dag):
        self._check_inited()
        bw_durations = {}
        for event in comp_traces:
            if event["ph"] == "X" and "BW" in event["name"]:
                tensor_name = event["name"]
                process_name = event["pid"]
                if process_name not in bw_durations:
                    bw_durations[process_name] = {}
                if tensor_name not in bw_durations[process_name]:
                    bw_durations[process_name][tensor_name] = []
                bw_durations[process_name][tensor_name].append((event["ts"], event["ts"] + event["dur"]))
        
        push_req_ops = {}
        interval = {}

        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op, part_id = key
            if op == PS_COMM_OPS.PUSH_REQ:
                for st, ed in durations:
                    if source not in interval:
                        interval[source] = IntervalTree()
                    if st != ed:
                        if st > ed:
                            continue
                        interval[source][st:ed] = tensor_name
                if source not in push_req_ops:
                    push_req_ops[source] = {}
                push_req_ops[source][tensor_name] = durations

        bw_delay_dict = {}

        def get_tensor_name_from_full_name(n):
            _, _, tensor_name, _, _ = self.parse_comm_event_name(parse_rawname(n))
            return tensor_name

        for process_name, tensor_dict in bw_durations.items():
            bw_delay_for_its = [ [] for _ in range(self.PROFILE_ITER_DURATION)]
            node_rank = process_name.split(".")[0].split("_")[-1]
            local_rank = process_name.split(".")[1].split("rank")[-1]
            for layer_name, evs in tensor_dict.items():
                # if len(bw_delay_for_its) < len(evs):
                #     for _ in range(len(evs)-len(bw_delay_for_its)):
                #         bw_delay_for_its.append([])
                long_name = gen_long_name(process_name, layer_name)
                if long_name in dag:
                    # push_tensor_names = [n.split(".")[-1] for n in dag.neighbors(long_name) if "Comm" in n]
                    push_tensor_names = [get_tensor_name_from_full_name(n) for n in dag.neighbors(long_name) if "PUSH_REQ" in n]
                    matched_tensor_names = []
                    for tensor_name in push_req_ops["worker_"+node_rank].keys():
                        if tensor_name in push_tensor_names:
                            matched_tensor_names.append(tensor_name)
                    for index in range(len(evs)):
                        # local_min_delay = float('inf')
                        for tensor_name in matched_tensor_names:
                            tensor_durations = push_req_ops["worker_"+node_rank][tensor_name]
                            if len(evs) != len(tensor_durations) or len(evs) > self.PROFILE_ITER_DURATION:
                                continue
                            bw_st, bw_ed = evs[index]
                            pu_st, pu_ed = tensor_durations[index]
                            if not interval["worker_"+node_rank].overlap(bw_ed - 200, bw_ed + 200) and not interval["worker_"+node_rank].overlap(pu_st - 200, pu_st - 5):
                                # local_min_delay = min(local_min_delay, pu_st - bw_ed)
                                bw_delay_for_its[index].append(pu_st - bw_ed)
                        # if node_rank not in bw_delay_dict:
                        #     bw_delay_dict[node_rank] = {}
                        # if local_rank not in bw_delay_dict[node_rank]:
                        #     bw_delay_dict[node_rank][local_rank] = []
                        # if local_min_delay != float('inf'):
                        #     bw_delay_dict[node_rank][local_rank].append(local_min_delay)
                        # else:
                        #     bw_delay_dict[node_rank][local_rank].append(None)
                else:
                    SingleLogger().warn("BYTEPS BW Delay: {} not in dag.".format(long_name))
            bw_avg_for_its = []
            for delays in bw_delay_for_its:
                if delays:
                    bw_avg_for_its.append(np.average(delays))
                else:
                    bw_avg_for_its.append(None)
            if node_rank not in bw_delay_dict:
                bw_delay_dict[node_rank] = {}
            if local_rank not in bw_delay_dict[node_rank]:
                bw_delay_dict[node_rank][local_rank] = []
            bw_delay_dict[node_rank][local_rank] = bw_avg_for_its

        node_delay_dict = {}
        for node_rank, local_dict in bw_delay_dict.items():
            min_avg = float("inf")
            local_mins = []
            num_iters = len(local_dict[list(local_dict.keys())[0]])
            for idx in range(num_iters):
                local_min = float("inf")
                for local_rank, delays in local_dict.items():
                    if delays[idx] is not None:
                        local_min = min(local_min, delays[idx])
                if not np.isnan(local_min) and not local_min == float("inf") and not local_min < 0:
                    local_mins.append(local_min)
            min_avg = np.average(local_mins)
            if np.isnan(min_avg) or min_avg == float("inf") or min_avg < 0:
                SingleLogger().warn("Cannot determine the BW delay of {}.".format("worker_"+node_rank))
                node_delay_dict["worker_"+node_rank] = 0
            else:
                SingleLogger().info("BW delay of {} is {} us.".format("worker_"+node_rank, min_avg))
                node_delay_dict["worker_"+node_rank] = min_avg

        self.bw_delays = node_delay_dict

        return node_delay_dict
