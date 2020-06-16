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
from byteps.preprocess import preprocess_pcap, parse_server_logs

args_ = arg_utils.SingleArg().args

class COMM_OPS(object):
    PUSH_REQ = "PUSH_REQ"
    PULL_REQ = "PULL_REQ"
    PUSH_RES = "PUSH_RES"
    PULL_RES = "PULL_RES"

class COMP_OPS(object):
    COPY_FIRST = "COPY_FIRST"
    SUM = "SUM"
    COPY_MERGED = "COPY_MERGED"

COMM_DEL = "::"

comm_ops = set([COMM_OPS.PUSH_REQ, COMM_OPS.PULL_REQ, COMM_OPS.PUSH_RES, COMM_OPS.PULL_RES])

comp_ops = set([COMP_OPS.COPY_FIRST, COMP_OPS.SUM, COMP_OPS.COPY_MERGED])

class ServerOpCounter(object):
	def __init__(self, byteps_graph):
		self.byteps_graph = byteps_graph
		self.num_workers = byteps_graph.get_num_workers()
		self.num_servers = byteps_graph.get_num_servers()
		self.server_op_counter = {}
		for server in byteps_graph.get_servers():
			self.server_op_counter[server] = {}
	
	def get_next_op(self, name):
		raw_name = parse_rawname_from_name(name)
		source, target, tensor_name, op = self.byteps_graph.parse_comm_event_name(raw_name)
		if target in self.server_op_counter:
			# is a push or pull request to servers
			if op == COMM_OPS.PUSH_REQ:
				if tensor_name not in self.server_op_counter[target]:
					self.server_op_counter[target][tensor_name] = 0
				if self.server_op_counter[target][tensor_name] == 0:
					# first push, a copy_first op
					comp_key = (target, tensor_name, COMP_OPS.COPY_FIRST, self.byteps_graph.comp_ops_tid[(target, tensor_name, COMP_OPS.COPY_FIRST)])
					res_name = self.byteps_graph.gen_comp_full_name(comp_key)
				else:
					comp_key = (target, tensor_name, COMP_OPS.SUM, self.byteps_graph.comp_ops_tid[(target, tensor_name, COMP_OPS.SUM)])
					res_name = self.byteps_graph.gen_comp_full_name(comp_key, sum_index=self.server_op_counter[target][tensor_name]-1)
				self.server_op_counter[target][tensor_name] += 1
				self.server_op_counter[target][tensor_name] = self.server_op_counter[target][tensor_name] % self.num_workers
			else:
				res_name = None
		else:
			res_name = None
		return res_name

class bytepsGraph:
    """ Helper class for processing the detailed communication trace of BytePS
    """
    def __init__(self):
        if args_.profile_start_step is None:
            self.PROFILE_ITER_START = 10
            SingleLogger().warn("[BYTEPS] profile_start_step UNSET. USING DEFAULT VALUE {}.".format(self.PROFILE_ITER_START))
        else:
            self.PROFILE_ITER_START = args_.profile_start_step
            SingleLogger().info("[BYTEPS] Using profile_start_step = {}.".format(self.PROFILE_ITER_START))
        if args_.profile_duration is None:
            self.PROFILE_ITER_DURATION = 11
            SingleLogger().warn("[BYTEPS] profile_duration UNSET. USING DEFAULT VALUE {}.".format(self.PROFILE_ITER_DURATION))
        else:
            self.PROFILE_ITER_DURATION = args_.profile_duration
            SingleLogger().info("[BYTEPS] Using profile_duration = {}.".format(self.PROFILE_ITER_DURATION))
        self.gradient_assignment_dict = {}
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

    def init(self, comm_trace_path, server_trace_path):
        try:
            SingleLogger().info("Reading comm trace from {}".format(comm_trace_path))
            with open(comm_trace_path, "r") as f:
                comm_trace = json.load(f)
            SingleLogger().info("Reading server trace from {}".format(server_trace_path))
            with open(server_trace_path, "r") as f:
                server_trace = json.load(f)
        except:
            SingleLogger().error("Cannot open trace files.")
            exit(1)
            return
        self.comm_trace_ = comm_trace
        self.server_trace_ = server_trace
        if isinstance(comm_trace, dict):
            self.comm_trace_content_ = comm_trace["traceEvents"]
        elif isinstance(comm_trace, list):
            self.comm_trace_content_ = comm_trace
        else:
            SingleLogger().error("Cannot parse BytePS comm trace.")
            return
        if isinstance(server_trace, dict):
            self.server_trace_content_ = server_trace["traceEvents"]
        elif isinstance(server_trace, list):
            self.server_trace_content_ = server_trace
        else:
            # SingleLogger().error("Cannot parse BytePS comm trace.")
            print("Cannot parse BytePS server trace.")
            return
        # parse tensor assignment information
        self._parse_trace()
        self._build_comm_graph()
        self._align_traces()
        self._calc_comm_delays()
        self._dump_cache()
        self._inited = True
    
    def init_from_cache(self, cache_path):
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        (self.gradient_assignment_dict, 
         self.comm_durations, self.comp_durations,
         self.servers_threads, self.partition_dict,
         self.comp_ops_tid, self.workers, self.graph,
         self.master_host_id, self.time_drift, self.comm_delays, self.bw_delays) = state
        self._inited = True
    
    def _dump_cache(self, cache_path=None):
        if cache_path is None:
            cache_path = os.path.join(args_.path, "bps_cache.pickle")
        state = (self.gradient_assignment_dict, 
                 self.comm_durations, self.comp_durations,
                 self.servers_threads, self.partition_dict,
                 self.comp_ops_tid, self.workers, self.graph,
                 self.master_host_id, self.time_drift, self.comm_delays, self.bw_delays)
        with open(cache_path, "wb") as f:
            pickle.dump(state, f)

    def _check_inited(self):
        if self._inited == False:
            raise RuntimeError("Must init byteps graph before use.")
    
    def get_master_host_id(self):
        return self.master_host_id

    def _add_prefix(self, device_name, tensor_name):
        return device_name + DEL + tensor_name

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
        tensor_name, op = event_name.split(".")
        return tensor_name, op
    
    def _parse_partition(self, tensor_name):
        if "~PART" in tensor_name:
            tensor_name_wo_part, part_id = tensor_name.split("~PART")
            if tensor_name_wo_part in self.partition_dict:
                self.partition_dict[tensor_name_wo_part].append(part_id)
            else:
                self.partition_dict[tensor_name_wo_part] = [part_id]
        
    def _gen_partitioned_name(self, tensor_name, part_id):
        return tensor_name + "~PART" + str(part_id)

    def _parse_trace(self):
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
                tensor_name, op = self._parse_event_name(event["name"])
                self._parse_partition(tensor_name)
                if "server" in source:
                    self.workers.add(target)
                    if source not in self.servers_threads:
                        self.servers_threads[source] = 0
                    op += "_RES"
                    if tensor_name in self.gradient_assignment_dict:
                        assert self.gradient_assignment_dict[tensor_name] == source
                    else:
                        self.gradient_assignment_dict[tensor_name] = source
                else:
                    self.workers.add(source)
                    if target not in self.servers_threads:
                        self.servers_threads[source] = 0 
                    op += "_REQ"
                    if tensor_name in self.gradient_assignment_dict:
                        assert self.gradient_assignment_dict[tensor_name] == target
                    else:
                        self.gradient_assignment_dict[tensor_name] = target
                if (source, target, tensor_name, op) not in self.comm_ops_dict:
                    self.comm_ops_dict[(source, target, tensor_name, op)] = []
                self.comm_ops_dict[(source, target, tensor_name, op)].append((event["ph"], event["ts"]))
            
        for key, events in self.comm_ops_dict.items():
            durations = []
            last_start = -1
            for index, ev in enumerate(events):
                if ev[0] == "B":
                    last_start = ev[1]
                else:
                    if last_start != -1:
                        durations.append((last_start, ev[1]))
                        last_start = -1
            if key not in self.comm_durations:
                self.comm_durations[key] = {}
            self.comm_durations[key] = durations

        # server trace
        for event in self.server_trace_content_:
            if event["ph"] == "M":
                if event["name"] == "process_name":
                    server = event["args"]["name"].strip()
                    self.pid_to_server[event["pid"]] = server

        for event in self.server_trace_content_:
            if event["ph"] != "M":
                server = self.pid_to_server[event["pid"]]
                tensor_name, op = self._parse_event_name(event["name"])
                self._parse_partition(tensor_name)
                self.servers_threads[server] = max(self.servers_threads[server], event["tid"])
                # sanity check
                assert self.gradient_assignment_dict[tensor_name] == server
                if (server, tensor_name, op) not in self.comp_ops_tid:
                    self.comp_ops_tid[(server, tensor_name, op)] = event["tid"]
                if (server, tensor_name, op, event["tid"]) not in self.comp_ops_dict:
                    self.comp_ops_dict[(server, tensor_name, op, event["tid"])] = []
                self.comp_ops_dict[(server, tensor_name, op, event["tid"])].append((event["ph"], event["ts"]))
        
        for key, events in self.comp_ops_dict.items():
            durations = []
            last_start = -1
            for index, ev in enumerate(events):
                if ev[0] == "B":
                    last_start = ev[1]
                else:
                    if last_start != -1:
                        durations.append((last_start, ev[1]))
                        last_start = -1
            if key not in self.comp_durations:
                self.comp_durations[key] = {}
            self.comp_durations[key] = durations

        for key, durations in self.comm_durations.items():
            _, _, _, op = key
            if op == COMM_OPS.PUSH_REQ or op == COMM_OPS.PUSH_RES:
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
            _, _, _, op = key
            if op == COMM_OPS.PUSH_REQ or op == COMM_OPS.PUSH_RES:
                if len(durations) != mode_len + 1:
                    self._ignored_tensors.add(key)
                chopped_durations = durations[self.PROFILE_ITER_START:self.PROFILE_ITER_START+self.PROFILE_ITER_DURATION]
            else:
                if len(durations) != mode_len:
                    self._ignored_tensors.add(key)
                chopped_durations = durations[self.PROFILE_ITER_START-1:self.PROFILE_ITER_START-1+self.PROFILE_ITER_DURATION]
            self.comm_durations[key] = chopped_durations
        
        for key, durations in self.comp_durations.items():
            if len(durations) != mode_len:
                self._ignored_tensors.add(key)
            chopped_durations = durations[self.PROFILE_ITER_START-1:self.PROFILE_ITER_START-1+self.PROFILE_ITER_DURATION]
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
        # comm_key is in format (source, target, tensor_name, op)
        source, target, _, _ = comm_key
        return source + COMM_DEL + target

    def gen_comm_event_name(self, comm_key):
        # comm_key is in format (source, target, tensor_name, op)
        _, _, tensor_name, op = comm_key
        return self.gen_comm_device_name(comm_key) + COMM_DEL + tensor_name + "." + op

    def gen_comm_full_name(self, comm_key):
        return self._add_prefix(self.gen_comm_device_name(comm_key), self.gen_comm_event_name(comm_key))

    def parse_comm_event_name(self, comm_event_name):
        comm_name, op = comm_event_name.split(".")
        source, target, tensor_name = self._parse_comm_name(comm_name)
        return source, target, tensor_name, op

    def _parse_comm_name(self, comm_name):
        # comm_name = event name - op
        source, target, tensor_name = comm_name.split(COMM_DEL)
        return source, target, tensor_name

    def gen_comm_unique_pid(self, comm_key):
        # comm_key is in format (source, target, tensor_name, op)
        # return self._str_hashing(self.gen_comm_device_name(comm_key))
        return self.gen_comm_device_name(comm_key)
    
    def gen_comp_device_name(self, comp_key):
        # comp_key is in format (server, tensor_name, op, tid)
        server, _, _, tid = comp_key
        return server + "_t" + str(tid)

    def gen_comp_event_name(self, comp_key, sum_index = None):
        # comp_key is in format (server, tensor_name, op, tid)
        _, tensor_name, op, _ = comp_key
        if sum_index is None:
            return self.gen_comp_device_name(comp_key) + COMM_DEL + tensor_name + "." + op
        else:
            return self.gen_comp_device_name(comp_key) + COMM_DEL + str(sum_index) + COMM_DEL + tensor_name + "." + op
    
    def gen_comp_full_name(self, comp_key, sum_index=None):
        return self._add_prefix(self.gen_comp_device_name(comp_key), self.gen_comp_event_name(comp_key, sum_index=sum_index))
    
    def _parse_comp_name(self, comp_name):
        # comp_name = event name - op
        _, server_id, tid, tensor_name = comp_name.split(COMM_DEL)
        return server_id, tid, tensor_name
    
    def gen_comp_unique_pid(self, comp_key):
        # comp_key is in format (server, tensor_name, op, tid)
        # return self._str_hashing(self.gen_comp_device_name(comp_key))
        return self.gen_comp_device_name(comp_key)
    
    def _str_hashing(self, s):
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

    def _get_node_from_dev_name(self, s):
        return int(s.split("_")[-1])
    
    def _align_traces(self):
        # comm_key is in format (source, target, tensor_name, op)
        source_ranks = set()
        durations_dict = {}
        unique_tensors = set()
        copy_first_ops = {}
        intervals = {}
        master_node = -1

        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op = key
            unique_tensors.add(tensor_name)
            source_rank = self._get_node_from_dev_name(source)
            source_ranks.add(source_rank)
            if "server" in source:
                if source_rank > master_node:
                    master_node = source_rank
            if source_rank not in durations_dict:
                durations_dict[source_rank] = {}
            if key not in intervals:
                intervals[key] = IntervalTree()
            for st, ed in durations:
                intervals[key][st:ed] = True
            durations_dict[source_rank][key] = durations

        for key, durations in self.comp_durations.items():
            # comp_key is in format 
            server, tensor_name, op, tid = key
            unique_tensors.add(tensor_name)
            if op != COMP_OPS.COPY_FIRST:
                continue
            if server not in copy_first_ops:
                copy_first_ops[server] = {}
            if server not in intervals:
                intervals[server] = IntervalTree()
            for st, ed in durations:
                intervals[server][st:ed] = True
            copy_first_ops[server][tensor_name] = durations
        
        if master_node == -1:
            SingleLogger().error("Cannot find server node traces.")
            return
        
        send_delays = {}
        for source_rank in durations_dict.keys():
            if source_rank != master_node:
                send_delays[source_rank] = float('inf')

        self.master_host_id = master_node

        def _before_align_is_first_push(key, index):
            source, target, tensor_name, op = key
            my_rank = self._get_node_from_dev_name(source)
            my_st, my_ed = durations_dict[master_node][("server_" + str(master_node), source, tensor_name, "PUSH_RES")][index]
            for source_rank in source_ranks:
                st, ed = durations_dict[master_node][("server_" + str(master_node), "worker_"+str(source_rank), tensor_name, "PUSH_RES")][index]
                if st < my_st:
                    return False
            return True

        for source_rank, key_dict in durations_dict.items():
            if source_rank == master_node:
                continue
            for key, durations in key_dict.items():
                source, target, tensor_name, op = key
                if key in self._ignored_tensors:
                    SingleLogger().warn(
                            "[BPS ALIGN]: Length mismatch between master server ({}) and node {} on tensor {}".format(
                                "server_" + str(master_node), source, tensor_name))
                    continue
                if target == "server_" + str(master_node) and op == "PUSH_REQ":
                    # an op that can be used to align traces
                    for index in range(len(durations)):
                        if not _before_align_is_first_push(key, index):
                            ms_key = ("server_" + str(master_node), source, tensor_name, "PUSH_RES")
                            ms_st, ms_ed = durations_dict[master_node][ms_key][index]
                            if not intervals[ms_key].overlap(ms_st-500, ms_st):
                                send_delays[source_rank] = min(send_delays[source_rank] ,ms_st - durations[index][1])
                        else:
                            # need to consult first_copy
                            fc_st, fc_ed = copy_first_ops["server_"+str(master_node)][tensor_name][index]
                            if not intervals["server_"+str(master_node)].overlap(fc_st-500, fc_st):
                                send_delays[source_rank] = min(send_delays[source_rank], fc_st - durations[index][1])

        SingleLogger().info("# Aligning BPS traces")
        SingleLogger().info("Aligning time based on node {}".format(master_node))
        for key, item in send_delays.items():
            SingleLogger().info("Shifting traces of node {} {} by {} us.".format(key, "forward" if item >= 0 else "backward", np.abs(item)))
        
        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op = key
            if self._get_node_from_dev_name(source) in send_delays:
                delay = send_delays[self._get_node_from_dev_name(source)]
                new_durations = []
                for st, ed in durations:
                    new_durations.append((st+delay, ed+delay))
                self.comm_durations[key] = new_durations
        
        for key, durations in self.comp_durations.items():
            server, tensor_name, op, tid = key
            if self._get_node_from_dev_name(server) in send_delays:
                delay = send_delays[self._get_node_from_dev_name(server)]
                new_durations = []
                for st, ed in durations:
                    new_durations.append((st+delay, ed+delay))
                self.comp_durations[key] = new_durations
        
        self.time_drift = send_delays

    def _calc_comm_delays(self):
        intervals = {}
        network_delays = {}
        push_req_ops = {}
        push_res_ops = {}
        pull_req_ops = {}
        copy_first_ops = {}

        source_ranks = set()
        durations_dict = {}

        unique_tensors = set()

        min_start_time = float('inf')
        for key, durations in self.comm_durations.items():
            source, target, tensor_name, op = key
            if source not in intervals:
                intervals[source] = IntervalTree()
            unique_tensors.add(tensor_name)
            source_rank = self._get_node_from_dev_name(source)
            source_ranks.add(source_rank)
            for st, ed in durations:
                min_start_time = min(min_start_time, st)
                intervals[source][st:ed] = (tensor_name, op)
            if op == COMM_OPS.PUSH_REQ:
                if (source, target) not in push_req_ops:
                    push_req_ops[(source, target)] = {}
                push_req_ops[(source, target)][key] = durations
            elif op == COMM_OPS.PUSH_RES:
                if (source, target) not in push_res_ops:
                    push_res_ops[(source, target)] = {}
                push_res_ops[(source, target)][key] = durations
            elif op == COMM_OPS.PULL_REQ:
                if (source, target) not in pull_req_ops:
                    pull_req_ops[(source, target)] = {}
                pull_req_ops[(source, target)][key] = durations

        for key, durations in self.comp_durations.items():
            # comp_key is in format 
            server, tensor_name, op, tid = key
            if server not in intervals:
                intervals[server] = IntervalTree()
            unique_tensors.add(tensor_name)
            if op != COMP_OPS.COPY_FIRST:
                continue
            for st, ed in durations:
                intervals[server][st:ed] = (tensor_name, op)
            if server not in copy_first_ops:
                copy_first_ops[server] = {}
            copy_first_ops[server][tensor_name] = durations

        # ops_name = ["push_res_ops", "push_res_ops", "pull_req_ops", "copy_first_ops"]
        # for index, ops in enumerate([push_res_ops, push_res_ops, pull_req_ops, copy_first_ops]):
        #     print("\n{}:\n".format(ops_name[index]))
        #     for k, v in ops.items():
        #         for kk, vv in v.items():
        #             if len(vv) != 30:
        #                 print(k, kk, len(vv))
        # exit(0)

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
            # for each push req, get delay for its push res or copy first
            for key, durations in push_req_ops[(source, target)].items():
                _, _, tensor_name, _ = key
                if key in self._ignored_tensors:
                    continue
                copy_first_op_durations = copy_first_ops[target][tensor_name]
                push_res_op_durations = push_res_ops[(target, source)][(target, source, tensor_name, COMM_OPS.PUSH_RES)]
                for index, (st, ed) in enumerate(durations):
                    if _is_first_push(key, index):
                        # get copy_first
                        cp_st, cp_ed = copy_first_op_durations[index]
                        # if not queued
                        if not intervals[target].overlap(st - 500, st + 500) and not intervals[target].overlap(cp_st - 500, cp_st):
                            latency = cp_st - ed
                            if (source, target) not in network_delays:
                                network_delays[(source, target)] = []
                            network_delays[(source, target)].append(latency)
                    else:
                        # get push_response
                        rs_st, rs_ed = push_res_op_durations[index]
                        # if not queued
                        if not intervals[target].overlap(st - 500, st + 500) and not intervals[target].overlap(rs_st-500, rs_st):
                            latency = rs_st - ed
                            # if latency > 10000 and source == "worker_0" and target == "server_0":
                            #     print("push_res, index:", index, key, st - min_start_time, rs_st-min_start_time)
                            #     print(intervals[target].overlap(st - 500, st+500))
                            #     exit(0)
                            if (source, target) not in network_delays:
                                network_delays[(source, target)] = []
                            network_delays[(source, target)].append(latency)

        for (source, target) in push_res_ops.keys():
            for key, durations in push_res_ops[(source, target)].items():
                _, _, tensor_name, _ = key
                if key in self._ignored_tensors:
                    continue
                pull_req_op_durations = pull_req_ops[(target, source)][(target, source, tensor_name, COMM_OPS.PULL_REQ)]
                for index, (st, ed) in enumerate(durations):
                    # get pull request
                    rq_st, rq_ed = pull_req_op_durations[index]
                    # if not queued
                    if not intervals[target].overlap(st - 500, st + 500) and not intervals[target].overlap(rq_st - 500, rq_st):
                        latency = rq_st - ed
                        # if latency > 40000 and source == "server_0" and target == "worker_1":
                        #     print("pull_req, index:", index, key, st - min_start_time, rq_st-min_start_time)
                        #     print(intervals[target].overlap(st - 100, st+100))
                        #     exit(0)
                        if (source, target) not in network_delays:
                            network_delays[(source, target)] = []
                        network_delays[(source, target)].append(latency)
        
        for key, items in network_delays.items():
            avg_delay = np.average(items)
            network_delays[key] = avg_delay
            SingleLogger().info("Comm delay for {} is {} us.".format(key, avg_delay))

        self.comm_delays = network_delays

    def gen_compatible_trace(self, dump_path=None):
        self._check_inited()
        trace = []
        for key, durations in self.comm_durations.items():
            for index, (st, ed) in enumerate(durations):
                json_event = {}
                json_event["name"] = self.gen_comm_event_name(key)
                json_event["ph"] = "X"
                json_event["ts"] = st
                json_event["pid"] = self.gen_comm_unique_pid(key)
                json_event["tid"] = 0
                json_event["dur"] = ed - st
                json_event["cat"] = "Comm"
                json_event["args"] = {}
                trace.append(json_event)

        for key, durations in self.comp_durations.items():
            _, _, op, _ = key
            for index, (st, ed) in enumerate(durations):
                if op == COMP_OPS.SUM:
                    for i in range(len(self.workers)-1):
                        json_event = {}
                        json_event["name"] = self.gen_comp_event_name(key, i)
                        json_event["ph"] = "X"
                        json_event["ts"] = st
                        json_event["pid"] = self.gen_comp_unique_pid(key)
                        json_event["tid"] = 0
                        json_event["dur"] = ed - st
                        json_event["cat"] = "operator"
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
                    json_event["cat"] = "operator"
                    json_event["args"] = {}
                    trace.append(json_event)

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
        if tensor_name in self.partition_dict:
            # has partitions
            partitions = [self._gen_partitioned_name(tensor_name, pid) for pid in self.partition_dict[tensor_name]]
        else:
            partitions = [tensor_name]
        full_names = []
        for partitioned_name in partitions:
            # (source, target, tensor_name, op)
            key = ("worker_"+str(source_id), self.gradient_assignment_dict[partitioned_name], partitioned_name, COMM_OPS.PUSH_REQ)
            full_names.append(self.gen_comm_full_name(key))
        return full_names
    
    def get_pull_res_node(self, source_id, tensor_name):
        self._check_inited()
        if tensor_name in self.partition_dict:
            # has partitions
            partitions = [self._gen_partitioned_name(tensor_name, pid) for pid in self.partition_dict[tensor_name]]
        else:
            partitions = [tensor_name]
        full_names = []
        for partitioned_name in partitions:
            # (source, target, tensor_name, op)
            key = (self.gradient_assignment_dict[partitioned_name], "worker_"+str(source_id), partitioned_name, COMM_OPS.PULL_RES)
            full_names.append(self.gen_comm_full_name(key))
        return full_names

    def _build_comm_graph(self):
        for tensor_name, assigned_server in self.gradient_assignment_dict.items():
            # push req -> push res -> pull req
            for worker_name in self.workers:
                self.graph.add_edge(
                    self.gen_comm_full_name(
                        (worker_name, assigned_server, tensor_name, COMM_OPS.PUSH_REQ)
                    ),
                    self.gen_comm_full_name(
                        (assigned_server, worker_name, tensor_name, COMM_OPS.PUSH_RES)
                    )
                )
                self.graph.add_edge(
                    self.gen_comm_full_name(
                        (assigned_server, worker_name, tensor_name, COMM_OPS.PUSH_RES)
                    ),
                    self.gen_comm_full_name(
                        (worker_name, assigned_server, tensor_name, COMM_OPS.PULL_REQ)
                    )
                )

            # copy_first -> sum * N-1 -> copy_merged
            self.graph.add_edge(
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    COMP_OPS.COPY_FIRST, 
                    self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.COPY_FIRST)])
                    ),
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    COMP_OPS.SUM, 
                    self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.SUM)]),
                    sum_index=0
                    ),
                weight=0
            )
            for i in range(1, len(self.workers)-1):
                self.graph.add_edge(
                    self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        COMP_OPS.SUM, 
                        self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.SUM)]),
                        sum_index=i-1
                        ),
                    self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        COMP_OPS.SUM, 
                        self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.SUM)]),
                        sum_index=i
                        ),
                    weight=0
                )
            self.graph.add_edge(
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    COMP_OPS.SUM, 
                    self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.SUM)]),
                    sum_index=len(self.workers)-2
                    ),
                self.gen_comp_full_name(
                    (assigned_server, 
                    tensor_name, 
                    COMP_OPS.COPY_MERGED, 
                    self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.COPY_MERGED)])
                    ),
                weight=0
            )

            # pull req -> pull res, copy_merged -> pull_res
            for worker_name in self.workers:
                self.graph.add_edge(
                    self.gen_comm_full_name((worker_name, assigned_server, tensor_name, COMM_OPS.PULL_REQ)),
                    self.gen_comm_full_name((assigned_server, worker_name, tensor_name, COMM_OPS.PULL_RES)),
                    weight=0
                )
                self.graph.add_edge(
                    self.gen_comp_full_name(
                        (assigned_server, 
                        tensor_name, 
                        COMP_OPS.COPY_MERGED, 
                        self.comp_ops_tid[(assigned_server, tensor_name, COMP_OPS.COPY_MERGED)])
                        ),
                    self.gen_comm_full_name((assigned_server, worker_name, tensor_name, COMM_OPS.PULL_RES)),
                    weight=0
                )

    def is_server_comp(self, name):
        for op in [COMP_OPS.COPY_FIRST, COMP_OPS.SUM]:
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
            source, target, tensor_name, op = key
            if op == COMM_OPS.PUSH_REQ:
                for st, ed in durations:
                    if source not in interval:
                        interval[source] = IntervalTree()
                    interval[source][st:ed] = tensor_name
                if source not in push_req_ops:
                    push_req_ops[source] = {}
                push_req_ops[source][tensor_name] = durations

        bw_delay_dict = {}

        def get_tensor_name_from_full_name(n):
            _, _, tensor_name, _ = self.parse_comm_event_name(parse_rawname_from_name(n))
            return tensor_name

        for process_name, tensor_dict in bw_durations.items():
            for layer_name, evs in tensor_dict.items():
                long_name = gen_long_name(process_name, layer_name)
                if long_name in dag:
                    node_rank = process_name.split(".")[0].split("_")[-1]
                    local_rank = process_name.split(".")[1].split("rank")[-1]
                    # push_tensor_names = [n.split(".")[-1] for n in dag.neighbors(long_name) if "Comm" in n]
                    push_tensor_names = [get_tensor_name_from_full_name(n) for n in dag.neighbors(long_name) if "PUSH_REQ" in n]
                    matched_tensor_names = []
                    for tensor_name in push_req_ops["worker_"+node_rank].keys():
                        if tensor_name in push_tensor_names:
                            matched_tensor_names.append(tensor_name)
                    for index in range(len(evs)):
                        local_min_delay = float('inf')
                        for tensor_name in matched_tensor_names:
                            tensor_durations = push_req_ops["worker_"+node_rank][tensor_name]
                            assert len(evs) == len(tensor_durations)
                            bw_st, bw_ed = evs[index]
                            pu_st, pu_ed = tensor_durations[index]
                            if not interval["worker_"+node_rank].overlap(bw_ed - 1000, bw_ed + 50):
                                local_min_delay = min(local_min_delay, pu_st - bw_ed)
                                # print(layer_name, tensor_name, index, pu_st - bw_ed)
                        if node_rank not in bw_delay_dict:
                            bw_delay_dict[node_rank] = {}
                        if local_rank not in bw_delay_dict[node_rank]:
                            bw_delay_dict[node_rank][local_rank] = []
                        if local_min_delay != float('inf'):
                            bw_delay_dict[node_rank][local_rank].append(local_min_delay)
                else:
                    SingleLogger().warn("BYTEPS BW Delay: {} not in dag.".format(long_name))
        
        node_delay_dict = {}
        for node_rank, local_dict in bw_delay_dict.items():
            min_avg = float("inf")
            for local_rank, delays in local_dict.items():
                avg = np.average(delays)
                if avg < min_avg:
                    min_avg = avg
            SingleLogger().info("BW delay of {} is {} us.".format("worker_"+node_rank, avg))
            node_delay_dict["worker_"+node_rank] = min_avg

        self.bw_delays = node_delay_dict

        return node_delay_dict