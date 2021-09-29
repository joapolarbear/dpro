import networkx as nx
import os
import pickle
import numpy as np
import math
import random
import ujson as json
import collections
from tqdm import tqdm, trange
from scipy.optimize import curve_fit
import copy

import arg_utils
from .base import _BaseGraphPass
from trace_utils import *
from cost_model._xla.pk_graph import PKGraph
from cost_model._tsfs.cost_model import predict_ps_inter_comm_time, predict_ps_intra_comm_time

from bps_helper.graph import PS_COMM_OPS, PS_COMP_OPS, PS_COMM_OPS_SETS, PS_COMP_OPS_SETS

args_ = arg_utils.SingleArg().args

FUSION_RATIO = 1
TRAIN_PERCENT = 0.9
IGNORE_LARGE_ERROR = True
### given a fused tensor, return N possible partition method
# N <= MAX_PARTITION_NUM
MAX_PARTITION_NUM = 5
ROOT_PATH = os.path.join(
    args_.workspace if args_.workspace else args_.path, ".opt_ws")
IGNORE_SYNC = True

### For those sub ops, we assume that tensor size has little effect on the execution time 
CONSTANT_SUB_OPS = [PS_COMM_OPS.PUSH_RES, PS_COMM_OPS.PULL_REQ] + list(PS_COMP_OPS_SETS)

def func_tensor_size_to_time(s, k, b):
    return k * s + b

class TensorFusionState:
    def __init__(self, graph_pass):
        self.graph_pass = graph_pass
        self.opt = self.graph_pass.opt
        
        self.ckpt_path = os.path.join(self.graph_pass.ckpt_dir, "ckpt_tensor_fusion_state.pickle")
        self.spec_dir = self.graph_pass.spec_dir
            
        self.cur_tensor2group = {}
        self.num_grp = 1 if args_.search_ts_group_num else None
        self.history_num_grp = {}
        self.tensor_group_info = {}

        self.grp_part_id2server = None
    
    def ps_new_tensor_group(self, fused_tensor_name, new_tensor_size, new_part_num):
        self.tensor_group_info[fused_tensor_name] = {
            'size': new_tensor_size,
            'part_num': new_part_num
        }
    
    def nccl_new_tensor_grp(self, new_tensor_grp, tensor_grp_u, tensor_grp_v, refer_grp=None):
        if self.is_tensor_grp_exist(new_tensor_grp):
            return
        self.tensor_group_info[new_tensor_grp] = self.tensor_group_info[refer_grp].copy()
        self.tensor_group_info[new_tensor_grp]["size"] = \
            self.tensor_group_info[tensor_grp_u]["size"] + \
            self.tensor_group_info[tensor_grp_v]["size"]
    
    def is_tensor_grp_exist(self, grp_name):
        return grp_name in self.tensor_group_info
    
    def _tensor_grp_size(self, op_name):
        return self.graph_pass.meta_info.tensor_grp_size(op_name)

    def parse_tensor_group_info(self, op_name, ref=None):
        ''' op_name must be sorted '''
        if op_name not in self.tensor_group_info:
            if self.opt.comm_backend == "NCCL":
                if ref is None:
                    chunkNum, sliceNum, channelNum, loopNum = self.opt.clct.nccl_graph.get_IDnum(self.graph_pass._from_tensor_name2rawname(op_name))
                    grp_info = {
                        "chunkNum": chunkNum,
                        "sliceNum": sliceNum,
                        "channelNum": channelNum,
                        "loopNum": loopNum,
                        "part_num": loopNum*channelNum*sliceNum*(chunkNum/2 + 1)
                    }
                else:
                    ### some group may not occur in the traces, ref to other groups
                    grp_info = self.parse_tensor_group_info(ref).copy()
            else:
                grp_info = {
                        "part_num": len(self.opt.clct.byteps_graph.partition_dict.get(op_name, ['0']))
                    }

            total_size = self._tensor_grp_size(op_name)
            grp_info["size"] = total_size
            self.tensor_group_info[op_name] = grp_info
        return self.tensor_group_info[op_name]
    
    def parse_tensor_cur_grp(self, tensor):
        return self.cur_tensor2group(tensor)
    
    def parse_tensor_num(self):
        return len(self.cur_tensor2group)

    def sorted_all_tensor_groups(self):
        return sorted(list(set(self.cur_tensor2group.values())), key=lambda grp_id: int(grp_id.split("+")[0]))
    
    def ret_weight_num_grp(self, grp_num):
        if grp_num not in self.history_num_grp:
            return 1
        else:
            return 1.0 / float(self.history_num_grp[grp_num] + 1)

    def is_grp_num_in_history(self, grp_num):
        return grp_num in self.history_num_grp
    
    def update_grp_history(self):
        self.history_num_grp[self.num_grp] = 1 if self.num_grp not in self.history_num_grp else self.history_num_grp[self.num_grp] + 1

    def update_part_num(self, tensor_grp, part_num):
        self.tensor_group_info[tensor_grp]["part_num"] = part_num

    def wrap_parse_ps_server_tid(self, server_id, tensor_name, sub_op, part_id, ref_tid=None):
        comm_key = (server_id, tensor_name, sub_op, str(part_id))
        if comm_key in self.tensor_group_info:
            return self.tensor_group_info[comm_key]

        if comm_key not in self.opt.clct.byteps_graph.comp_ops_tid:
            assert ref_tid is not None
            self.opt.clct.byteps_graph.comp_ops_tid[comm_key] = ref_tid
            self.tensor_group_info[comm_key] = ref_tid
            return ref_tid
        else:
            tid = self.opt.clct.byteps_graph.comp_ops_tid[comm_key]
            self.tensor_group_info[comm_key] = tid
            return tid
        '''
        ### TODO (huhanpeng): currently direcly use the server_tid of the first tensor as the 
        # tid of the fused tensor
        tid_list = []
        for _tensor_id_str in tensor_name.split("+"):
            try:
                tid = self.opt.clct.byteps_graph.comp_ops_tid[(server_id, _tensor_id_str, sub_op, str(part_id))]
                tid_list.append(tid)
            except KeyError:
                ### For fused tensor, the partition id may be large and some
                pass
        assert len(tid_list) > 0, comm_key
        self.tensor_group_info[comm_key] = tid_list[0]
        return tid_list[0]
        '''
    
    def dump_tensor_grp_mapping(self, _file_name=None):
        file_name = 'tensor_fusion_grp_mapping.json' if _file_name is None else _file_name

        tensor_ids, tensor_grps = zip(*list(self.cur_tensor2group.items()))
        tensor_grps = set(tensor_grps)
        tensor_ids = set(tensor_ids)
        assert len(tensor_ids) == self.graph_pass.meta_info.gradient_num(), \
            ("incompleted tensor_ids {} : {}".format(sorted(tensor_ids), self.graph_pass.meta_info.gradient_num()))

        with open(os.path.join(self.spec_dir, file_name), 'w') as f:
            json.dump({"mapping": list(tensor_grps)}, f)
        
    def dump_tensor_partitions(self, _file_name=None):
        file_name = 'tensor_partition_spec.txt' if _file_name is None else _file_name
        tensor_grps = list(self.cur_tensor2group.values())
        tensor_grps = set(tensor_grps)
        with open(os.path.join(self.spec_dir, file_name), 'w') as f:
            for tensor_grp in tensor_grps:
                grp_info = self.tensor_group_info[tensor_grp]
                f.write("Distributed_Push_Pull/BytePSPushPull.{} {}\n".format(tensor_grp.replace("+", "_"), int(grp_info["size"] / grp_info["part_num"])))
    
    def update_tensor2grp(self, n):
        if "." not in n:
            op_name = n
        else:
            op_name = self.graph_pass._wrap_parse_all_info(n)[2]
        for tensor_id in op_name.split("+"):
            self.cur_tensor2group[int(tensor_id)] = op_name

    def dump(self, dump_path=None):
        _path = self.ckpt_path if dump_path is None else dump_path
        with open(_path, "wb") as f:
            pickle.dump([self.tensor_group_info, self.cur_tensor2group, self.num_grp, self.history_num_grp], f)
        
    def load(self, load_path=None):
        _path = self.ckpt_path if load_path is None else load_path
        assert os.path.isfile(_path)
        with open(_path, "rb") as f:
            self.tensor_group_info, self.cur_tensor2group, self.num_grp, self.history_num_grp = pickle.load(f)
    
    def copy(self):
        new_instance = self.__class__(self.graph_pass)
        new_instance.tensor_group_info = copy.deepcopy(self.tensor_group_info)
        new_instance.cur_tensor2group = self.cur_tensor2group.copy()
        new_instance.num_grp = self.num_grp
        new_instance.history_num_grp = self.history_num_grp.copy()
        return new_instance
    
    def update_tensor2server(self):
        servre_num = len(self.opt.clct.byteps_graph.pid_to_server.values())
        grp_part_id2server = {}
        soreted_groups = self.sorted_all_tensor_groups()
        key = 0
        for grp in soreted_groups:
            grp_info = self.parse_tensor_group_info(grp)
            part_num = grp_info["part_num"]
            grp_part_id2server[grp] = {}
            for part_id in range(part_num):
                grp_part_id2server[grp][part_id] = key % servre_num
                key += 1
        self.grp_part_id2server = grp_part_id2server

class TensorFusionGraphPass(_BaseGraphPass):
    ''' This is a cost model for HOROVOD tensor fusion
    '''
    def __init__(self, opt, root_path):
        super().__init__(opt)

        self.root_path = root_path
        self.ckpt_path = os.path.join(self.ckpt_dir, "ckpt_tensor_fusion.pickle")

        self.token = ["++", "--"]
        self.cord_pid = self.opt.cord_pid

        self.tsfs_state = TensorFusionState(self)
        if self.tsfs_state.num_grp is not None:
            SingleLogger().info("Search the optimal number of tensor fusion groups")
        else:
            SingleLogger().info("Search the optimal tensor fusion strategies")

        for n in self.dag.nodes():
            if "Comm" in n:
                ### Assume each host has the same tensor fusion pattern
                if self.cord_pid in n:
                    self.tsfs_state.update_tensor2grp(n)
        
        ### Store the cost model for tensor fusion/partition
        self.pid_to_cm = None

        ### Tensor level cost model, a tuple where the first element is the slope
        self.send_cm = None
        self.recv_cm = None

        self.enable_partition = True
        self.enable_defusion = True

    def _init_search_space_num_grp(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []
        
        assert self.tsfs_state.num_grp is not None
        if self.tsfs_state.num_grp == 1:
            if not self.tsfs_state.is_grp_num_in_history(self.tsfs_state.num_grp + 1):
                search_space.append(("++", self.tsfs_state.num_grp + 1, None))
                weights.append(self.tsfs_state.ret_weight_num_grp(self.tsfs_state.num_grp + 1))
        else:
            if not self.tsfs_state.is_grp_num_in_history(self.tsfs_state.num_grp + 1):
                search_space.append(("++", self.tsfs_state.num_grp + 1, None))
                weights.append(self.tsfs_state.ret_weight_num_grp(self.tsfs_state.num_grp + 1))
            if not self.tsfs_state.is_grp_num_in_history(self.tsfs_state.num_grp - 1):
                search_space.append(("++", self.tsfs_state.num_grp - 1, None))
                weights.append(self.tsfs_state.ret_weight_num_grp(self.tsfs_state.num_grp - 1))
        return search_space, weights

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        if self.tsfs_state.num_grp is not None:
            return self._init_search_space_num_grp(candidates, _dag, _pkg)
        
        search_space = []
        weights = []
        # debug_st = [
        #     ("++", "host0.rank2->Comm.1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28.Sync", "host0.rank2->Comm.0.Sync"),
        #     ("++", "host0.rank2->Comm.29+30+31.Sync",
        #      "host0.rank2->Comm.32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+47+48+49+50+51+52+53+54+55+56+57+58+59+60+61+62+63+64+65+66.Sync"),
        #     ("++", "host0.rank2->Comm.0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28.Sync",
        #      "host0.rank2->Comm.29+30+31+32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+47+48+49+50+51+52+53+54+55+56+57+58+59+60+61+62+63+64+65+66.Sync"),
        #     ("--", "host0.rank2->Comm.1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28.Sync", 5)
        # ]
        # return [debug_st[0]], [1]

        for n, l in candidates:
            if "Comm" in n and "Sync" in n:
                to_fuse_u = n
                to_fuse_v = None

                all_info = self._wrap_parse_all_info(n)
                target_id = None
                last_id = None
                for tensor_id_str in sorted([int(_n) for _n in all_info[2].split("+")], reverse=True):
                    if last_id is None:
                        last_id = int(tensor_id_str)
                    else:
                        if int(tensor_id_str) == last_id - 1:
                            last_id = int(tensor_id_str)
                        else:
                            target_id = last_id - 1
                            SingleLogger().info("Unordered tensor {}, target {}.".format(all_info[2], target_id))
                            break
                if target_id is None and last_id > 0:
                    target_id = last_id - 1

                if target_id is None:
                    SingleLogger().info("{} is the last tensor (contains tensor id 0)"
                        "can not be further fused.".format(to_fuse_u))
                else:
                    grp_name = self.tsfs_state.parse_tensor_cur_grp(target_id)
                    to_fuse_v = self._wrap_gen_long_name(
                        all_info[0], all_info[1], grp_name, all_info[3], all_info[4])
                    assert to_fuse_u != to_fuse_v, (target_id, last_id, all_info[2]) 
                    search_space.append(("++", to_fuse_u, to_fuse_v))
                    weights.append(l)
                '''
                ### For a fused tensor, it may have multiple BW inputs
                bw_list = [u for u, _ in _dag.in_edges(n)]
                to_check_list = []
                for bw_node in bw_list:
                    _succ = [n for n in _dag.successors(bw_node) if ("BW" in n and n not in bw_list)]
                    # assert len(_succ) < 2, (bw_node, _succ)
                    if len(_succ) > 0:
                        to_check_list += _succ

                ### Find the nearest tensors
                have_process = set()
                while len(to_check_list) > 0 and to_fuse_v is None:
                    bw_node = to_check_list.pop()
                    for _succ in list(_dag.successors(bw_node)):
                        if "BW" in _succ:
                            if _succ not in bw_list:
                                assert _succ not in have_process, (
                                    bw_node, _succ)
                                have_process.add(_succ)
                                to_check_list.append(_succ)
                        elif "Comm" in _succ and "Sync" in _succ and _succ != to_fuse_u:
                            to_fuse_v = _succ
                            break
                        else:
                            raise ValueError(_succ)
                ### to_fuse_v may be None if to_fuse_u has been the laste tensor
                if to_fuse_v is not None:
                    search_space.append(("++", to_fuse_u, to_fuse_v))
                    weights.append(l)
                '''

                if self.enable_defusion:
                    ### tensor partition
                    all_infos = self._wrap_parse_all_info(to_fuse_u)
                    sorted_tensor_ids = [int(_n) for _n in all_infos[2].split("+")]
                    mu, sigma = (len(sorted_tensor_ids)-1) / 2, (len(sorted_tensor_ids)-1) / 4
                    s = np.random.normal(mu, sigma, min(len(sorted_tensor_ids) - 1, MAX_PARTITION_NUM))
                    _set = set([int(_n) for _n in s])
                    for _s in _set:
                        if _s > 0 and _s <= len(sorted_tensor_ids)-1:
                            ### Here _s denotes the partition occurs before the _s'th tensor
                            search_space.append(("--", to_fuse_u, _s))
                            weights.append(l)
   
        return search_space, weights

    def apply(self, s, __dag, __pkg):
        op, target, next_ = s
        if op == "++":
            ### Fuse two nodes
            if isinstance(target, int):
                return self._apply_grp_num(__dag, __pkg, target)
            return self._tensor_fusion(__dag, __pkg, target, next_)
        elif op == "--":
            return self._tensor_defusion(__dag, __pkg, target, next_)
    
    def _apply_grp_num(self, _dag, _pkg, num_grp):
        ''' The method used in horovod to construct tensor groups
        * here `l` is the list of tensors, `n` is the number of groups
        ```
            d, r = divmod(len(l), n)
            return [l[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]
        ```
        '''

        tensor_num = self.tsfs_state.parse_tensor_num()
        num_per_grp, _ = divmod(tensor_num, num_grp)
        
        trajectory = []
        residual = []
        groups = self.tsfs_state.sorted_all_tensor_groups()
        for grp in groups:
            tensor_ids = grp.split("+")
            if len(residual) + len(tensor_ids) == num_per_grp:
                if len(residual) > 0:
                    to_fuse_u = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                    residual = []
            elif len(residual) + len(tensor_ids) < num_per_grp:
                if len(residual) > 0:
                    to_fuse_u = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                residual += tensor_ids
            else:
                if len(residual) > 0:
                    tensor_name = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("--", tensor_name, num_per_grp - len(residual)))
                    to_fuse_u = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name(
                        self.cord_pid, "Comm", "+".join(tensor_ids[:(num_per_grp - len(residual))]), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                    tensor_ids = tensor_ids[(num_per_grp - len(residual)):]
                while len(tensor_ids) > num_per_grp:
                    tensor_name = self._wrap_gen_long_name(self.cord_pid, "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("--", tensor_name, num_per_grp))
                    tensor_ids = tensor_ids[num_per_grp:]
                if len(tensor_ids) < num_per_grp:
                    residual = tensor_ids
                else:
                    residual = []

        SingleLogger().info("From {} groups to {} groups, apply {} strategies in totoal ...".format(self.tsfs_state.num_grp, num_grp, len(trajectory)))
        rst = [True, [], []]
        for st in tqdm(trajectory, total=len(trajectory)):
            # print(st)
            succ_, nodes_to_add, nodes_to_rm = self.apply(st, _dag, _pkg)
            rst[0] &= succ_
            rst[1] += nodes_to_add
            rst[2] += nodes_to_rm

        self.tsfs_state.update_grp_history()
        return rst

    def _apply_partition_size(self, G, PKG, part_size_in_B):
        groups = self.tsfs_state.sorted_all_tensor_groups()
        for grp in groups:
            grp_info = self.tsfs_state.parse_tensor_group_info(grp)
            old_partition_num = grp_info["part_num"]
            new_part_num = math.ceil(float(grp_info['size']) / part_size_in_B)
            if old_partition_num == new_part_num:
                continue
            SingleLogger().info("Group {}, size = {:.3f} MB, {} ==> {} partitions".format(
                grp, grp_info['size'] / (1024 * 1024), old_partition_num, new_part_num))
            self._tensor_partition(G, PKG, grp, new_part_num)

    def _tensor_defusion(self, _dag, _pkg: PKGraph, u, loc):
        if self.opt.comm_backend == "BYTEPS":
            raise NotImplementedError()
        ### need to use original local DFG to parse dependency info
        local_dfg = self.opt.clct.dag

        all_infos = self._wrap_parse_all_info(u)
        sorted_tensor_ids = sorted([int(n) for n in all_infos[2].split("+")])
        ids = [sorted_tensor_ids[:loc], sorted_tensor_ids[loc:]]
        rawnames = ["+".join([str(n) for n in _ids]) for _ids in ids]

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        all_pid = sorted(self.opt.clct.all_pid())
        for _pid in all_pid:
            _pid_sync = self._wrap_gen_long_name(_pid, all_infos[1], all_infos[2], all_infos[3], all_infos[4])
            
            ### add new node: sync
            new_part_names, part_times = self._defuse_name_avg(_dag, _pid_sync, rawnames)
            for _name, _time in zip(new_part_names, part_times):
                nodes_to_add.append((_name, {"avg": _time}))

            ### Handle BW->Sync edges
            ### each sync may have multiple input bw nodes
            def remap_fused_bw_nodes(_node):
                if "+" in self.opt.cst_md_mng.strategy2model and _pkg is not None:
                    # has XLA fusion, needs to get fused node names from PKG
                    if _node in _pkg.nodename2fusednode:
                        return _pkg.nodename2fusednode[_node]
                    else:
                        return _node
                else:
                    return _node

            def parse_bw_dependency(_node, edges_to_add, edges_to_rm, _old_node):
                _info = self._wrap_parse_all_info(_node)
                for tensor_id in _info[2].split("+"):
                    op_type_and_name = "{}.{}".format(_info[1], tensor_id)
                    _bw_set = set([remap_fused_bw_nodes(gen_long_name(_info[0], n, None)) for n, _ in local_dfg.in_edges(op_type_and_name)])
                    edges_to_add += [(_bw, _node) for _bw in _bw_set]
                    edges_to_rm += [(_bw, _old_node) for _bw in _bw_set]
            
            parse_bw_dependency(new_part_names[0], edges_to_add, edges_to_rm, _pid_sync)
            parse_bw_dependency(new_part_names[1], edges_to_add, edges_to_rm, _pid_sync)
            nodes_to_rm.add(_pid_sync)

            prev_part_names = new_part_names
            edges_to_handel = []
            for _next in _dag.successors(_pid_sync):
                edges_to_handel.append((prev_part_names, _pid_sync, _next))

            while len(edges_to_handel) > 0:
                prev_part_names, _prev, _cur = edges_to_handel.pop(0)
                edges_to_rm += [(_prev, _cur)]
                nodes_to_rm.add(_cur)
                ### add new node
                new_part_names, part_times = self._defuse_name_avg(_dag, _cur, rawnames)
                for _name, _time in zip(new_part_names, part_times):
                    nodes_to_add.append((_name, {"avg": _time}))

                ### add new edges
                edges_to_add += [(prev_part_names[0], new_part_names[0]),
                                 (prev_part_names[1], new_part_names[1])]

                ### add successors to handle
                _all_infos = [
                    self._wrap_parse_all_info(_prev),
                    self._wrap_parse_all_info(_cur)]

                if ("Sync" in _prev and _all_infos[0][0] != _all_infos[1][0]) or \
                        ("RECV" in _cur and "SEND" not in _prev):
                    ### avoid repeatedly add following edges,
                    #   *  Sync -> other GPUs tensor ops
                    #   *  Sync ->Memcpy -> RECV
                    pass
                else:
                    last_comm = False
                    for succ_ in _dag.successors(_cur):
                        if "Comm" in succ_:
                            edges_to_handel.append((new_part_names, _cur, succ_))
                        elif "UPDATE_" in succ_:
                            last_comm = True
                            break
                        else:
                            raise ValueError("Invalide succ {} of {}".format(succ_, _cur))
                    
                    ### For last communication tensor (MEMCOPY)
                    if last_comm:
                        update_list = []
                        for succ_ in _dag.successors(_cur):
                            assert "UPDATE_" in succ_
                            update_list.append(succ_)
                            edges_to_rm.append((_cur, succ_))    
                        for _update_op in update_list:
                            edges_to_add += [(new_part_names[0], _update_op), (new_part_names[1], _update_op)]
                        # def parse_update_dependency(_node, edges_to_add, edges_to_rm, _old_node):
                        #     _info = self._wrap_parse_all_info(_node)
                        #     for tensor_id in _info[2].split("+"):
                        #         op_type_and_name = "{}.{}".format(_info[1], tensor_id)
                        #         _update_list = [gen_long_name(_info[0], n, None) for n in local_dfg.successors(op_type_and_name)]
                        #         edges_to_add += [(_node, _update_op) for _update_op in _update_list]
                        #         edges_to_rm += [(_old_node, _update_op) for _update_op in _update_list]
                        #         for n in _update_list:
                        #             if n == "host1.rank7->UPDATE_.DistributedGradientDescentOptimizer_Allreduce/truediv_78":
                        #                 print(tensor_id)
                        #                 print([(_node, _update_op) for _update_op in _update_list])
                        #                 print([(_old_node, _update_op) for _update_op in _update_list])
                        #                 assert n in _dag.nodes()
                        
                        # parse_update_dependency(new_part_names[0], edges_to_add, edges_to_rm, _cur)
                        # parse_update_dependency(new_part_names[1], edges_to_add, edges_to_rm, _cur)
        
        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)

        forbidden_list = [all_infos[2]]
        self._check_dag_not_contain(_dag, forbidden_list)

        for n, _ in nodes_to_add:
            self.tsfs_state.update_tensor2grp(n)
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

    def _tensor_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # SingleLogger().info("Fusing Tensor {} & {}.".format(u_, v_))
        ### All infos including [pid, op_cat, op_name, sub_op, suffix]
        pair_all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        
        tensor_group_infos = [
            self.tsfs_state.parse_tensor_group_info(pair_all_infos[0][2]),
            self.tsfs_state.parse_tensor_group_info(pair_all_infos[1][2])]

        if self.opt.comm_backend == "NCCL":
            assert pair_all_infos[0][0] == pair_all_infos[1][0] and \
                pair_all_infos[0][1] == pair_all_infos[1][1] and \
                pair_all_infos[0][3] == pair_all_infos[1][3], (u_, v_)

            edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm = self._nccl_tensor_fusion_impl(
                _dag, _pkg, u_, v_, pair_all_infos, tensor_group_infos
            )
        elif self.opt.comm_backend == "BYTEPS":
            assert pair_all_infos[0][0].split("::")[0] == pair_all_infos[1][0].split("::")[0] and \
                pair_all_infos[0][1] == pair_all_infos[1][1] and \
                pair_all_infos[0][3] == pair_all_infos[1][3], (u_, v_)

            edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm = self._ps_tensor_fusion_impl(
                _dag, _pkg, u_, v_, pair_all_infos, tensor_group_infos
            )
        else:
            raise ValueError()

        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)

        forbidden_list = [_info[2] for _info in pair_all_infos]
        try:
            self._check_dag_not_contain(_dag, forbidden_list, tensor_group_infos)
        except:
            import code
            code.interact(local=locals())

        for n, _ in nodes_to_add:
            self.tsfs_state.update_tensor2grp(n)
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

    def _nccl_tensor_fusion_impl(self, _dag, _pkg: PKGraph,
        u_, v_, pair_all_infos, tensor_group_infos):
        if self.opt.comm_backend == "NCCL":
            ### select the base_idx, according to loopNum
            if tensor_group_infos[0]["sliceNum"] >= tensor_group_infos[1]["sliceNum"] and \
                    tensor_group_infos[0]["loopNum"] >= tensor_group_infos[1]["loopNum"] and \
                    tensor_group_infos[0]["channelNum"] >= tensor_group_infos[1]["channelNum"]:
                base_idx = 0
            elif tensor_group_infos[1]["sliceNum"] >= tensor_group_infos[0]["sliceNum"] and \
                tensor_group_infos[1]["loopNum"] >= tensor_group_infos[0]["loopNum"] and \
                tensor_group_infos[1]["channelNum"] >= tensor_group_infos[0]["channelNum"]:
                base_idx = 1
            else:
                raise ValueError("Invalid tensor group info for {} and {}: {}".format(
                    u_, v_, tensor_group_infos))
        else:
            base_idx = 0

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        all_pid = sorted(self.opt.clct.all_pid())
        for _pid in all_pid:
            _pid_sync = [
                self._wrap_gen_long_name(_pid, pair_all_infos[0][1], pair_all_infos[0][2], pair_all_infos[0][3], pair_all_infos[0][4]),
                self._wrap_gen_long_name(_pid, pair_all_infos[1][1], pair_all_infos[1][2], pair_all_infos[1][3], pair_all_infos[1][4])]

            ### add new node: sync
            new_fused_name, fused_time = self._nccl_concat_name_avg(_dag, _pid_sync[0], _pid_sync[1], base_idx=base_idx)
            nodes_to_add.append((new_fused_name, {"avg": fused_time}))
            # if self.cord_pid == _pid:
            #     print("add node {}".format(new_fused_name))

            ### Handle BW->Sync edges
            ### each sync may have multiple input bw nodes
            for _sync in _pid_sync:
                _in_bw = list(_dag.predecessors(_sync))
                edges_to_add += [(_bw, new_fused_name) for _bw in _in_bw]
                edges_to_rm += [(_bw, _sync) for _bw in _in_bw]     
                nodes_to_rm.add(_sync)

            prev_fused_name = new_fused_name

            def fused_with(_node, _idx=0):
                _all_info = self._wrap_parse_all_info(_node)
                assert _all_info[2] == pair_all_infos[_idx][2], (_all_info, pair_all_infos[_idx][2])
                return self._wrap_gen_long_name(_all_info[0], _all_info[1],
                            pair_all_infos[1-_idx][2], _all_info[3], _all_info[4])

            ### edges_to_handel, each element is a triple of 
            ### (prev_fused_name, prev_name, cur_name)

            edges_to_handel = []
            for to_fuse_u in _dag.successors(_pid_sync[base_idx]):
                edges_to_handel.append((prev_fused_name, _pid_sync[base_idx], to_fuse_u))

            while len(edges_to_handel) > 0:
                to_fuses = [None, None]
                prev_names = [None, None]
                prev_fused_name, prev_name_, to_fuse_ = edges_to_handel.pop(0)
                to_fuses[base_idx] = to_fuse_
                to_fuses[1-base_idx] = fused_with(to_fuse_, base_idx)
                prev_names[base_idx] = prev_name_
                prev_names[1-base_idx] = fused_with(prev_name_, base_idx)
                edges_to_rm += [(prev_names[0], to_fuses[0]), (prev_names[1], to_fuses[1])]
                nodes_to_rm.update((to_fuses[0], to_fuses[1]))

                ### add new node and edge
                new_fused_name, fused_time = self._nccl_concat_name_avg(
                    _dag, to_fuses[0], to_fuses[1], base_idx)
                nodes_to_add.append((new_fused_name, {"avg": fused_time}))
                edges_to_add.append((prev_fused_name, new_fused_name))

                # if self.cord_pid == _pid:
                #     print("add node {}".format(new_fused_name))
                #     print("add edge {} {}".format(prev_fused_name, new_fused_name))

                _all_infos = [
                    self._wrap_parse_all_info(prev_fused_name),
                    self._wrap_parse_all_info(new_fused_name)]
                if ("Sync" in prev_fused_name and _all_infos[0][0] != _all_infos[1][0]) or \
                        ("RECV" in new_fused_name and "SEND" not in prev_fused_name):
                    ### avoid repeatedly add following edges,
                    #   *  Sync -> other GPUs tensor ops
                    #   *  Sync -> MEMCPY -> RECV
                    pass
                else:
                    last_comm = False
                    for succ_ in _dag.successors(to_fuses[base_idx]):
                        if "Comm" in succ_:
                            edges_to_handel.append((new_fused_name, to_fuses[base_idx], succ_))
                        elif "UPDATE_" in succ_:
                            last_comm = True
                            break
                        else:
                            raise ValueError("Invalide succ {} of {}".format(
                                succ_, to_fuses[base_idx]))
                    
                    ### For last communication tensor (MEMCOPY)
                    if last_comm:
                        update_list = []
                        for succ_ in _dag.successors(to_fuses[0]):
                            assert "UPDATE_" in succ_
                            update_list.append(succ_)
                            edges_to_rm.append((to_fuses[0], succ_))
                        for succ_ in _dag.successors(to_fuses[1]):
                            assert "UPDATE_" in succ_
                            update_list.append(succ_)
                            edges_to_rm.append((to_fuses[1], succ_))
                        for _update_op in update_list:
                            edges_to_add.append((new_fused_name, _update_op))      

        return edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm
    
    def _ps_tensor_fusion_impl(self, _dag, _pkg: PKGraph,
        u_, v_, pair_all_infos, tensor_group_infos, new_part_num=None):
        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        processed_nodes = set()

        entry_sub_ops = [PS_COMM_OPS.PUSH_REQ, PS_COMP_OPS.COPY_FIRST]

        tensor_name_u, tensor_name_v = pair_all_infos[0][2], pair_all_infos[1][2]
        part_num_u, part_num_v = tensor_group_infos[0]["part_num"], tensor_group_infos[1]["part_num"]
        new_part_num = max(part_num_u, part_num_v) if new_part_num is None else new_part_num
        new_tensor_size = tensor_group_infos[0]["size"] + tensor_group_infos[1]["size"]
        new_partition_size = new_tensor_size / new_part_num

        fused_tensor_name = self._gen_fused_tensor_name(tensor_name_u, tensor_name_v)
        self.tsfs_state.ps_new_tensor_group(fused_tensor_name, new_tensor_size, new_part_num)

        def __update_node_attr(fused_u_v, _pid, sub_op, __dag):
            fused_time = self.predict_comm_time(new_partition_size, _pid, sub_op)
            attr_dict = {}
            for attr_ in set(__dag.nodes[fused_u_v[0]].keys()).union(set(__dag.nodes[fused_u_v[1]].keys())):
                if attr_ == "avg":
                    attr_dict["avg"] = fused_time
                else:
                    # attr_dict[attr_] = (__dag.nodes[__node].get(attr_, 0) * part_num_u \
                    #     + __dag.nodes[fused_v].get(attr_, 0) * part_num_v) / new_part_num
                    attr_dict[attr_] = max(__dag.nodes[fused_u_v[0]].get(attr_, 0), 
                        __dag.nodes[fused_u_v[1]].get(attr_, 0))
            return attr_dict

        for node in _dag.nodes():
            if "Comm" not in node:
                continue
            node_all_info = self._wrap_parse_all_info(node)
            if not (node_all_info[1] == "Comm" and node_all_info[2] == tensor_name_u and node_all_info[3] in entry_sub_ops):
                continue
            if node_all_info[3] == PS_COMM_OPS.PUSH_REQ:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(node)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(node)
            
            if part_id != '0':
                continue
            
            edges_to_process = []
            cur_node_copy = self._gen_analogous_name(node, new_part_id=0, new_tensor_name=tensor_name_v)
            if node_all_info[3] == PS_COMP_OPS.COPY_FIRST:
                ### No bw predecessors
                for succ_ in _dag.successors(node):
                    edge_u = (node, succ_)
                    edge_v = (cur_node_copy,
                        self._gen_analogous_name(succ_, new_part_id=0, new_tensor_name=tensor_name_v))
                    edges_to_process.append((edge_u, edge_v))
            else:  
                for pred_ in _dag.predecessors(node):
                    assert "BW" in pred_
                    edge_u = (pred_, node)
                    edge_v = (pred_, cur_node_copy)
                    edges_to_process.append((edge_u, edge_v))
                for pred_ in _dag.predecessors(cur_node_copy):
                    assert "BW" in pred_
                    edge_u = (pred_, node)
                    edge_v = (pred_, cur_node_copy)
                    edges_to_process.append((edge_u, edge_v))
            
            while len(edges_to_process) > 0:
                edge_u, edge_v = edges_to_process.pop(0)
                prev_node, cur_node = edge_u
                # print("[TSFS] EDGE", prev_node, cur_node)
                _pair_all_infos = [
                    self._wrap_parse_all_info(prev_node),
                    self._wrap_parse_all_info(cur_node)]
                
                ### remove edges related to node u
                for part_id in range(part_num_u):
                    _edge_u = (
                        self._gen_analogous_name(edge_u[0], new_part_id=part_id),
                        self._gen_analogous_name(edge_u[1], new_part_id=part_id)
                    )
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_rm.add(_edge_u[0])
                    if edge_u[1] not in processed_nodes and not "UPDATE_" in edge_u[1]:
                        nodes_to_rm.add(_edge_u[1])
                    edges_to_rm.append(_edge_u)
                    # print("DELETE_", edge_u[0], edge_u[1], _edge_u[0], _edge_u[1])
                
                for part_id in range(part_num_v):
                    _edge_v = (
                        self._gen_analogous_name(edge_v[0], new_part_id=part_id),
                        self._gen_analogous_name(edge_v[1], new_part_id=part_id)
                    )
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_rm.add(_edge_v[0])
                    if edge_u[1] not in processed_nodes and not "UPDATE_" in edge_u[1]:
                        nodes_to_rm.add(_edge_v[1])
                    edges_to_rm.append(_edge_v)
                    # print("DELETE_", edge_v[0], edge_v[1], _edge_v[0], _edge_v[1])

                ### Parse node attributes  
                if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                    prev_attr_dict = __update_node_attr(
                        (edge_u[0], edge_v[0]), _pair_all_infos[0][0], _pair_all_infos[0][3], _dag)
                else:
                    prev_attr_dict = {}
                if edge_u[1] not in processed_nodes and not "UPDATE_" in edge_u[1]:
                    attr_dict = __update_node_attr(
                        (edge_u[1], edge_v[1]), _pair_all_infos[1][0], _pair_all_infos[1][3], _dag)
                else:
                    attr_dict = {}
                ### Add new edges
                ### Need to add edges, basically, copy
                for part_id in range(new_part_num):
                    new_edge = (
                        self._gen_analogous_name(prev_node, new_part_id=part_id,
                            new_tensor_name=fused_tensor_name, create_node=True),
                        self._gen_analogous_name(cur_node, new_part_id=part_id,
                            new_tensor_name=fused_tensor_name, create_node=True)
                    )
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_add.append((new_edge[0], prev_attr_dict))
                    if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                        nodes_to_add.append((new_edge[1], attr_dict))
                    edges_to_add.append(new_edge)
                    # print("ADD", prev_node, cur_node, new_edge[0], new_edge[1])

                if edge_u[1] not in processed_nodes and not "UPDATE_" in edge_u[1]:
                    ### Hanle suceesors
                    processed_nodes.add(edge_u[1])
                    is_last_comm = False
                    for succ_ in _dag.successors(edge_u[1]):
                        if "UPDATE_" in succ_:
                            is_last_comm = True
                            break
                        _edge_u = (edge_u[1], succ_)
                        _edge_v = (edge_v[1],
                            self._gen_analogous_name(succ_, new_part_id=0, new_tensor_name=tensor_name_v))
                        edges_to_process.append((_edge_u, _edge_v))

                    if is_last_comm:
                        for update_op in _dag.successors(edge_u[1]):
                            assert "UPDATE_" in update_op
                            _edge_u = (edge_u[1], update_op)
                            _edge_v = (edge_v[1], update_op)
                            edges_to_process.append((_edge_u, _edge_v))
                        for update_op in _dag.successors(edge_v[1]):
                            assert "UPDATE_" in update_op
                            _edge_u = (edge_u[1], update_op)
                            _edge_v = (edge_v[1], update_op)
                            edges_to_process.append((_edge_u, _edge_v))
        return edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm

    def _tensor_partition(self, _dag, _pkg: PKGraph, tensor_grp_name, k_star, verbose=True):
        ### return all info including (pid, op_cat, op_name, sub_op, suffix)
        grp_info = self.tsfs_state.parse_tensor_group_info(tensor_grp_name)
        
        entry_sub_ops = [PS_COMM_OPS.PUSH_REQ, PS_COMP_OPS.COPY_FIRST]

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        nodes_to_update = {}
        processed_nodes = set()

        # all_pid = sorted(self.opt.clct.all_pid())
        old_partition_num = grp_info["part_num"]
        if k_star == old_partition_num:
            SingleLogger().debug("[Tensor Partition] do nothing with the same partition number for {}".format(tensor_grp_name))
            return False, None, None
        if verbose:
            SingleLogger().info("Partition tensor {} from {} to {} pieces".format(tensor_grp_name, old_partition_num, k_star))
        
        new_part_num = k_star
        new_partition_size = grp_info["size"] / new_part_num
        # print("[TSFS] From partition: ", old_parititons, "to", new_partitions)
        
        def __update_node_attr(__node, __all_info, __dag):
            ### Update node attributes
            sub_op = __all_info[3]
            _pid = __all_info[0]
            fused_time = self.predict_comm_time(new_partition_size, _pid, sub_op)
            attr_dict = {}
            for attr_ in __dag.nodes[__node]:
                if attr_ == "avg":
                    attr_dict["avg"] = fused_time
                else:
                    # attr_dict[attr_] = __dag.nodes[__node][attr_] * old_partition_num / new_part_num
                    attr_dict[attr_] = __dag.nodes[__node][attr_]
            return attr_dict

        for node in _dag.nodes():
            if "Comm" not in node:
                continue
            node_all_info = self._wrap_parse_all_info(node)
            if not (node_all_info[1] == "Comm" and node_all_info[2] == tensor_grp_name and node_all_info[3] in entry_sub_ops):
                continue
            if node_all_info[3] == PS_COMM_OPS.PUSH_REQ:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(node)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(node)
            
            if part_id != '0':
                continue

            # print("\n[TSFS] ENTRY", node)
            
            if node_all_info[3] == PS_COMP_OPS.COPY_FIRST:
                ### No bw predecessors
                edges_to_process = [(node, succ_) for succ_ in _dag.successors(node)]
            else:
                edges_to_process = [(bw_op, node) for bw_op in _dag.predecessors(node)]
            
            while len(edges_to_process) > 0:
                prev_node, cur_node = edges_to_process.pop(0)
                # print("[TSFS] EDGE", prev_node, cur_node)
                _pair_all_infos = [
                    self._wrap_parse_all_info(prev_node),
                    self._wrap_parse_all_info(cur_node)]
                
                ### Different partititons of the same tensor+sub_op share the same attributes
                if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                    prev_attr_dict = __update_node_attr(prev_node, _pair_all_infos[0], _dag)
                else:
                    prev_attr_dict = {}
                if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                    attr_dict = __update_node_attr(cur_node, _pair_all_infos[1], _dag)
                else:
                    attr_dict = {}

                for part_id in range(new_part_num):
                    if part_id < (old_partition_num - 1):
                        ### only need to update node attributeds
                        prev_node_copy = self._gen_analogous_name(prev_node, new_part_id=part_id, create_node=False)
                        cur_node_copy = self._gen_analogous_name(cur_node, new_part_id=part_id, create_node=False)
                        if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                            nodes_to_update[prev_node_copy] = prev_attr_dict
                        if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                            nodes_to_update[cur_node_copy] = attr_dict
                    else:
                        ### new_part_num > old_partition_num
                        ### Need to add edges, basically, copy
                        prev_node_copy = self._gen_analogous_name(prev_node, new_part_id=part_id, create_node=True)
                        cur_node_copy = self._gen_analogous_name(cur_node, new_part_id=part_id, create_node=True)
                        if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                            nodes_to_add.append((prev_node_copy, prev_attr_dict))
                        if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                            nodes_to_add.append((cur_node_copy, attr_dict))
                        edges_to_add.append((prev_node_copy, cur_node_copy))
                        # print("ADD_", prev_node_copy, cur_node_copy)
                
                if new_part_num < old_partition_num:
                    ### Need to delete edges
                    for part_id in range(new_part_num, old_partition_num):
                        prev_node_copy = self._gen_analogous_name(prev_node, new_part_id=part_id, create_node=False)
                        cur_node_copy = self._gen_analogous_name(cur_node, new_part_id=part_id, create_node=False)
                        if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                            nodes_to_rm.add(prev_node_copy)
                        if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                            nodes_to_rm.add(cur_node_copy)
                        edges_to_rm.append((prev_node_copy, cur_node_copy))
                        # print("DELETE_", prev_node_copy, cur_node_copy)

                if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                    ### Hanle suceesors
                    processed_nodes.add(cur_node)
                    edges_to_process += [(cur_node, succ_) for succ_ in _dag.successors(cur_node)]

        self.tsfs_state.update_part_num(tensor_grp_name, new_part_num)
        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)
        nx.set_node_attributes(_dag, nodes_to_update)

        return True, nodes_to_add, nodes_to_rm

    def _from_tensor_name2rawname(self, tensor_name):
        return "Comm.{}".format(tensor_name)

    def _wrap_parse_all_info(self, n):
        if "Comm" not in n and "+" in n:
            return self._wrap_parse_all_info(n.split("+")[0])
        pid, std_name, cat, suffix = parse_allinfo_from_name(n)
        name_splits = std_name.split(".")
        if len(name_splits) == 3:
            op_cat, op_name, sub_op = name_splits
        elif len(name_splits) == 2:
            op_cat, op_name = name_splits
            sub_op = None
        else:
            print(n)
            raise
        return pid, op_cat, op_name, sub_op, suffix
    
    def _wrap_gen_long_name(self, pid, op_cat, op_name, sub_op, suffix):
        return gen_long_name(pid, "{}.{}.{}".format(op_cat, op_name, sub_op), suffix)
    
    def _gen_fused_tensor_name(self, tensor_u, tensor_v):
        sorted_tensor_ids = sorted([int(id_str) for id_str in tensor_u.split("+") + tensor_v.split("+")])
        return "+".join([str(id) for id in sorted_tensor_ids])

    def _wrap_query_server_id(self, tensor_name, part_id):
        tensor_name_w_part = self.opt.clct.byteps_graph._gen_partitioned_name(tensor_name, part_id)
        if tensor_name_w_part in self.opt.clct.byteps_graph.tensor_part2server:
            return self.opt.clct.byteps_graph.tensor_part2server[tensor_name_w_part]
        ### For new tensors (fused tensors) or new part_id, we need to decide the new mapping to servers
        grp_info = self.tsfs_state.parse_tensor_group_info(tensor_name)
        if "server_id" not in grp_info:
            grp_info["server_id"] = {}
        if part_id not in grp_info["server_id"]:
            ### Not cached
            involved_server = list(self.opt.clct.byteps_graph.tensor_part2server.values())
            server_cnt = collections.Counter(involved_server)
            grp_info["server_id"][part_id] = (sorted(list(server_cnt.items()), key = lambda x: x[1]))[0][0]
        return grp_info["server_id"][part_id]

    def _gen_analogous_name(self, origin_name, new_part_id=None, new_tensor_name=None, create_node=False):
            __all_info = self._wrap_parse_all_info(origin_name)
            if __all_info[1] in ["BW", "UPDATE_", "FW"]:
                return origin_name
            assert __all_info[1] == "Comm", origin_name
            if __all_info[3] in PS_COMM_OPS_SETS:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(origin_name)
                part_id = part_id if new_part_id is None else new_part_id
                if new_tensor_name is not None:
                    tensor_name = new_tensor_name
                if new_part_id is not None or new_tensor_name is not None:
                    new_server_name = self._wrap_query_server_id(tensor_name, part_id)
                    if "server" in source:
                        source = new_server_name
                    else:
                        target = new_server_name
                comm_key = (source, target, tensor_name, sub_op, str(part_id))
                return self.opt.clct.byteps_graph.gen_comm_full_name(comm_key)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(origin_name)
                part_id = part_id if new_part_id is None else new_part_id
                if new_tensor_name is not None:
                    tensor_name = new_tensor_name
                if new_part_id is not None or new_tensor_name is not None:
                    server_id = self._wrap_query_server_id(tensor_name, part_id)
                if create_node:
                    ### Create a node that never occurs before, e.g., fused tensor or new partition id
                    # In this case, we use the default tid
                    tid = self.tsfs_state.wrap_parse_ps_server_tid(server_id, tensor_name, sub_op, part_id, ref_tid=0)
                elif new_part_id is not None or new_tensor_name is not None:
                    tid = self.tsfs_state.wrap_parse_ps_server_tid(server_id, tensor_name, sub_op, part_id)
                comp_key = (server_id, tensor_name, sub_op, tid, str(part_id))
                return self.opt.clct.byteps_graph.gen_comp_full_name(comp_key, sum_index=sum_index)

    def _check_dag_not_contain(self, _dag, forbidden_list, tensor_group_infos=None):
        for n in _dag.nodes():
            if "Comm" not in n:
                continue
            for f in forbidden_list:
                if "Comm.{}.".format(f) in n:
                    for grp_info in tensor_group_infos:
                        print(grp_info)
                    raise ValueError("{} is still in the dag: node {}".format(f, n))

    def _nccl_concat_name_avg(self, _dag, u_, v_, base_idx=0):
        ''' Concate u_ and v_ into a new tensor name and calculate the new tensor's time
        * NOTE: u_/v_ may not exist in _dag, since two tensors needed to be fused may have 
        * different loopid, channelid, and sliceid
        '''
        pair_all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        new_raw_name = self._gen_fused_tensor_name(pair_all_infos[0][2], pair_all_infos[1][2])
        new_name = self._wrap_gen_long_name(pair_all_infos[0][0], pair_all_infos[0][1], new_raw_name, pair_all_infos[0][3], pair_all_infos[0][4])
        self.tsfs_state.nccl_new_tensor_grp(new_raw_name, pair_all_infos[0][2], pair_all_infos[1][2], refer_grp=pair_all_infos[base_idx][2])

        if pair_all_infos[0][3] in ["SEND", "RECV"]:
            grp_infos = [
                self.tsfs_state.parse_tensor_group_info(pair_all_infos[0][2]),
                self.tsfs_state.parse_tensor_group_info(pair_all_infos[1][2])]
            sizes = [grp_info["size"] for grp_info in grp_infos]
            fused_time = self.predict_comm_time(
                (sizes[0] + sizes[1])/grp_infos[base_idx]["part_num"], pair_all_infos[0][0], pair_all_infos[0][3])
        elif pair_all_infos[0][3] in ["QUEUE", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
            fused_time = FUSION_RATIO * max(_dag.nodes[u_]["avg"], _dag.nodes[v_]["avg"])
        elif pair_all_infos[0][3] in ["Sync"]:
            if IGNORE_SYNC:
                fused_time = 0
            else:
                fused_time = FUSION_RATIO * max(_dag.nodes[u_]["avg"], _dag.nodes[v_]["avg"])
        else:
            raise ValueError("Unrecognized sub op name {} ({})".format(
                pair_all_infos[0][3], u_))
        return new_name, fused_time
    
    def _update_bw2tensor(self, comm_node, _dag):
        tensor_name = parse_op_name(comm_node)
        for bw_op in _dag.predecessors(comm_node):
            bw_op_std_name = parse_rawname(bw_op)
            if bw_op_std_name not in self.bw2tensor:
                self.bw2tensor[bw_op_std_name] = set([tensor_name])
            self.bw2tensor[bw_op_std_name].add(tensor_name)
    
    def _defuse_name_avg(self, _dag, u, rawnames):
        all_infos = self._wrap_parse_all_info(u)
        new_part_names = [self._wrap_gen_long_name(
            all_infos[0], all_infos[1], n, all_infos[3], all_infos[4]) for n in rawnames]
        grp_infos = [self.tsfs_state.parse_tensor_group_info(n, ref=all_infos[2]) for n in rawnames]

        part_times = []
        for idx, new_name in enumerate(new_part_names):
            if all_infos[3] in ["SEND", "RECV"]:
                part_time = self.predict_comm_time(
                    grp_infos[idx]["size"]/grp_infos[idx]["part_num"], all_infos[0], all_infos[3])
            elif all_infos[3] in ["QUEUE", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
                part_time = _dag.nodes[u]["avg"] / FUSION_RATIO
            elif all_infos[3] in ["Sync"]:
                part_time = _dag.nodes[u]["avg"] / FUSION_RATIO if not IGNORE_SYNC else 0
            else:
                raise ValueError("Unrecognized sub op name {} ({})".format(
                    all_infos[3], u))
            part_times.append(part_time)
        return new_part_names, part_times
        
    def predict_comm_time(self, _size, _pid, _sub_op):
        if self.opt.comm_backend == "BYTEPS" and _sub_op in [PS_COMM_OPS.PUSH_REQ, PS_COMM_OPS.PULL_RES]:
            return predict_ps_inter_comm_time(_size, is_push=(_sub_op==PS_COMM_OPS.PUSH_REQ))
            ### 20210827_01: Previous method using coarse grained profiled push_pull time
            # inter_node_time = predict_ps_inter_comm_time(_size)
            # for other_sub_op in CONSTANT_SUB_OPS:
            #     if other_sub_op in PS_COMP_OPS_SETS:
            #         __pid = "server_" + _pid[_pid.find("server_") + len("server_"):].split("::")[0]
            #     else:
            #         __pid = _pid
            #     inter_node_time -= self.predict_comm_time(_size, __pid, other_sub_op)
            # return inter_node_time / 2

        _pid = self._pid_used_for_cost_model(_pid)
        if _sub_op == "RECV" and _sub_op not in self.pid_to_cm[_pid]:
            params = self.pid_to_cm[_pid]["SEND"]["param"]
        else:
            params = self.pid_to_cm[_pid][_sub_op]["param"]
        if params is None:
            return 0
        return func_tensor_size_to_time(_size, *(params[0]))

    def _pid_used_for_cost_model(self, _pid):
        if "server_" in _pid and "_t" in _pid:
            return _pid.split("_t")[0]
        elif "worker_" in _pid and "server_" in _pid:
            source, target = _pid.split("::")
            if "worker_" in source:
                return _pid
            else:
                return target + "::" + source
        else:
            return _pid

    def _fit_tsfs_cost_model(self, no_suffix_time):
        '''
        no_suffix_time: a dict, where 
            key is the tensor long name without suffix, and we can parse sub_op from key
            value is a list of communication time
        '''
        self.pid_to_cm = {}
        ### Collect average communication time and tensor size
        for name_no_suffix in no_suffix_time.keys():
            all_info = self._wrap_parse_all_info(name_no_suffix)
            _pid = self._pid_used_for_cost_model(all_info[0])
            if _pid not in self.pid_to_cm:
                self.pid_to_cm[_pid] = {}
            if all_info[3] not in self.pid_to_cm[_pid]:
                self.pid_to_cm[_pid][all_info[3]] = {"data":[], "param": None}
            grp_info = self.tsfs_state.parse_tensor_group_info(all_info[2])
            _size = grp_info["size"] / grp_info["part_num"]

            avg_list = [_avg for _avg in no_suffix_time[name_no_suffix] if _avg > 0]
            if len(avg_list) == 0:
                continue
            _avg = sum(avg_list) / len(avg_list)
            # use median instead of average
            # _avg = sorted(avg_list)[int(len(avg_list)/2)]
            if _avg > 0:
                self.pid_to_cm[_pid][all_info[3]]["data"].append([_size, _avg])
            else:
                raise ValueError("Avg for {} is {} < 0".format(name_no_suffix, _avg))
        
        ### Fit the cost model for each GPU/pid/device
        SingleLogger().info("Fit the cost model for each GPU...")
        fit_data_save_dir = args_.fit_data_save_dir
        for pid in sorted(self.pid_to_cm.keys()):
            for sub_op in sorted(self.pid_to_cm[pid].keys()):
                data_param_dict = self.pid_to_cm[pid][sub_op]
                if len(data_param_dict["data"]) == 0:
                    continue
                
                all_data = np.array(data_param_dict["data"])
                if fit_data_save_dir is not None:
                    np.savetxt(os.path.join(fit_data_save_dir, "{}_{}.txt".format(pid, sub_op)), all_data)
                all_data = all_data.T
                ### data shape = (n_dim, n_samples), there are two dimensions: size and avg
                n_dim, n_samples = all_data.shape

                fit_flag = True
                if self.opt.comm_backend == "BYTEPS":
                    if sub_op in CONSTANT_SUB_OPS:
                        ### For those sub ops, we assume that tensor size has little effect on the execution time 
                        popt = [0, np.median(all_data[1])]
                        data_param_dict["param"] = [popt, None]
                        SingleLogger().debug(" - Tensor Fusion CM for {} {} with a constant value {:.3f} ms".format(pid, sub_op, popt[1]))
                        fit_flag = False
                    # else:        
                    #     SingleLogger().info(" - Tensor Fusion CM for {} {} is ignored".format(pid, sub_op))
                
                if fit_flag:
                    try_cnt = 0
                    while True:
                        train_idx = np.random.choice(n_samples, int(TRAIN_PERCENT * n_samples), replace=False)
                        mask = np.zeros(n_samples, dtype=bool)
                        mask[train_idx] = True
                        train_data = all_data[:, mask]
                        test_data = all_data[:, ~mask]
                        popt, pcov = curve_fit(func_tensor_size_to_time, train_data[0], train_data[1],
                                            bounds=((0, 0), (np.inf, np.inf)), p0=(1, 1), maxfev=100000)
                        pred_ = func_tensor_size_to_time(test_data[0], *popt)
                        mape = 100 * np.average(np.abs(pred_ - test_data[1]) / test_data[1])  
                        if mape < 60:
                            SingleLogger().debug(" - Tensor Fusion CM for {} {}: {} % "
                                "({} training data, {} test data)".format(pid, sub_op, mape, train_data.shape[1], test_data.shape[1]))
                            data_param_dict["param"] = [popt, pcov]
                            data_param_dict["data"] = None
                            break
                        elif try_cnt < n_samples:
                            try_cnt += 1
                        else:
                            if IGNORE_LARGE_ERROR:
                                SingleLogger().debug(" - Tensor Fusion CM for {} {}: {} % "
                                    "({} training data, {} test data)".format(pid, sub_op, mape, train_data.shape[1], test_data.shape[1]))
                                data_param_dict["param"] = [popt, pcov]
                                data_param_dict["data"] = None
                                break
                            # import code
                            # code.interatct(local=locals())
                            SingleLogger().warn(" - Fail to fit a linear Tensor Fusion CM "
                                "for {} {} after {} times mape > 60, ".format(pid, sub_op, try_cnt))
                            SingleLogger().debug("data: {}".format(str(all_data)))
                            break
    
    def _tensor_level_send_recv_cm(self):
        # self.pid_to_cm[all_info[0]] = {
        #                 "SEND": {"data":[], "param": None},
        #                 "RECV": {"data":[], "param": None}}
        send_slope_list = []
        recv_slope_list = []
        send_bias_list = []
        recv_bias_list = []

        if self.opt.comm_backend == "NCCL":
            send_sub_op = "SEND"
            recv_sub_op = "RECV"
        elif self.opt.comm_backend == "BYTEPS":
            send_sub_op = PS_COMM_OPS.PUSH_REQ
            recv_sub_op = PS_COMM_OPS.PULL_REQ
        else:
            raise
        for _dict in self.pid_to_cm.values():
            if send_sub_op in _dict and _dict[send_sub_op]["param"] is not None:
                send_slope_list.append(_dict[send_sub_op]["param"][0][0])
                send_bias_list.append(_dict[send_sub_op]["param"][0][1])
            if recv_sub_op in _dict and _dict[recv_sub_op]["param"] is not None:
                recv_slope_list.append(_dict[recv_sub_op]["param"][0][0])
                recv_bias_list.append(_dict[recv_sub_op]["param"][0][1])
        
        self.send_cm = (np.average(send_bias_list), np.average(send_slope_list))
        self.recv_cm = (np.average(recv_bias_list), np.average(recv_slope_list))
        assert self.send_cm[0] > 0 and self.recv_cm[0] > 0

    def load_init_ckpt(self, G_prime=None):
        ''' 
        G_prime: Other cost model may initialize the DFG, init DFG based on that
        '''
        init_ckpt_path = os.path.join(self.ckpt_dir, "init_ckpt_tensor_fusion.pickle")
        init_ckpt_state_path = os.path.join(self.ckpt_dir, "init_ckpt_tensor_fusion_state.pickle")
        re_load = False
        if os.path.isfile(init_ckpt_path) and os.path.isfile(init_ckpt_state_path):
            try:
                with open(init_ckpt_path, "rb") as f:
                    G, PKG, trajectory, self.pid_to_cm, self.bw2tensor = pickle.load(f)
                self.tsfs_state.load(init_ckpt_state_path)
                if self.tsfs_state.num_grp is not None:
                    SingleLogger().info("Initialzed the graph with {} tensor group(s) ...".format(self.tsfs_state.num_grp))
                SingleLogger().info("Reading init state from cache.")
            except:
                re_load = True
        else:
            re_load = True
        
        # re_load = True
        # G = self.dag.copy() if G_prime is None else G_prime.copy()
        # PKG = None
        # comm_node = "server_2::worker_0->Comm.210.PULL_RES~>server_2::worker_0::0"
        # comm_node1 = "server_2::worker_0->Comm.210+211.PULL_RES~>server_2::worker_0::0"
        # print(list(G.successors(comm_node)))
        # self._tensor_fusion(G, PKG, "Comm.210", "Comm.211")
        # print(list(G.successors(comm_node1)))
        # raise

        if re_load or args_.test_ts_group_num is not None:
            G = self.dag.copy() if G_prime is None else G_prime.copy()
            PKG = None

            self.bw2tensor = {}
            if self.opt.comm_backend == "NCCL":
                trajectory = self._nccl_load_init_ckpt(G, PKG)
            elif self.opt.comm_backend == "BYTEPS":
                trajectory = self._ps_load_init_ckpt(G, PKG)
            else:
                raise ValueError()

            ### Update nodes avg with tensor fusion cost model
            rst = []
            for node in G.nodes():
                if "Comm" not in node:
                    continue
                _pid, _, op_name, sub_op, _ = self._wrap_parse_all_info(node)
                if self.opt.comm_backend == "NCCL":
                    if IGNORE_SYNC and "Sync" in node:
                        G.nodes[node]["avg"] = 0
                        continue
                    elif sub_op not in ["SEND", "RECV"]:
                        continue

                grp_info = self.tsfs_state.parse_tensor_group_info(op_name)
                tensor_size = int(grp_info["size"] / grp_info["part_num"])
                tensor_time = self.predict_comm_time(tensor_size, _pid, sub_op)
                if ("PUSH_REQ" in node or "PULL_RES" in node) and G.nodes[node]["avg"] > 0.1:
                    rst.append([tensor_size, G.nodes[node]["avg"], tensor_time])
                    SingleLogger().debug("{}: size={}B, t_profile={:.3f}ms, t_predict={:.3f}ms".format(
                        node.split("->Comm.")[1], tensor_size, G.nodes[node]["avg"], tensor_time))
                G.nodes[node]["avg"] = tensor_time
            rst = np.array(rst)
            rst.dump(os.path.join(ROOT_PATH, "tsfs_profile_time_vs_predict.txt"))

            if args_.test_ts_group_num is None:
                ### By default, fuse tensors that connected to the same BW ops.
                #   But when testing tensor group numbers, do not fuse those tensors
                self.init_fuse_tensors(G, PKG)
                ### Cache
                self.opt.clct.byteps_graph._dump_cache()
                with open(init_ckpt_path, "wb") as f:
                    pickle.dump([G, PKG, trajectory, self.pid_to_cm,
                        self.bw2tensor], f)
                self.tsfs_state.dump(init_ckpt_state_path)
                SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))
            else:
                SingleLogger().info(bcolors.CGREEN + "Test tensor fusion group: use {} tensor groups".format(
                    args_.test_ts_group_num) + bcolors.ENDC)
        
        # self._tensor_level_send_recv_cm()
        self.dump_tensor_grp_mapping(_file_name="tensor_fusion_grp_mapping_init.json")
        self.dump_tensor_partitions(_file_name="tensor_partition_init.txt")

        ### Test
        # self._tensor_partition(G, PKG, "200", 6)
        # self._tensor_fusion(G, PKG,
        #     "worker0::server0->Comm.200.PUSH_REQ~>worker0::server0::0", 
        #     "worker0::server0->Comm.201.PUSH_REQ~>worker0::server0::0") 

        return G, PKG, trajectory

    def _nccl_load_init_ckpt(self, _dag, PKG):
        ''' Modify _dag in place
        '''
        ### Check uncontinuous tensor groups, and split them for futher tensor fusion
        ### Since Horovod 0.21.0 Tensor Group API requires tensors in a group to be continuous
        trajectory = []
        source_nodes = [n for n in _dag.nodes() if "Comm" in n]
        no_suffix_time = {}
        for n in tqdm(source_nodes, total=len(source_nodes)):
            ### return all info including (pid, op_cat, op_name, sub_op, suffix)
            all_info = self._wrap_parse_all_info(n)

            ### Check whether tensors in one group are continuous
            # If NOT, spit the group into multiple tensor groups
            # ASSUMPTION: each host has the same tensor fusion pattern
            if self.cord_pid == all_info[0] and "Sync" == all_info[3]:
                self.tsfs_state.update_tensor2grp(n)
                self._update_bw2tensor(n, _dag)
                ### check uncontinuous fusions
                groups = []
                cur_group = []
                prev_id = None
                op_name = all_info[2]
                for idx, tensor_id in enumerate(op_name.split("+")):
                    if prev_id is None or prev_id + 1 == int(tensor_id):
                        ### Continuous tensors, directly add it to the group
                        cur_group.append(int(tensor_id))
                    else:
                        ### Non-continuous tensors, create a new tensor group
                        assert prev_id + 1 < int(tensor_id), (n, prev_id, tensor_id)
                        groups.append(cur_group)
                        cur_group = [int(tensor_id)]
                    prev_id = int(tensor_id)
                if len(cur_group) > 0:
                    groups.append(cur_group)
                if len(groups) > 1:
                    ### tensor ids are not continuous, divide this into multiple group
                    SingleLogger().info("Non-continuous tensor group {}, groups: {}".format(n, groups))
                    name_to_split = n
                    for idx in range(len(groups) - 1):
                        ### except for the last group
                        grp = groups[idx]
                        trajectory.append(("--", name_to_split, len(grp)))
                        _info = self._wrap_parse_all_info(name_to_split)
                        name_to_split = self._wrap_gen_long_name(
                            _info[0], _info[1], "+".join(_info[2].split("+")[len(grp):]), _info[3], _info[4])
                        # print(grp, trajectory[-1])
            
            ### Collect data to fit tensor fusion cost model,
            if all_info[3] == "SEND" or all_info[3] == "RECV":
                name_no_suffix = self._wrap_gen_long_name(all_info[0], all_info[1], all_info[2], all_info[3], None)
                if name_no_suffix not in no_suffix_time:
                    no_suffix_time[name_no_suffix] = []
                no_suffix_time[name_no_suffix].append(_dag.nodes[n]["avg"])

        self._fit_tsfs_cost_model(no_suffix_time=no_suffix_time)

        ### applying strategies to make tensors in each group are continuous
        SingleLogger().info("Applying initialized strategies...")
        for st in tqdm(trajectory, total=len(trajectory)):
            self.apply(st, _dag, PKG)
        
        ### If we only search the optimal group number, apply the initialized
        # tensor fusion group number, e.g., 1
        if self.tsfs_state.num_grp is not None:
            SingleLogger().info("Initialzed the graph with {} tensor group(s) ...".format(self.tsfs_state.num_grp))
            self._apply_grp_num(_dag, PKG, self.tsfs_state.num_grp)
        
        return trajectory
    
    def _ps_load_init_ckpt(self, _dag, PKG):
        trajectory = []
        no_suffix_time = {}
        source_nodes = [n for n in _dag.nodes() if "Comm" in n]
        only_worker_name = None
        for comm_node in tqdm(source_nodes, total=len(source_nodes)):
            ### return all info including (pid, op_cat, op_name, sub_op, suffix)
            all_info = self._wrap_parse_all_info(comm_node)

            if only_worker_name is None:
                only_worker_name = all_info[0].split("::")[0]
            if only_worker_name == all_info[0].split("::")[0]:
                self.tsfs_state.update_tensor2grp(comm_node)
                self._update_bw2tensor(comm_node, _dag)
            self.tsfs_state.parse_tensor_group_info(all_info[2])
            name_no_suffix = self._wrap_gen_long_name(all_info[0], all_info[1], all_info[2], all_info[3], None)
            if name_no_suffix not in no_suffix_time:
                no_suffix_time[name_no_suffix] = []
            no_suffix_time[name_no_suffix].append(_dag.nodes[comm_node]["avg"])
        
        self._fit_tsfs_cost_model(no_suffix_time=no_suffix_time)
        return trajectory

    def checkpoint(self):
        ### Need to cache byteps graph state, e.g. comp_ops_tid
        self.opt.clct.byteps_graph._dump_cache()
        self.tsfs_state.dump()
        
    def load_ckpt(self):
        ### load byteps graph state
        byteps_cache_path = self.opt.clct.pm.search(FileName.BYTEPS_CACHE)
        assert byteps_cache_path is not None
        SingleLogger().info("Inited BytePS graph helper from cache: {}.".format(byteps_cache_path))
        self.opt.clct.byteps_graph.init_from_cache(byteps_cache_path)
        
        self.tsfs_state.load()

    def init_fuse_tensors(self, G, PKG):
        tensors_to_fuse = [sorted(list(tensor_name_set)) for tensor_name_set in self.bw2tensor.values() 
            if len(tensor_name_set) > 1]
        for tensors in tensors_to_fuse:
            SingleLogger().info("[TSFS] Init: fuse {}".format(tensors))
            prev_tensor = self._from_tensor_name2rawname(tensors[0])
            for idx in range(1, len(tensors)):
                tensor_v = self._from_tensor_name2rawname(tensors[idx])
                _, nodes_introduced, _ = self._tensor_fusion(G, PKG, prev_tensor, tensor_v)
                prev_tensor = self._from_tensor_name2rawname(parse_op_name(nodes_introduced[0]))
    
    def dump_tensor_grp_mapping(self, _file_name=None):
        self.tsfs_state.dump_tensor_grp_mapping(_file_name)
    
    def dump_tensor_partitions(self, _file_name=None):
        if self.enable_partition:
            self.tsfs_state.dump_tensor_partitions(_file_name)
    
    def if_fusion_better(self, op_name_u, op_name_v, dp_state, _dag, no_theorem=False):
        ''' Decide if fusing two tensors (u, v) is better based on some heuristics
            Return a tuple (is_fuse, k_star), 
            where `is_fuse` denotes whether to fuse the two tensors
            `k_star` is the optimal partition number
        '''
        if no_theorem:
            ### Via partially replay
            old_graph_state = self.tsfs_state.copy()
            G_star = _dag.copy()
            fused_tensor_name = self._gen_fused_tensor_name(op_name_u, op_name_v)
            self._tensor_fusion(G_star, None, self._from_tensor_name2rawname(op_name_u),
                self._from_tensor_name2rawname(op_name_v))
            k_star_fuse, t_sync_fuse = self.best_partition_partial_replay(fused_tensor_name, G_star, no_throrem=True)
            self.tsfs_state = old_graph_state

            G_star = _dag.copy()
            k_star_null, t_sync_null = self.best_partition_partial_replay(op_name_v, G_star, no_throrem=True)


            # op_name_v = '180'
            # partitions = self.opt.clct.byteps_graph.partition_dict.get(op_name_v, ['0'])
            # grp_info = self.tsfs_state.parse_tensor_group_info(op_name_v)
            # print("old part num : {}".format(grp_info["part_num"]))
            # print("origin partitions {}".format(partitions))
            # name2mapping_fn = lambda x: x if "Comm.{}.".format(op_name_v) in x else "None"

            # G_star = _dag.copy()
            # self._tensor_partition(G_star, None, op_name_v, 2)
            # full_time_fuse = self.opt.evaluate(G_star, name2mapping_fn=name2mapping_fn, _path="full_replay_part2.json")[0]
            # partial_time_origin = self.estimate_time_related_to_comm(
            #     [op_name_v], G_star,
            #     "partial_replay_part2.json")
            # print(full_time_fuse, partial_time_origin)

            # G_star = _dag.copy()
            # self._tensor_partition(G_star, None, op_name_v, 3)
            # full_time_fuse = self.opt.evaluate(G_star, name2mapping_fn=name2mapping_fn, _path="full_replay_part3.json")[0]
            # partial_time_origin = self.estimate_time_related_to_comm(
            #     [op_name_v], G_star,
            #     "partial_replay_part3.json")
            # print(full_time_fuse, partial_time_origin)

            # G_star = _dag.copy()
            # self._tensor_partition(G_star, None, op_name_v, 4)
            # full_time_fuse = self.opt.evaluate(G_star, name2mapping_fn=name2mapping_fn, _path="full_replay_part4.json")[0]
            # partial_time_origin = self.estimate_time_related_to_comm(
            #     [op_name_v], G_star,
            #     "partial_replay_part4.json")
            # print(full_time_fuse, partial_time_origin)

            # G_star = _dag.copy()
            # self._tensor_partition(G_star, None, op_name_v, 5)
            # full_time_fuse = self.opt.evaluate(G_star, name2mapping_fn=name2mapping_fn, _path="full_replay_part5.json")[0]
            # partial_time_origin = self.estimate_time_related_to_comm(
            #     [op_name_v], G_star,
            #     "partial_replay_part5.json")
            # print(full_time_fuse, partial_time_origin)

            # raise


            if t_sync_fuse < t_sync_null:
                return True, k_star_fuse, t_sync_fuse, t_sync_null
            else:
                return False, k_star_null, t_sync_fuse, t_sync_null


        end_comm_time_u = dp_state.q_e[-2]  ### q_{n-1}^e
        end_comp_time_v = dp_state.p_e[-1]  ### p_{n}^e
        tensor_size_u = dp_state.q_m[-2]
        tensor_size_v = dp_state.q_m[-1]

        ### Previous method, totally reply on theorem
        # k_star_fuse, t_sync_fuse = self.best_partition_naive(tensor_size_u + tensor_size_v)
        # k_star_null, t_sync_null = self.best_partition_naive(tensor_size_v)

        # SingleLogger().info("xx \t k_star_fuse \t t_sync_fuse \t k_star_null \t t_sync_null")
        # SingleLogger().info("naive: \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(k_star_fuse, t_sync_fuse, k_star_null, t_sync_null))

        ### Via partially replay

        ### Fusion
        G_star = _dag.copy()
        old_graph_state = self.tsfs_state.copy()
        fused_tensor_name = self._gen_fused_tensor_name(op_name_u, op_name_v)
        self._tensor_fusion(G_star, None, self._from_tensor_name2rawname(op_name_u),
            self._from_tensor_name2rawname(op_name_v))
        k_star_fuse, t_sync_fuse = self.best_partition_partial_replay(fused_tensor_name, G_star)
        
        # validate
        self._tensor_partition(G_star, None, fused_tensor_name, k_star_fuse, verbose=False)
        self.tsfs_state = old_graph_state

        partial_time_fuse = self.estimate_time_related_to_comm(
            [fused_tensor_name], G_star,
            "partial_replay_fuse.json")
        full_time_fuse = self.opt.evaluate(G_star, _path="full_replay_fuse.json")[0]

        ### No Fusion
        G_star = _dag.copy()
        k_star_null, t_sync_null = self.best_partition_partial_replay(op_name_v, G_star)

        # validate
        partial_time_origin = self.estimate_time_related_to_comm(
            [op_name_u, op_name_v], _dag,
            "partial_replay_origin.json")
        full_time_origin = self.opt.evaluate(_dag, _path="full_replay_origin.json")[0]
        print("Origin: partial: {:.3f}, full {:.3f}".format(partial_time_origin, full_time_origin))
        print("Fuse: partial: {:.3f}, full {:.3f}".format(partial_time_fuse, full_time_fuse))
        
        # SingleLogger().info("Partial: \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(k_star_fuse, t_sync_fuse, k_star_null, t_sync_null))
        # SingleLogger().info("q_n-1^e: {:.3f}, p_n^e: {:.3f}".format(end_comm_time_u, end_comp_time_v))

        if end_comm_time_u > end_comp_time_v + t_sync_fuse - t_sync_null:
            ### Fusion is better
            return True, k_star_fuse, t_sync_fuse, t_sync_null
        else:
            return False, k_star_null, t_sync_fuse, t_sync_null
    
    def best_partition_naive(self, tensor_size, ref_time=None):
        ''' Find the the optimal partition number given tensor size
            Return a tuple (k_star, sync_time), where `k_star` is the optimal patition number
            and `sync_time` is the estimated execution time of this tensor if it's
            partitioned into k_star pieces 
        '''
        if not self.enable_partition:
            return 1, ref_time

        self.aggr_cm = (0, 0)
        if self.send_cm[1] > self.recv_cm[1]:
            ### Recv is faster
            k_star = math.sqrt(
                (tensor_size * (self.recv_cm[1] + self.aggr_cm[1])) / self.send_cm[0])
        else:
            ### Send is faster
            k_star = math.sqrt(
                (tensor_size * (self.send_cm[1] + self.aggr_cm[1])) / self.recv_cm[0])
        
        k_star = max(round(k_star), 1)
        partition_size = tensor_size / k_star
        t_send = func_tensor_size_to_time(partition_size, self.send_cm[1], self.send_cm[0])
        t_recv = func_tensor_size_to_time(partition_size, self.recv_cm[1], self.recv_cm[0])
        t_aggr = func_tensor_size_to_time(partition_size, self.aggr_cm[1], self.aggr_cm[0])

        if self.send_cm[1] > self.recv_cm[1]:
            sync_time = k_star * t_send + t_aggr + t_recv
        else:
            sync_time = t_send + t_aggr + k_star * t_recv
        return k_star, sync_time

    def best_partition_partial_replay(self, tensor_name, _dag, no_throrem=False):
        ### `tensor_name` must be a tensor in `dag`
        grp_info = self.tsfs_state.parse_tensor_group_info(tensor_name)
        MAX_TEST_TENSOR_GROUP_TIME = 5
        sampled_partition_num = list(range(1, 2 * grp_info["part_num"] + 1))
        if len(sampled_partition_num) > MAX_TEST_TENSOR_GROUP_TIME:
            sampled_partition_num = random.sample(sampled_partition_num, MAX_TEST_TENSOR_GROUP_TIME)

        k_star = None
        comm_time_star = None
        for partition_num in sampled_partition_num:
            old_graph_state = self.tsfs_state.copy()
            G_star = _dag.copy()
            self._tensor_partition(G_star, None, tensor_name, partition_num, verbose=False)
            self.tsfs_state = old_graph_state
            if no_throrem:
                comm_time = self.opt.estimate_time_related_to_comm([tensor_name], G_star)
            else:
                comm_time = self.opt.estimate_comm_time_via_replay(tensor_name, G_star)
            if k_star is None or comm_time < comm_time_star:
                k_star = partition_num
                comm_time_star = comm_time
        return k_star, comm_time_star
    
    def get_current_comm_from_unfused_bw(self, _node):
        assert "+" not in _node
        assert "BW" in _node
        local_dag = self.opt.clct.dag
        pid, std_name, cat, suffix = parse_allinfo_from_name(_node)
        unfused_comm_nodes = [n for n in local_dag.successors(std_name) 
                                if parse_cat_from_name(n) == CatName.COMM.value]
        current_comm_nodes = set()
        for comm_node in unfused_comm_nodes:
            tensor_id = int(parse_op_name(comm_node))
            group_name = self.tsfs_state.parse_tensor_cur_grp(tensor_id)
            current_comm_nodes.add(self._wrap_gen_long_name(pid, CatName.COMM.value, group_name, "Sync", suffix))
        return current_comm_nodes
    
    def get_current_comm_from_unfused_update(self, _node):
        assert "+" not in _node
        assert "UPDATE" in _node
        local_dag = self.opt.clct.dag
        pid, std_name, cat, suffix = parse_allinfo_from_name(_node)
        unfused_comm_nodes = [n for n in local_dag.predecessors(std_name) 
                                if parse_cat_from_name(n) == CatName.COMM.value]
        current_comm_nodes = set()
        for comm_node in unfused_comm_nodes:
            tensor_id = int(parse_op_name(comm_node))
            group_name = self.tsfs_state.parse_tensor_cur_grp(tensor_id)
            current_comm_nodes.add(self._wrap_gen_long_name(pid, CatName.COMM.value, group_name, "MEMCPY_OUT_FUSION_BUFFER", suffix))
        return current_comm_nodes

    def update_tensor2server(self):
        self.tsfs_state.update_tensor2server()