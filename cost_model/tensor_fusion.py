import networkx as nx
import time
import os
import pickle
import numpy as np
import math
import ujson as json
from tqdm import tqdm, trange
from scipy.optimize import curve_fit

import arg_utils
from .base import _BaseGraphPass
from trace_utils import *
from cost_model._xla.pk_graph import PKGraph

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

def func_tensor_size_to_time(s, k, b):
    return k * s + b


class TensorFusionGraphPass(_BaseGraphPass):
    ''' This is a cost model for HOROVOD tensor fusion
    '''

    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["++", "--"]
        self.tensor_group_info = {}
        self.cur_tensor2group = {}

        self.cord_pid = self.opt.cord_pid

        for n in self.dag.nodes():
            if "Comm" in n:
                ### Assume each host has the same tensor fusion pattern
                if self.cord_pid in n:
                    self._update_tensor2grp(n)
                    
        self.ckpt_path = os.path.join(ROOT_PATH, "ckpt_tensor_fusion.pickle")
        self.cache_change = []

        self.num_grp = 1 if args_.search_ts_group_num else None
        self.history_num_grp = {}
        if self.num_grp is not None:
            SingleLogger().info("Search the optimal number of tensor fusion groups")
        else:
            SingleLogger().info("Search the optimal tensor fusion strategies")

        ### Store the cost model for tensor fusion/partition
        self.pid_to_cm = None

        ### Tensor level cost model, a tuple where the first element is the slope
        self.send_cm = None
        self.recv_cm = None

        self.enable_partition = True
        self.enable_defusion = True

    def _ret_weight_num_grp(self, num_grp):
        if num_grp not in self.history_num_grp:
            return 1
        else:
            return 1.0 / float(self.history_num_grp[num_grp] + 1)
        
    def _init_search_space_num_grp(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []
        
        assert self.num_grp is not None
        if self.num_grp == 1:
            if (self.num_grp + 1) not in self.history_num_grp:
                search_space.append(("++", self.num_grp + 1, None))
                weights.append(self._ret_weight_num_grp(self.num_grp + 1))
        else:
            if (self.num_grp + 1) not in self.history_num_grp:
                search_space.append(("++", self.num_grp + 1, None))
                weights.append(self._ret_weight_num_grp(self.num_grp + 1))
            if (self.num_grp - 1) not in self.history_num_grp:
                search_space.append(("++", self.num_grp - 1, None))
                weights.append(self._ret_weight_num_grp(self.num_grp - 1))
        return search_space, weights

    def _apply_grp_num(self, _dag, _pkg, num_grp):
        ''' The method used in horovod to construct tensor groups
        * here `l` is the list of tensors, `n` is the number of groups
        ```
            d, r = divmod(len(l), n)
            return [l[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]
        ```
        '''
        tensor_num = len(self.cur_tensor2group)
        num_per_grp, _ = divmod(tensor_num, num_grp)
        
        trajectory = []
        residual = []
        groups = sorted(list(set(self.cur_tensor2group.values())), key=lambda grp_id: int(grp_id.split("+")[0]))
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

        SingleLogger().info("From {} groups to {} groups, apply {} strategies in totoal ...".format(self.num_grp, num_grp, len(trajectory)))
        rst = [True, [], []]
        for st in tqdm(trajectory, total=len(trajectory)):
            # print(st)
            succ_, nodes_to_add, nodes_to_rm = self.apply(st, _dag, _pkg)
            rst[0] &= succ_
            rst[1] += nodes_to_add
            rst[2] += nodes_to_rm
        self.cache_change.append(num_grp)
        return rst

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
            group_name = self.cur_tensor2group[tensor_id]
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
            group_name = self.cur_tensor2group[tensor_id]
            current_comm_nodes.add(self._wrap_gen_long_name(pid, CatName.COMM.value, group_name, "MEMCPY_OUT_FUSION_BUFFER", suffix))
        return current_comm_nodes

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        if self.num_grp is not None:
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
                    grp_name = self.cur_tensor2group[target_id]
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

    def _update_tensor2grp(self, n):
        op_name = self._wrap_parse_all_info(n)[2]
        for tensor_id in op_name.split("+"):
            self.cur_tensor2group[int(tensor_id)] = op_name

    def _wrap_parse_all_info(self, n):
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

    def flush(self, is_accept):
        ''' Some strategies may not be accepted,
            * If accepted, change the interal state of this Pass
            * Otherwise, keep it the same
        '''
        if is_accept:
            for n in self.cache_change:
                if isinstance(n, int):
                    self.history_num_grp[self.num_grp] = 1 if self.num_grp not in self.history_num_grp else self.history_num_grp[self.num_grp] + 1
                    self.num_grp = n
                elif isinstance(n, tuple):
                    tensor_grp_name, partitions = n
                    self.tensor_group_info[tensor_grp_name] = partitions
                elif self.cord_pid in n:
                    self._update_tensor2grp(n)
        self.cache_change = []

    def _tensor_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # SingleLogger().info("Fusing Tensor {} & {}.".format(u_, v_))
        pair_all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        
        assert pair_all_infos[0][0] == pair_all_infos[1][0] and \
            pair_all_infos[0][1] == pair_all_infos[1][1] and \
            pair_all_infos[0][3] == pair_all_infos[1][3] and \
            pair_all_infos[0][4] == pair_all_infos[1][4]

        tensor_group_infos = [
            self._parse_tensor_group_info(pair_all_infos[0][2]),
            self._parse_tensor_group_info(pair_all_infos[1][2])]

        if self.opt.comm_backend == "NCCL":
            edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm = self._nccl_tensor_fusion_impl(
                _dag, _pkg, u_, v_, pair_all_infos, tensor_group_infos
            )
        elif self.opt.comm_backend == "BYTEPS":
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
        self._check_dag_not_contain(_dag, forbidden_list, tensor_group_infos)

        self.cache_change += [n for n, _ in nodes_to_add]
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

    def _nccl_tensor_fusion_impl(self, _dag, _pkg: PKGraph,
        u_, v_, pair_all_infos, tensor_group_infos):
        if self.opt.comm_backend == "NCCL":
            ### select the base_idx, according to loopNum
            if tensor_group_infos[0]["sliceNum"] >= tensor_group_infos[1]["sliceNum"] and \
                    tensor_group_infos[0]["loopNum"] >= tensor_group_infos[1]["loopNum"]:
                base_idx = 0
            elif tensor_group_infos[1]["sliceNum"] >= tensor_group_infos[0]["sliceNum"] and \
                tensor_group_infos[1]["loopNum"] >= tensor_group_infos[0]["loopNum"]:
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
            new_fused_name, fused_time = self._concat_name_avg(_dag, _pid_sync[0], _pid_sync[1], base_idx=base_idx)
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
                new_fused_name, fused_time = self._concat_name_avg(
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
        part_num_u, part_num_v = len(tensor_group_infos[0]["partitions"]), len(tensor_group_infos[1]["partitions"])
        new_part_num = max(part_num_u, part_num_v) if new_part_num is None else new_part_num
        new_tensor_size = tensor_group_infos[0]["size"] + tensor_group_infos[1]["size"]
        new_partition_size = new_tensor_size / new_part_num

        sorted_tensor_ids = sorted([int(id_str) for id_str in tensor_name_u.split("+") + tensor_name_v.split("+")])
        fused_tensor_name = "+".join([str(id) for id in sorted_tensor_ids])

        print("[TSFS] From partition num: ", part_num_u, part_num_v, "to", new_part_num)
        
        def __update_node_attr(__node, __all_info, __dag):
            sub_op = __all_info[3]
            _pid = __all_info[0]
            fused_time = self.predict_comm_time(new_partition_size, _pid, sub_op)
            fused_v = self._gen_analogous_name(__node, new_part_id=None, new_tensor_name=tensor_name_v)
            attr_dict = {}
            for attr_ in __dag.nodes[__node]:
                if attr_ == "avg":
                    attr_dict["avg"] = fused_time
                else:
                    attr_dict[attr_] = (__dag.nodes[__node][attr_] * part_num_u + __dag.nodes[fused_v][attr_] * part_num_v)/new_part_num
            return attr_dict

        for node in _dag.nodes():
            node_all_info = self._wrap_parse_all_info(node)
            if not (node_all_info[1] == "Comm" and node_all_info[2] == tensor_name_u and node_all_info[3] in entry_sub_ops):
                continue
            if node_all_info[3] == PS_COMM_OPS.PUSH_REQ:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(node)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(node)
            
            if part_id != '0':
                continue
            
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
                
                ### remove edges related to node u
                for part_id in range(part_num_u):
                    prev_node_copy = self._gen_analogous_name(prev_node, part_id)
                    cur_node_copy = self._gen_analogous_name(cur_node, part_id)
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_rm.add(prev_node_copy)
                    if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                        nodes_to_rm.add(cur_node_copy)
                    edges_to_rm.append((prev_node_copy, cur_node_copy))
                    # print("DELETE_", prev_node, cur_node, prev_node_copy, cur_node_copy)
                
                for part_id in range(part_num_v):
                    prev_node_copy = self._gen_analogous_name(prev_node, part_id, new_tensor_name=tensor_name_v)
                    cur_node_copy = self._gen_analogous_name(cur_node, part_id, new_tensor_name=tensor_name_v)
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_rm.add(prev_node_copy)
                    if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                        nodes_to_rm.add(cur_node_copy)
                    edges_to_rm.append((prev_node_copy, cur_node_copy))
                    # print("DELETE_", prev_node, cur_node, prev_node_copy, cur_node_copy)
                
                if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                    prev_attr_dict = __update_node_attr(prev_node, _pair_all_infos[0], _dag)
                else:
                    prev_attr_dict = {}
                if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                    attr_dict = __update_node_attr(cur_node, _pair_all_infos[1], _dag)
                else:
                    attr_dict = {}
                ### Add new edges
                ### Need to add edges, basically, copy
                for part_id in range(new_part_num):
                    prev_node_copy = self._gen_analogous_name(prev_node, part_id, new_tensor_name=fused_tensor_name)
                    cur_node_copy = self._gen_analogous_name(cur_node, part_id, new_tensor_name=fused_tensor_name)
                    if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                        nodes_to_add.append((prev_node_copy, prev_attr_dict))
                    if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                        nodes_to_add.append((cur_node_copy, attr_dict))
                    edges_to_add.append((prev_node_copy, cur_node_copy))
                    # print("ADD", prev_node, cur_node, prev_node_copy, cur_node_copy)

                if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                    ### Hanle suceesors
                    processed_nodes.add(cur_node)
                    edges_to_process += [(cur_node, succ_) for succ_ in _dag.successors(cur_node)]

        return edges_to_add, edges_to_rm, nodes_to_add, nodes_to_rm

    def _gen_analogous_name(self, origin_name, new_part_id=None, new_tensor_name=None):
            __all_info = self._wrap_parse_all_info(origin_name)
            if __all_info[1] in ["BW", "UPDATE_"]:
                return origin_name
            assert __all_info[1] == "Comm", origin_name
            if __all_info[3] in PS_COMM_OPS_SETS:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(origin_name)
                part_id = part_id if new_part_id is None else new_part_id
                tensor_name = tensor_name if new_tensor_name is None else new_tensor_name
                comm_key = (source, target, tensor_name, sub_op, str(part_id))
                return self.opt.clct.byteps_graph.gen_comm_full_name(comm_key)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(origin_name)
                part_id = part_id if new_part_id is None else new_part_id
                if new_tensor_name is not None:
                    tensor_name = new_tensor_name
                if new_part_id is not None or new_tensor_name is not None:
                    tid = self._wrap_parse_ps_server_tid(server_id, tensor_name, sub_op, part_id)
                comp_key = (server_id, tensor_name, sub_op, tid, str(new_part_id))
                return self.opt.clct.byteps_graph.gen_comp_full_name(comp_key, sum_index=sum_index)

    def _wrap_parse_ps_server_tid(self, server_id, tensor_name, sub_op, part_id):
        comm_key = (server_id, tensor_name, sub_op, str(part_id))
        server_tid = self.tensor_group_info.get(comm_key, None)
        if server_tid is not None:
            return server_tid
        else:
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

    def _tensor_partition(self, _dag, _pkg: PKGraph, comm_op, k_star):
        ### return all info including (pid, op_cat, op_name, sub_op, suffix)
        assert isinstance(k_star, int), k_star
        all_info = self._wrap_parse_all_info(comm_op)
        tensor_grp_name = all_info[2]
        grp_info = self._parse_tensor_group_info(tensor_grp_name)
        
        entry_sub_ops = [PS_COMM_OPS.PUSH_REQ, PS_COMP_OPS.COPY_FIRST]

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        nodes_to_update = {}
        processed_nodes = set()

        # all_pid = sorted(self.opt.clct.all_pid())
        old_parititons = grp_info["partitions"]
        old_partition_num = len(old_parititons)
        if k_star == old_partition_num:
            return
        
        new_partitions = list([str(part_id) for part_id in range(k_star)])
        new_partition_size = grp_info["size"] / k_star
        print("[TSFS] From partition: ", old_parititons, "to", new_partitions)
        
        def __update_node_attr(__node, __all_info, __nodes_to_update, __dag):
            ### Update node attributes
            sub_op = __all_info[3]
            _pid = __all_info[0]
            fused_time = self.predict_comm_time(new_partition_size, _pid, sub_op)
            attr_dict = {}
            for attr_ in __dag.nodes[__node]:
                if attr_ == "avg":
                    attr_dict["avg"] = fused_time
                else:
                    attr_dict[attr_] = __dag.nodes[cur_node][attr_] * old_partition_num / k_star
            
            for part_id in range(k_star):
                comm_op = self._gen_analogous_name(cur_node, part_id)
                __nodes_to_update[comm_op] = attr_dict

        for node in _dag.nodes():
            node_all_info = self._wrap_parse_all_info(node)
            if not (node_all_info[1] == "Comm" and node_all_info[2] == tensor_grp_name and node_all_info[3] in entry_sub_ops):
                continue
            if node_all_info[3] == PS_COMM_OPS.PUSH_REQ:
                source, target, tensor_name, sub_op, part_id = self.opt.clct.byteps_graph.parse_comm_event_name(node)
            else:
                server_id, tid, tensor_name, sub_op, part_id, sum_index = self.opt.clct.byteps_graph.parse_comp_name(node)
            
            if part_id != '0':
                continue

            print("\n[TSFS] ENTRY", node)
            
            if node_all_info[3] == PS_COMP_OPS.COPY_FIRST:
                ### No bw predecessors
                edges_to_process = [(node, succ_) for succ_ in _dag.successors(node)]
            else:
                edges_to_process = [(bw_op, node) for bw_op in _dag.predecessors(node)]
            
            while len(edges_to_process) > 0:
                prev_node, cur_node = edges_to_process.pop(0)
                print("[TSFS] EDGE", prev_node, cur_node)
                _pair_all_infos = [
                    self._wrap_parse_all_info(prev_node),
                    self._wrap_parse_all_info(cur_node)]
                if k_star > old_partition_num:
                    ### Need to add edges, basically, copy
                    for part_id in range(old_partition_num, k_star):
                        prev_node_copy = self._gen_analogous_name(prev_node, part_id)
                        cur_node_copy = self._gen_analogous_name(cur_node, part_id)
                        if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                            nodes_to_add.append(prev_node_copy)
                        if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                            nodes_to_add.append(cur_node_copy)
                        edges_to_add.append((prev_node, cur_node_copy))
                        # print("ADD", prev_node_copy, cur_node_copy)
                elif k_star < old_partition_num:
                    ### Need to delete edges
                    for part_id in range(k_star, old_partition_num):
                        prev_node_copy = self._gen_analogous_name(prev_node, part_id)
                        cur_node_copy = self._gen_analogous_name(cur_node, part_id)
                        if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                            nodes_to_rm.add(prev_node_copy)
                        if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                            nodes_to_rm.add(cur_node_copy)
                        edges_to_rm.append((prev_node, cur_node_copy))
                        # print("DELETE_", prev_node_copy, cur_node_copy)

                if _pair_all_infos[0][3] == PS_COMP_OPS.COPY_FIRST:
                    __update_node_attr(prev_node, _pair_all_infos[0], nodes_to_update, _dag)

                if cur_node not in processed_nodes and not "UPDATE_" in cur_node:
                    __update_node_attr(cur_node, _pair_all_infos[1], nodes_to_update, _dag)

                    ### Hanle suceesors
                    processed_nodes.add(cur_node)
                    edges_to_process += [(cur_node, succ_) for succ_ in _dag.successors(cur_node)]

        self.cache_change.append((tensor_grp_name, new_partitions))
        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)
        nx.set_node_attributes(_dag, nodes_to_update)

        return True, nodes_to_add, nodes_to_rm

    def _check_dag_not_contain(self, _dag, forbidden_list, tensor_group_infos=None):
        for n in _dag.nodes():
            if "Comm" not in n:
                continue
            for f in forbidden_list:
                if "Comm.{}.".format(f) in n:
                    for grp_info in tensor_group_infos:
                        print(grp_info)
                    raise ValueError("{} is still in the dag: node {}".format(f, n))

    def _concat_name_avg(self, _dag, u_, v_, base_idx=0):
        ''' Concate u_ and v_ into a new tensor name and calculate the new tensor's time
        * NOTE: u_/v_ may not exist in _dag, since two tensors needed to be fused may have 
        * different loopid, channelid, and sliceid
        '''
        pair_all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        sorted_tensor_ids = sorted([int(n) for n in pair_all_infos[0][2].split(
            "+") + pair_all_infos[1][2].split("+")])
        new_raw_name = "+".join([str(n) for n in sorted_tensor_ids])
        new_name = self._wrap_gen_long_name(pair_all_infos[0][0], pair_all_infos[0][1], new_raw_name, pair_all_infos[0][3], pair_all_infos[0][4])
        if new_raw_name not in self.tensor_group_info:
            self.tensor_group_info[new_raw_name] = self.tensor_group_info[pair_all_infos[base_idx][2]].copy()
            self.tensor_group_info[new_raw_name]["size"] = \
                self.tensor_group_info[pair_all_infos[0][2]]["size"] + \
                self.tensor_group_info[pair_all_infos[1][2]]["size"]
            # print("add new name {}".format(new_raw_name))

        if pair_all_infos[0][3] in ["SEND", "RECV"]:
            grp_infos = [
                self._parse_tensor_group_info(pair_all_infos[0][2]),
                self._parse_tensor_group_info(pair_all_infos[1][2])]
            sizes = [grp_info["size"] for grp_info in grp_infos]
            fused_time = self.predict_comm_time(
                (sizes[0] + sizes[1])/grp_infos[base_idx]["partNum"], pair_all_infos[0][0], pair_all_infos[0][3])
            # if pair_all_infos[0][0] == "host0.rank3" and "0_0_0_0" in u_:
            #     print("Tensor Fusion Log")
            #     print(" *** {} - total size={}, partnum={}".format(u_, sizes[0], grp_infos[0]["partNum"]))
            #     print(" *** {} - total size={}, partnum={}".format(v_, sizes[1], grp_infos[1]["partNum"]))
            #     print(" *** -> total size={}, partnum={}, size={}, pred={}".format(
            #         self.tensor_group_info[new_raw_name]["size"],
            #         self.tensor_group_info[new_raw_name]["partNum"],
            #         (sizes[0] + sizes[1])/grp_infos[base_idx]["partNum"], fused_time))
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
    
    def _tensor_defusion(self, _dag, _pkg: PKGraph, u, loc):
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

        self.cache_change += [n for n, _ in nodes_to_add]
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

    def _defuse_name_avg(self, _dag, u, rawnames):
        all_infos = self._wrap_parse_all_info(u)
        new_part_names = [self._wrap_gen_long_name(
            all_infos[0], all_infos[1], n, all_infos[3], all_infos[4]) for n in rawnames]
        grp_infos = [self._parse_tensor_group_info(n, ref=all_infos[2]) for n in rawnames]

        part_times = []
        for idx, new_name in enumerate(new_part_names):
            if all_infos[3] in ["SEND", "RECV"]:
                part_time = self.predict_comm_time(
                    grp_infos[idx]["size"]/grp_infos[idx]["partNum"], all_infos[0], all_infos[3])
            elif all_infos[3] in ["QUEUE", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
                part_time = _dag.nodes[u]["avg"] / FUSION_RATIO
            elif all_infos[3] in ["Sync"]:
                part_time = _dag.nodes[u]["avg"] / FUSION_RATIO if not IGNORE_SYNC else 0
            else:
                raise ValueError("Unrecognized sub op name {} ({})".format(
                    all_infos[3], u))
            part_times.append(part_time)
        return new_part_names, part_times

    def _parse_tensor_group_info(self, op_name, ref=None):
        ''' op_name must be sorted '''
        if op_name not in self.tensor_group_info:
            if self.opt.comm_backend == "NCCL":
                if ref is None:
                    chunkNum, sliceNum, channelNum, loopNum = self.opt.clct.nccl_graph.get_IDnum("Comm." + op_name)
                    grp_info = {
                        "chunkNum": chunkNum,
                        "sliceNum": sliceNum,
                        "channelNum": channelNum,
                        "loopNum": loopNum,
                        "partNum": loopNum*channelNum*sliceNum*(chunkNum/2 + 1)
                    }
                else:
                    ### some group may not occur in the traces, ref to other groups
                    grp_info = self._parse_tensor_group_info(ref).copy()
            else:
                grp_info = {
                        "partitions": self.opt.clct.byteps_graph.partition_dict.get(op_name, ['0'])
                    }

            total_size = self._tensor_grp_size(op_name)
            grp_info["size"] = total_size
            self.tensor_group_info[op_name] = grp_info
        return self.tensor_group_info[op_name]

    def _tensor_grp_size(self, op_name):
        return self.meta_info.tensor_grp_size(op_name)
        
    def checkpoint(self):
        try:
            with open(self.ckpt_path, "wb") as f:
                pickle.dump([self.tensor_group_info, self.cur_tensor2group, self.num_grp, self.history_num_grp], f)
        except:
            print(len(self.tensor_group_info), self.history_num_grp)
            raise
        
    def load_ckpt(self):
        if os.path.isfile(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                self.tensor_group_info, self.cur_tensor2group, self.num_grp, self.history_num_grp = pickle.load(f)
        self.cache_change = []

    def predict_comm_time(self, _size, _pid, _sub_op):
        if _sub_op == "RECV" and _sub_op not in self.pid_to_cm[_pid]:
            params = self.pid_to_cm[_pid]["SEND"]["param"]
        else:
            params = self.pid_to_cm[_pid][_sub_op]["param"]
        if params is None:
            return 0
        return func_tensor_size_to_time(_size, *(params[0]))
    
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
            if all_info[0] not in self.pid_to_cm:
                self.pid_to_cm[all_info[0]] = {}
            if all_info[3] not in self.pid_to_cm[all_info[0]]:
                self.pid_to_cm[all_info[0]][all_info[3]] = {"data":[], "param": None}
            grp_info = self._parse_tensor_group_info(all_info[2])
            _size = grp_info["size"] / len(grp_info["partitions"])

            avg_list = [_avg for _avg in no_suffix_time[name_no_suffix] if _avg > 0]
            if len(avg_list) == 0:
                continue
            _avg = sum(avg_list) / len(avg_list)
            ### use median instead of average
            # _avg = sorted(avg_list)[int(len(avg_list)/2)]
            if _avg > 0:
                self.pid_to_cm[all_info[0]][all_info[3]]["data"].append([_size, _avg])
            else:
                raise ValueError("Avg for {} is {} < 0".format(name_no_suffix, _avg))
        
        ### Fit the cost model for each GPU/pid/device
        SingleLogger().info("Fit the cost model for each GPU...")
        for pid in sorted(self.pid_to_cm.keys()):
            for sub_op in sorted(self.pid_to_cm[pid].keys()):
                data_param_dict = self.pid_to_cm[pid][sub_op]
                if len(data_param_dict["data"]) == 0:
                    continue

                ### data shape = (n_dim, n_samples)
                all_data = np.array(data_param_dict["data"]).T
                n_dim, n_samples = all_data.shape
                mask = np.zeros(n_samples, dtype=bool)

                try_cnt = 0
                while True:
                    train_idx = np.random.choice(n_samples, int(TRAIN_PERCENT * n_samples), replace=False)
                    mask[train_idx] = True
                    train_data = all_data[:, mask]
                    test_data = all_data[:, ~mask]
                    popt, pcov = curve_fit(func_tensor_size_to_time, train_data[0], train_data[1],
                                        bounds=((0, 0), (np.inf, np.inf)), p0=(1, 1), maxfev=100000)
                    pred_ = func_tensor_size_to_time(test_data[0], *popt)
                    mape = 100 * np.average(np.abs(pred_ - test_data[1]) / test_data[1])  
                    if mape < 60:
                        SingleLogger().info(" - Tensor Fusion CM for {} {}: {} % "
                            "({} training data, {} test data)".format(pid, sub_op, mape, train_data.shape[1], test_data.shape[1]))
                        data_param_dict["param"] = [popt, pcov]
                        data_param_dict["data"] = None
                        break
                    elif try_cnt < n_samples:
                        try_cnt += 1
                    else:
                        if IGNORE_LARGE_ERROR:
                            data_param_dict["param"] = [popt, pcov]
                            data_param_dict["data"] = None
                            break
                        # import code
                        # code.interatct(local=locals())
                        SingleLogger().warn(" - Fail to fit a linear Tensor Fusion CM "
                            "for {} {} after {} times mape > 60, ".format(pid, sub_op, try_cnt))
                        SingleLogger().debug("data: {}".format(str(all_data)))
                        break

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
                self._update_tensor2grp(n)
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
            self.flush(True)
        
        ### If we only search the optimal group number, apply the initialized
        # tensor fusion group number, e.g., 1
        if self.num_grp is not None:
            SingleLogger().info("Initialzed the graph with {} tensor group(s) ...".format(self.num_grp))
            self._apply_grp_num(_dag, PKG, self.num_grp)
            self.flush(True)
        
        return trajectory
    
    def _ps_load_init_ckpt(self, _dag, PKG):
        trajectory = []
        no_suffix_time = {}
        source_nodes = [n for n in _dag.nodes() if "Comm" in n]
        only_pid = None
        for comm_node in tqdm(source_nodes, total=len(source_nodes)):
            ### return all info including (pid, op_cat, op_name, sub_op, suffix)
            all_info = self._wrap_parse_all_info(comm_node)

            if only_pid is None:
                only_pid = all_info[0]
            if only_pid == all_info[0]:
                self._update_tensor2grp(comm_node)

            self._parse_tensor_group_info(all_info[2])
            name_no_suffix = self._wrap_gen_long_name(all_info[0], all_info[1], all_info[2], all_info[3], None)
            if name_no_suffix not in no_suffix_time:
                no_suffix_time[name_no_suffix] = []
            no_suffix_time[name_no_suffix].append(_dag.nodes[comm_node]["avg"])
        
        self._fit_tsfs_cost_model(no_suffix_time=no_suffix_time)
        return trajectory

    def load_init_ckpt(self, G_prime=None):
        ''' 
        G_prime: Other cost model may initialize the DFG, init DFG based on that
        '''
        init_ckpt_path = os.path.join(ROOT_PATH, "tensor_fusion_init_ckpt.pickle")
        re_load = False
        if os.path.isfile(init_ckpt_path):
            try:
                with open(init_ckpt_path, "rb") as f:
                    G, PKG, trajectory, self.tensor_group_info, self.cur_tensor2group,\
                        self.num_grp, self.history_num_grp, self.pid_to_cm = pickle.load(f)
                if self.num_grp is not None:
                    SingleLogger().info("Initialzed the graph with {} tensor group(s) ...".format(self.num_grp))
                SingleLogger().info("Reading init state from cache.")
            except:
                re_load = True
        else:
            re_load = True
            
        if re_load:
            G = self.dag.copy() if G_prime is None else G_prime.copy()
            PKG = None

            if self.opt.comm_backend == "NCCL":
                trajectory = self._nccl_load_init_ckpt(G, PKG)
            elif self.opt.comm_backend == "BYTEPS":
                trajectory = self._ps_load_init_ckpt(G, PKG)
            else:
                raise ValueError()

            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, trajectory, self.tensor_group_info,
                    self.cur_tensor2group, self.num_grp, self.history_num_grp, self.pid_to_cm], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))
        
        if IGNORE_SYNC and self.opt.comm_backend == "NCCL":
            for n in G.nodes():
                if "Sync" in n:
                    G.nodes[n]["avg"] = 0
        
        # self._tensor_level_send_recv_cm()
        self.dump_tensor_grp_mapping(_file_name="tensor_fusion_grp_mapping_init.json")
        self.cache_change = []

        ### Test
        # self._tensor_partition(G, PKG, "worker0::server0->Comm.200.PUSH_REQ~>worker0::server0::0", 6)
        self._tensor_fusion(G, PKG,
            "worker0::server0->Comm.200.PUSH_REQ~>worker0::server0::0", 
            "worker0::server0->Comm.201.PUSH_REQ~>worker0::server0::0")
        raise

        return G, PKG, trajectory
    
    def dump_tensor_grp_mapping(self, _file_name=None):
        file_name = 'tensor_fusion_grp_mapping.json' if _file_name is None else _file_name

        tensor_ids, tensor_grps = zip(*list(self.cur_tensor2group.items()))
        tensor_grps = set(tensor_grps)
        tensor_ids = set(tensor_ids)
        assert len(tensor_ids) == self.meta_info.gradient_num(), \
            ("incompleted tensor_ids {} : {}".format(sorted(tensor_ids), self.meta_info.gradient_num()))

        with open(os.path.join(ROOT_PATH, file_name), 'w') as f:
            json.dump({"mapping": list(tensor_grps)}, f)
    
    def _tensor_level_send_recv_cm(self):
        # self.pid_to_cm[all_info[0]] = {
        #                 "SEND": {"data":[], "param": None},
        #                 "RECV": {"data":[], "param": None}}
        send_slope_list = []
        recv_slope_list = []
        send_bias_list = []
        recv_bias_list = []
        for _dict in self.pid_to_cm.values():
            if _dict["SEND"]["param"] is not None:
                send_slope_list.append(_dict["SEND"]["param"][0][0])
                send_bias_list.append(_dict["SEND"]["param"][0][1])
            if _dict["RECV"]["param"] is not None:
                recv_slope_list.append(_dict["RECV"]["param"][0][0])
                recv_bias_list.append(_dict["RECV"]["param"][0][1])
        
        self.send_cm = (np.average(send_bias_list), np.average(send_slope_list))
        self.recv_cm = (np.average(recv_bias_list), np.average(recv_slope_list))
        assert self.send_cm[0] > 0 and self.recv_cm[0] > 0
    
    def if_fusion_better(self, op_name_u, op_name_v, dp_state):
        ''' Decide if fusing two tensors (u, v) is better based on some heuristics
            Return a tuple (is_fuse, k_star), 
            where `is_fuse` denotes whether to fuse the two tensors
            `k_star` is the optimal partition number
        '''
        end_comm_time_u = dp_state.q_e[-2]  ### q_{n-1}^e
        end_comp_time_v = dp_state.p_n[-1]  ### p_{n}^e
        tensor_size_u = dp_state.q_m[-2]
        tensor_size_v = dp_state.q_m[-1]
        
        k_star_fuse, t_sync_fuse = self.best_partition(tensor_size_u + tensor_size_v)
        k_star_null, t_sync_null = self.best_partition(tensor_size_v)
        if end_comm_time_u > end_comp_time_v + t_sync_fuse - t_sync_null:
            ### Fusion is better
            return True, k_star_fuse, t_sync_fuse, t_sync_null
        else:
            return False, k_star_null, t_sync_fuse, t_sync_null
    
    def best_partition(self, tensor_size, ref_time=None):
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

