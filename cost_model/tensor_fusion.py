import networkx as nx
import time
import os
import pickle
import numpy as np
import ujson as json
from tqdm import tqdm, trange
from scipy.optimize import curve_fit

import arg_utils
from .base import _BaseGraphPass
from trace_utils import *
from cost_model._xla.pk_graph import PKGraph

args_ = arg_utils.SingleArg().args
ENABLE_PARTITION = True
FUSION_RATIO = 1
TRAIN_PERCENT = 0.9
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

        for n in self.dag.nodes():
            if "Comm" in n:
                ### Assume each host has the same tensor fusion pattern
                if "host0.rank0" in n:
                    self._update_tensor2grp(n)
                    
        self.ckpt_path = os.path.join(ROOT_PATH, "ckpt_tensor_fusion.pickle")
        self.cache_change = []

        self.num_grp = 1
        self.history_num_grp = {}
        if self.num_grp is not None:
            SingleLogger().info("Search the optimal number of tensor fusion groups")
        else:
            SingleLogger().info("Search the optimal tensor fusion strategies")

        ### Store the cost model for tensor fusion/partition
        self.pid_to_cm = None

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
                    to_fuse_u = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                    residual = []
            elif len(residual) + len(tensor_ids) < num_per_grp:
                if len(residual) > 0:
                    to_fuse_u = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                residual += tensor_ids
            else:
                if len(residual) > 0:
                    tensor_name = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(tensor_ids), "Sync", None)
                    trajectory.append(("--", tensor_name, num_per_grp - len(residual)))
                    to_fuse_u = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(residual), "Sync", None)
                    to_fuse_v = self._wrap_gen_long_name(
                        "host0.rank0", "Comm", "+".join(tensor_ids[:(num_per_grp - len(residual))]), "Sync", None)
                    trajectory.append(("++", to_fuse_u, to_fuse_v))
                    tensor_ids = tensor_ids[(num_per_grp - len(residual)):]
                while len(tensor_ids) > num_per_grp:
                    tensor_name = self._wrap_gen_long_name("host0.rank0", "Comm", "+".join(tensor_ids), "Sync", None)
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
        pid, raw_name, cat, suffix = parse_allinfo_from_name(_node)
        unfused_comm_nodes = [n for n in local_dag.successors(raw_name) 
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
        pid, raw_name, cat, suffix = parse_allinfo_from_name(_node)
        unfused_comm_nodes = [n for n in local_dag.predecessors(raw_name) 
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

                if ENABLE_PARTITION:
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
            return self._op_fusion(__dag, __pkg, target, next_)
        elif op == "--":
            return self._op_defusion(__dag, __pkg, target, next_)

    def _update_tensor2grp(self, n):
        rawname = self._wrap_parse_all_info(n)[2]
        for tensor_id in rawname.split("+"):
            self.cur_tensor2group[int(tensor_id)] = rawname

    def _wrap_parse_all_info(self, n):
        pid, raw_name, cat, suffix = parse_allinfo_from_name(n)
        try:
            op_name, layer_name, sub_op = raw_name.split(".")
        except:
            print(n)
            raise
        return pid, op_name, layer_name, sub_op, suffix
    
    def _wrap_gen_long_name(self, pid, op_name, layer_name, sub_op, suffix):
        return gen_long_name(pid, "{}.{}.{}".format(op_name, layer_name, sub_op), suffix)

    def flush(self, is_accept):
        if is_accept:
            for n in self.cache_change:
                if isinstance(n, int):
                    self.history_num_grp[self.num_grp] = 1 if self.num_grp not in self.history_num_grp else self.history_num_grp[self.num_grp] + 1
                    self.num_grp = n
                elif "host0.rank0" in n:
                    self._update_tensor2grp(n)
        self.cache_change = []

    def _op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # SingleLogger().info("Fusing Tensor {} & {}.".format(u_, v_))
        all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        
        ### select the base_idx, according to loopNum
        tensor_group_infos = [
            self._parse_tensor_group_info(all_infos[0][2]),
            self._parse_tensor_group_info(all_infos[1][2])]
        if tensor_group_infos[0]["sliceNum"] >= tensor_group_infos[1]["sliceNum"] and \
                tensor_group_infos[0]["loopNum"] >= tensor_group_infos[1]["loopNum"]:
            base_idx = 0
        elif tensor_group_infos[1]["sliceNum"] >= tensor_group_infos[0]["sliceNum"] and \
            tensor_group_infos[1]["loopNum"] >= tensor_group_infos[0]["loopNum"]:
            base_idx = 1
        else:
            raise ValueError("Invalid tensor group info for {} and {}: {}".format(
                u_, v_, tensor_group_infos))

        assert all_infos[0][0] == all_infos[1][0] and \
            all_infos[0][1] == all_infos[1][1] and \
            all_infos[0][3] == all_infos[1][3] and \
            all_infos[0][4] == all_infos[1][4]

        edges_to_add = []
        edges_to_rm = []
        nodes_to_add = []
        nodes_to_rm = set()
        all_pid = sorted(self.opt.clct.all_pid())
        for _pid in all_pid:
            _pid_sync = [
                self._wrap_gen_long_name(_pid, all_infos[0][1], all_infos[0][2], all_infos[0][3], all_infos[0][4]),
                self._wrap_gen_long_name(_pid, all_infos[1][1], all_infos[1][2], all_infos[1][3], all_infos[1][4])]

            ### add new node: sync
            new_fused_name, fused_time = self._concat_name_avg(_dag, _pid_sync[0], _pid_sync[1], base_idx=base_idx)
            nodes_to_add.append((new_fused_name, {"avg": fused_time}))
            # if "host0.rank0" == _pid:
            #     print("add node {}".format(new_fused_name))

            ### Handle BW->Sync edges
            ### each sync may have multiple input bw nodes
            for _sync in _pid_sync:
                _in_bw = [n1 for n1, _ in _dag.in_edges(_sync)]
                edges_to_add += [(_bw, new_fused_name) for _bw in _in_bw]
                edges_to_rm += [(_bw, _sync) for _bw in _in_bw]     
                nodes_to_rm.add(_sync)

            prev_fused_name = new_fused_name

            def fused_with(_node, _idx=0):
                _all_info = self._wrap_parse_all_info(_node)
                assert _all_info[2] == all_infos[_idx][2], (_all_info, all_infos[_idx][2])
                return self._wrap_gen_long_name(_all_info[0], _all_info[1],
                            all_infos[1-_idx][2], _all_info[3], _all_info[4])

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
                nodes_to_add.append((new_fused_name, {"avg":fused_time}))
                edges_to_add.append((prev_fused_name, new_fused_name))

                # if "host0.rank0" == _pid:
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

        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)

        forbidden_list = [_info[2] for _info in all_infos]
        self._check_dag_not_contain(_dag, forbidden_list, tensor_group_infos)

        self.cache_change += [n for n, _ in nodes_to_add]
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

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
        all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
        sorted_tensor_ids = sorted([int(n) for n in all_infos[0][2].split(
            "+") + all_infos[1][2].split("+")])
        new_raw_name = "+".join([str(n) for n in sorted_tensor_ids])
        new_name = self._wrap_gen_long_name(all_infos[0][0], all_infos[0][1], new_raw_name, all_infos[0][3], all_infos[0][4])
        if new_raw_name not in self.tensor_group_info:
            self.tensor_group_info[new_raw_name] = self.tensor_group_info[all_infos[base_idx][2]].copy()
            self.tensor_group_info[new_raw_name]["size"] = \
                self.tensor_group_info[all_infos[0][2]]["size"] + \
                self.tensor_group_info[all_infos[1][2]]["size"]
            # print("add new name {}".format(new_raw_name))

        if all_infos[0][3] in ["SEND", "RECV"]:
            grp_infos = [
                self._parse_tensor_group_info(all_infos[0][2]),
                self._parse_tensor_group_info(all_infos[1][2])]
            sizes = [grp_info["size"] for grp_info in grp_infos]
            fused_time = self.predict_comm_time(
                (sizes[0] + sizes[1])/grp_infos[base_idx]["partNum"], all_infos[0][0], all_infos[0][3])
            # if all_infos[0][0] == "host0.rank3" and "0_0_0_0" in u_:
            #     print("Tensor Fusion Log")
            #     print(" *** {} - total size={}, partnum={}".format(u_, sizes[0], grp_infos[0]["partNum"]))
            #     print(" *** {} - total size={}, partnum={}".format(v_, sizes[1], grp_infos[1]["partNum"]))
            #     print(" *** -> total size={}, partnum={}, size={}, pred={}".format(
            #         self.tensor_group_info[new_raw_name]["size"],
            #         self.tensor_group_info[new_raw_name]["partNum"],
            #         (sizes[0] + sizes[1])/grp_infos[base_idx]["partNum"], fused_time))
        elif all_infos[0][3] in ["QUEUE", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
            fused_time = FUSION_RATIO * max(_dag.nodes[u_]["avg"], _dag.nodes[v_]["avg"])
        elif all_infos[0][3] in ["Sync"]:
            if IGNORE_SYNC:
                fused_time = 0
            else:
                fused_time = FUSION_RATIO * max(_dag.nodes[u_]["avg"], _dag.nodes[v_]["avg"])
        else:
            raise ValueError("Unrecognized sub op name {} ({})".format(
                all_infos[0][3], u_))
        return new_name, fused_time
    
    def _op_defusion(self, _dag, _pkg: PKGraph, u, loc):
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

    def _parse_tensor_group_info(self, raw_name, ref=None):
        ''' raw_name must be sorted '''
        if raw_name not in self.tensor_group_info:
            if ref is None:
                chunkNum, sliceNum, channelNum, loopNum = self.opt.clct.nccl_graph.get_IDnum("Comm." + raw_name)
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
            total_size = 0
            for tensor_id_str in raw_name.split("+"):
                tensor_id = int(tensor_id_str)
                total_size += self.meta_info.tensor_id2size(tensor_id)
            grp_info["size"] = total_size
            self.tensor_group_info[raw_name] = grp_info
        return self.tensor_group_info[raw_name]

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
    
    def load_init_ckpt(self, G_prime=None):
        ''' Other cost model may initialize the DFG, init DFG based on that
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

            ### Check uncontinuous tensor groups, and split them for futher tensor fusion
            ### Since Horovod 0.21.0 Tensor Group API requires tensors in a group to be continuous
            trajectory = []
            source_nodes = [n for n in G.nodes() if "Comm" in n]
            self.pid_to_cm = {}
            no_suffix_time = {}
            for n in tqdm(source_nodes, total=len(source_nodes)):
                all_info = self._wrap_parse_all_info(n)
                ### Assume each host has the same tensor fusion pattern
                if "host0.rank0" == all_info[0] and "Sync" == all_info[3]:
                    self._update_tensor2grp(n)
                    ### check uncontinuous fusions
                    groups = []
                    cur_group = []
                    prev_id = None
                    rawname = all_info[2]
                    for idx, tensor_id in enumerate(rawname.split("+")):
                        if prev_id is None or prev_id + 1 == int(tensor_id):
                            cur_group.append(int(tensor_id))
                        else:
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
                # TODO (hhp): only collect SEND operators now
                if all_info[3] == "SEND" or all_info[3] == "RECV":
                    name_no_suffix = self._wrap_gen_long_name(all_info[0], all_info[1], all_info[2], all_info[3], None)
                    if name_no_suffix not in no_suffix_time:
                        no_suffix_time[name_no_suffix] = []
                    no_suffix_time[name_no_suffix].append(G.nodes[n]["avg"])

            for n in no_suffix_time.keys():
                all_info = self._wrap_parse_all_info(n)
                if all_info[0] not in self.pid_to_cm:
                    self.pid_to_cm[all_info[0]] = {
                        "SEND": {"data":[], "param": None},
                        "RECV": {"data":[], "param": None}}
                grp_info = self._parse_tensor_group_info(all_info[2])
                # assert grp_info["partNum"] < len(no_suffix_time[n]), (n, grp_info, len(no_suffix_time[n]))
                _size = grp_info["size"] / grp_info["partNum"]

                avg_list = [_avg for _avg in no_suffix_time[n] if _avg > 0]
                if len(avg_list) == 0:
                    continue
                _avg = sum(avg_list) / len(avg_list)
                ### use median instead of average
                # _avg = sorted(avg_list)[int(len(avg_list)/2)]
                if _avg > 0:
                    self.pid_to_cm[all_info[0]][all_info[3]]["data"].append([_size, _avg])
            
            ### Fit the cost model for each GPU
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
                                            bounds=((0, -np.inf), (np.inf, np.inf)), p0=(1, 1), maxfev=100000)
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
                            SingleLogger().warn(" - Fail to fit a linear Tensor Fusion CM "
                                "for {} {} after {} times mape > 60, data: {}".format(pid, sub_op, try_cnt, str(all_data)))
                            break
            ### applying strategies to make tensors in each group are continuous
            SingleLogger().info("Applying initialized strategies...")
            for st in tqdm(trajectory, total=len(trajectory)):
                self.apply(st, G, PKG)
                self.flush(True)
            
            if self.num_grp is not None:
                SingleLogger().info("Initialzed the graph with {} tensor group(s) ...".format(self.num_grp))
                self._apply_grp_num(G, PKG, self.num_grp)
                self.flush(True)

            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, trajectory, self.tensor_group_info,
                    self.cur_tensor2group, self.num_grp, self.history_num_grp, self.pid_to_cm], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))
        
        if IGNORE_SYNC:
            for n in G.nodes():
                if "Sync" in n:
                    G.nodes[n]["avg"] = 0

        self.dump_tensor_grp_mapping(_file_name="tensor_fusion_grp_mapping_init.json")
        self.cache_change = []
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

