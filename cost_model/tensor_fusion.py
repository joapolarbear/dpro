import networkx as nx
import time
import os
import pickle
import numpy as np
import ujson as json

import arg_utils
from .base import _BaseCostModel
from trace_utils import *
from cost_model_xla.pk_graph import PKGraph

args_ = arg_utils.SingleArg().args
FUSION_RATIO = 1
### given a fused tensor, return N possible partition method
# N <= MAX_PARTITION_NUM
MAX_PARTITION_NUM = 5
ROOT_PATH = os.path.join(args_.workspace, ".opt_ws")
class _TensorFusionCM(_BaseCostModel):
    ''' This is a cost model for HOROVOD tensor fusion
    '''

    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["++", "--"]
        self.meta_info = self.opt.clct.para_dict
        self.node_attr_cache = {}
        self.tensor_group_info = {}
        self.cur_tensor2group = {}

        for n in self.dag.nodes():
            if "Comm" in n:
                self._cache_node_attr(n, self.dag.nodes[n])
                ### Assume each host has the same tensor fusion pattern
                if "host0.rank0" in n:
                    self._update_tensor2grp(n)
                    
        self.ckpt_path = os.path.join(ROOT_PATH, "ckpt_tensor_fusion.pickle")
        self.cache_change = []

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
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
                    SingleLogger().info("{} is the last tensor, can not be further fused.".format(to_fuse_u))
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

    def _cache_node_attr(self, n, attrs):
        self.node_attr_cache[n] = attrs
    
    def _get_node_attr(self, n, attr_):
        if attr_ in self.node_attr_cache[n]:
            return self.node_attr_cache[n][attr_]
        else:
            raise ValueError(n, attr_)
    
    def _parse_node_attr(self, _dag, new_name, avg=None):
        ''' Parse the fused tensor attribute corresponding to `new_name` and set _dag
        * If new_name has been cached, directly set _dag with the cached attributes
        * Otherwise, combine the attribution of all original nodes
            * If avg is not given, query the cost model
        Return
        ------
        avg: average time
        '''
        if new_name in self.node_attr_cache:
            nx.set_node_attributes(_dag, {new_name: self.node_attr_cache[new_name]})
            # _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            assert avg is not None, "avg must be given for new_name: {}".format(new_name)
            # combine attr avg
            attrs = {"avg": avg}
            ### set and cache the attribute
            nx.set_node_attributes(_dag, {new_name: attrs})
            self._cache_node_attr(new_name, _dag.nodes[new_name])

        return self.node_attr_cache[new_name]["avg"]

    def flush(self, is_accept):
        if is_accept:        
            for n in self.cache_change:
                if "host0.rank0" in n:
                    self._update_tensor2grp(n)
        self.cache_change = []
        
    def _op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
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
                if "Sync" in prev_fused_name and _all_infos[0][0] != _all_infos[1][0]:
                    ### avoid repeatedly add edges for Sync -> other GPUs tensor op
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
        if u_ not in self.node_attr_cache and u_ in _dag.nodes:
            self._cache_node_attr(u_, _dag.nodes[u_])
        if v_ not in self.node_attr_cache and v_ in _dag.nodes:
            self._cache_node_attr(v_, _dag.nodes[v_])
        
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
                self.tensor_group_info[all_infos[0][2]]["size"]
            # print("add new name {}".format(new_raw_name))

        if new_name not in self.node_attr_cache:
            if all_infos[0][3] in ["SEND", "RECV", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
                sizes = [
                    self._parse_tensor_group_info(all_infos[0][2])["size"],
                    self._parse_tensor_group_info(all_infos[1][2])["size"]]
                fused_time = FUSION_RATIO * self._get_node_attr(
                    [u_, v_][base_idx], "avg") * (sizes[0] + sizes[1]) / sizes[base_idx]
            elif all_infos[0][3] in ["QUEUE", "Sync"]:
                fused_time = FUSION_RATIO * max(self._get_node_attr(u_, "avg"), self._get_node_attr(v_, "avg"))
            else:
                raise ValueError("Unrecognized sub op name {} ({})".format(
                    all_infos[0][3], u_))
            self._cache_node_attr(new_name, {"avg": fused_time})
        else:
            fused_time = self.node_attr_cache[new_name]["avg"]
        return new_name, fused_time
    
    def _op_defusion(self, _dag, _pkg: PKGraph, u, loc):
        all_infos = self._wrap_parse_all_info(u)

        sorted_tensor_ids = sorted([int(n) for n in all_infos[2].split("+")])
        ids = [sorted_tensor_ids[:loc], sorted_tensor_ids[loc:]]
        rawnames = ["+".join([str(n) for n in _ids]) for _ids in ids]
        grp_infos = [self._parse_tensor_group_info(n, ref=all_infos[2]) for n in rawnames]

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
            _in_bw = [n for n, _ in _dag.in_edges(_pid_sync)]
            edges_to_add += [(_bw, new_part_names[0]) for _bw in _in_bw] + \
                [(_bw, new_part_names[1]) for _bw in _in_bw]
            edges_to_rm += [(_bw, _pid_sync) for _bw in _in_bw]     
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
                if "Sync" in _prev and _all_infos[0][0] != _all_infos[1][0]:
                    ### avoid repeatedly add edges for Sync -> other GPUs tensor op
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
        
        _dag.add_edges_from(edges_to_add)
        _dag.add_nodes_from(nodes_to_add)
        _dag.remove_edges_from(edges_to_rm)
        _dag.remove_nodes_from(nodes_to_rm)

        forbidden_list = [all_infos[2]]
        self._check_dag_not_contain(_dag, forbidden_list)

        self.cache_change += [n for n, _ in nodes_to_add]
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

    def _defuse_name_avg(self, _dag, u, rawnames):
        if u not in self.node_attr_cache and u in _dag.nodes:
            self._cache_node_attr(u, _dag.nodes[u])

        all_infos = self._wrap_parse_all_info(u)
        new_part_names = [self._wrap_gen_long_name(
            all_infos[0], all_infos[1], n, all_infos[3], all_infos[4]) for n in rawnames]
        grp_infos = [self._parse_tensor_group_info(n, ref=all_infos[2]) for n in rawnames]

        part_times = []
        for idx, new_name in enumerate(new_part_names):
            if new_name not in self.node_attr_cache:
                if all_infos[3] in ["SEND", "RECV", "MEMCPY_IN_FUSION_BUFFER", "NCCL_ALLREDUCE", "MEMCPY_OUT_FUSION_BUFFER"]:
                    part_time = self._get_node_attr(u, "avg") * grp_infos[idx]["size"] \
                        / ((grp_infos[0]["size"] + grp_infos[1]["size"]) * FUSION_RATIO)
                elif all_infos[3] in ["QUEUE", "Sync"]:
                    part_time = self._get_node_attr(u, "avg") / FUSION_RATIO
                else:
                    raise ValueError("Unrecognized sub op name {} ({})".format(
                        all_infos[3], u))
                self._cache_node_attr(new_name, {"avg": part_time})
            else:
                part_time = self.node_attr_cache[new_name]["avg"]
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
                    "loopNum": loopNum
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
        with open(self.ckpt_path, "wb") as f:
            pickle.dump([self.node_attr_cache, self.tensor_group_info, self.cur_tensor2group], f)

    def load_ckpt(self):
        if os.path.isfile(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                self.node_attr_cache, self.tensor_group_info, self.cur_tensor2group = pickle.load(f)
        self.cache_change = []

    def load_init_ckpt(self):
        init_ckpt_path = os.path.join(ROOT_PATH, "tensor_fusion_init_ckpt.pickle")
        if os.path.isfile(init_ckpt_path):
            with open(init_ckpt_path, "rb") as f:
                self.cur_tensor2group = pickle.load(f)[0]
            SingleLogger().info("Reading init state from cache.")
        else:
            for n in self.dag.nodes():
                if "Comm" in n:
                    self._cache_node_attr(n, self.dag.nodes[n])
                    ### Assume each host has the same tensor fusion pattern
                    if "host0.rank0" in n:
                        self._update_tensor2grp(n)
            with open(init_ckpt_path, "wb") as f:
                pickle.dump([self.cur_tensor2group], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))
        self.dump_tensor_grp_mapping(_file_name="tensor_fusion_grp_mapping_init.json")
        self.cache_change = []
        return None, None, None

    def dump_tensor_grp_mapping(self, _file_name=None):
        file_name = 'tensor_fusion_grp_mapping.json' if _file_name is None else _file_name

        tensor_ids, tensor_grps = zip(*list(self.cur_tensor2group.items()))
        tensor_grps = set(tensor_grps)
        tensor_ids = set(tensor_ids)
        assert len(tensor_ids) == len(self.meta_info.gradient_name_list()), \
            ("incompleted tensor_ids {} : {}".format(sorted(tensor_ids), len(self.meta_info.gradient_name_list())))

        with open(os.path.join(ROOT_PATH, file_name), 'w') as f:
            json.dump({"mapping": list(tensor_grps)}, f)
