import networkx as nx
import time
import os
import pickle
import numpy as np

import arg_utils
from .base import _BaseCostModel
from trace_utils import *
from cost_model_xla.pk_graph import PKGraph

args_ = arg_utils.SingleArg().args
FUSION_RATIO = 1

class _TensorFusionCM(_BaseCostModel):
    ''' This is a cost model for HOROVOD tensor fusion
    '''

    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["++"]
        self.meta_info = self.opt.clct.para_dict
        self.node_attr_cache = {}
        for n in self.dag.nodes():
            if "Comm" in n:
                self._cache_node_attr(n, self.dag.nodes[n])
        self.tensor_group_info = {}

        self.ckpt_path = os.path.join(args_.workspace, "ckpt_tensor_fusion.pickle")

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []

        for n, l in candidates:
            if "Comm" in n and "Sync" in n:
                to_fuse_u = n
                to_fuse_v = None
                ### For a fused tensor, it may have multiple BW inputs
                bw_list = [u for u, _ in _dag.in_edges(n)]
                to_check_list = []
                for bw_node in bw_list:
                    _succ = [n for n in _dag.successors(bw_node) if ("BW" in n and n not in bw_list)]
                    assert len(_succ) < 2, (bw_node, _succ)
                    if len(_succ) > 0:
                        to_check_list.append(_succ[0])
            
                ### Find the nearest tensors
                while len(to_check_list) > 0 and to_fuse_v is None:
                    bw_node = to_check_list.pop()
                    for _succ in list(_dag.successors(bw_node)):
                        if "BW" in _succ:
                            if _succ not in bw_list:
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

                ### tensor partition
                all_infos = self._wrap_parse_all_info(n)
                sorted_tensor_ids = [int(n) for n in all_infos[2].split("+")]
                mu, sigma = (len(sorted_tensor_ids)-1) / 2, (len(sorted_tensor_ids)-1) / 2
                s = np.random.normal(mu, sigma, int(self.ratio_of_partition * (len(sorted_tensor_ids) - 1)))
                _set = set([int(n) for n in s])
                for _s in _set:
                    ### Here _s denotes which the partition occurs before the _s'th tensor
                    search_space.append(("--", n, _s))
                    weights.append(l)

            
        return search_space, weights

    def apply(self, s, __dag, __pkg):
        op, target, next_ = s
        if op == "++":
            ### Fuse two nodes
            return self._op_fusion(__dag, __pkg, target, next_)
        elif op == "--":
            return self._op_defusion(__dag, __pkg, target, next_)

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

    def _op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        all_infos = [
            self._wrap_parse_all_info(u_),
            self._wrap_parse_all_info(v_)]
          
        tensor_group_infos = [
            self._parse_tensor_group_info(all_infos[0][2]),
            self._parse_tensor_group_info(all_infos[1][2])]
        if tensor_group_infos[0]['size'] >= tensor_group_infos[1]['size']:
            base_idx = 0
        else:
            base_idx = 1

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
            new_fused_name, fused_time = self._concat_name_avg(_dag, _pid_sync[0], _pid_sync[1])
            nodes_to_add.append((new_fused_name, {"avg": fused_time}))

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
        return True, [n for n, _ in nodes_to_add], nodes_to_rm

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
        self.tensor_group_info[new_raw_name] = self.tensor_group_info[all_infos[base_idx][2]].copy()
        self.tensor_group_info[new_raw_name]["size"] = \
            self.tensor_group_info[all_infos[0][2]]["size"] + \
            self.tensor_group_info[all_infos[0][2]]["size"]

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
        rawnames = ["+".join(n) for n in ids]
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
                nodes_to_rm.update((_cur))

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
                    part_time = self._get_node_attr(n, "avg") * grp_infos[idx]["size"] \
                        / ((grp_infos[0]["size"] + grp_infos[1]["size"]) * FUSION_RATIO)
                elif all_infos[3] in ["QUEUE", "Sync"]:
                    part_time = self._get_node_attr(n, "avg") / FUSION_RATIO
                else:
                    raise ValueError("Unrecognized sub op name {} ({})".format(
                        all_infos[3], u))
                self._cache_node_attr(new_name, {"avg": part_time})
            else:
                part_time = self.node_attr_cache[new_name]["avg"]
            part_times.append(part_time)
        return new_part_names, part_times

    def _parse_tensor_group_info(self, raw_name, ref=None):
        if raw_name not in self.tensor_group_info:
            if ref is None:
                chunkNum, sliceNum, channelNum, loopNum = self.opt.clct.nccl_graph.get_IDnum("Comm." + raw_name)
            else:
                ### some group may not occur in the traces, ref to other groups
                chunkNum, sliceNum, channelNum, loopNum = self.opt.clct.nccl_graph.get_IDnum("Comm." + ref)
            total_size = 0
            for tensor_id_str in raw_name.split("+"):
                tensor_id = int(tensor_id_str)
                total_size += self.meta_info.tensor_id2size(tensor_id)
            self.tensor_group_info[raw_name] = {
                "chunkNum": chunkNum,
                "sliceNum": sliceNum,
                "channelNum": channelNum,
                "loopNum": loopNum,
                "size": total_size
            }
        return self.tensor_group_info[raw_name]

    def checkpoint(self):
        with open(self.ckpt_path, "wb") as f:
            pickle.dump([self.node_attr_cache, self.tensor_group_info], f)

    def load_ckpt(self):
        with open(self.ckpt_path, "rb") as f:
            self.node_attr_cache, self.tensor_group_info = pickle.load(f)

    def load_init_ckpt(self):
        return None, None, None
