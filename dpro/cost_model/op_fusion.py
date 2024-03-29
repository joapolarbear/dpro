from multiprocessing import Value
import os, sys
import networkx as nx
from scipy.stats.mstats import gmean
import numpy as np
import pickle
import itertools
import json
from tqdm import tqdm, trange
import subprocess
import re
import time

from ..arg_utils import SingleArg
from ..trace_utils import painted_timeline, parse_cat_from_name, parse_pid_from_name, \
    _map_tf_op2layer, parse_cat_fine_grained, _parse_tf_layer_names, \
    gen_long_name, parse_op_name, \
    SingleLogger, GAP_STR_OP2OP, GAP_STR_OP2COMM, CatName, FileName
from ..base import bcolors

from .base import _BaseGraphPass, OptQueryCostModelError
from ._xla.utils import parse_xla_candidate_ops, IGNORE_OP_TYPES
from ._xla.pk_graph import PKGraph, contract_nodes_nx, \
    defuse_nodes_inplace_nx, postorder_contract_nx, \
    subgraph_partition_connected_nx_using_topo, get_concated_names, \
    contract_groups
from ._xla.xla_module_cost_model import XLAModuleCostModel

args_ = SingleArg().args
FUSION_TIME_ESTIMATE_RATIO = 0.8
FORCE_ESTIMATE_FUSION_TIME = False
ENABLE_PRUNING = False
NAIVE_SIMULATE = False

class AttrCache():
    def __init__(self):
        self._cache = {}

    def _parse_key(self, node_name):
        if "+" in node_name:
            key = tuple(sorted(node_name.split("+")))
        else:
            key = node_name
        return key

    def __contains__(self, node_name):
        key = self._parse_key(node_name)
        return key in self._cache

    def __getitem__(self, node_name):
        key = self._parse_key(node_name)
        return self._cache[key]

    def __setitem__(self, node_name, value):
        key = self._parse_key(node_name)
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)


class XLAGraphPass(_BaseGraphPass):
    def __init__(self, opt, root_path):
        super().__init__(opt)
        self.root_path = root_path

        if not args_.simulate:
            self.cost_models = self._load_cm()
        else:
            self.cost_models = None
        self.forbidden_list = set()
        self.initial_forbidden_list = set()
        # self._init_forbidden_list()
        self.token = ["+", "-"]

        ### Need to cache
        self.ckpt_path = os.path.join(self.ckpt_dir, "xla_ckpt.pickle")
        ### Used to cache the node attribtue
        self.node_attr_cache = AttrCache()
        ### Cache iteration time on single GPU
        self.iter_time_single_gpu = None
        self.iter_time_single_gpu_base_line = None
        self.iter_time_single_gpu_path = os.path.join(args_.path, "xla_iter_time_single_gpu.json")

        self.explore_fusion = True
        self.enable_partition = True
        self.xla_init_forbid_bw = True
        if args_.layer_by_layer:
            self.xla_init_forbid_bw = True

        self.cord_pid = self.opt.cord_pid

        self.init_dfg = None
        self.init_op2fused = None

        self.base_cmd = os.getenv("REPLAY_CMD", None)
        if self.base_cmd is None:
            SingleLogger().warn(bcolors.CRED + "The replay command is not set by REPLAY_CMD, use the default cmd" + bcolors.ENDC)
            self.base_cmd = ['python3', '/usr/local/byteps/launcher/launch.py',
                        'python3', '/home/tiger/horovod_examples/tensorflow2/tensorflow2_synthetic_benchmark.py',
                        '--comm_backend', 'bps']
        else:
            self.base_cmd = [sub_cmd for sub_cmd in self.base_cmd.split(" ") if len(sub_cmd) > 0]

    def load_init_ckpt(self, G_prime=None):
        ''' Other cost model may initialize the DFG, init DFG based on that
        '''
        if os.path.isfile(self.iter_time_single_gpu_path):
            with open(self.iter_time_single_gpu_path, "r") as f:
                self.iter_time_single_gpu = json.load(f)
        else:
            self.iter_time_single_gpu = {}

        init_ckpt_path = os.path.join(self.ckpt_dir, "xla_init_ckpt.pickle")
        trajectory = []
        if os.path.isfile(init_ckpt_path):
            with open(init_ckpt_path, "rb") as f:
                G, PKG, node_attr_cache, initial_partitions_formed, \
                    self.forbidden_list, self.init_op2fused = pickle.load(
                    f)
                self.node_attr_cache = node_attr_cache
            SingleLogger().info("Reading init graph from cache: {}.".format(init_ckpt_path))
        else:
            G = self.dag.copy() if G_prime is None else G_prime.copy()
            PKG = PKGraph(G)
            self._init_forbidden_list(G_prime=G)
            # # randomly contract edges if possible
            # k = int(len(G.edges()) * init_edges_to_contract)
            for node in G.nodes():
                if node not in self.node_attr_cache:
                    self._cache_node_attr(node, G.nodes[node])

            if args_.layer_num_limit:
                self._sample_strategies(
                    G, PKG, layer_num_limit=args_.layer_num_limit)
                exit(0)

            G, initial_partitions_formed = self._init_partition(G, PKG)
            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, self.node_attr_cache, initial_partitions_formed,
                             self.forbidden_list, self.init_op2fused], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))
        
        if "BPF_DUMP_INIT_CLUSTER_TO" in os.environ:
            self._dump_cluster_mapping(G, os.environ["BPF_DUMP_INIT_CLUSTER_TO"], partition=True)
        SingleLogger().info("Successfully initialized {} partitions.".format(initial_partitions_formed))

        self.checkpoint()
        
        # self._check_dag_avg(G)
        self.init_dfg = G.copy()

        if not args_.pretty:
            ### Debug, check subgraph
            check_dag = nx.DiGraph()
            nodes = [n for n in self.init_dfg if "host0.rank0" in n and ("BW" in n or ("Comm" in n and "Sync" in n))]
            sub_dag = self.init_dfg.subgraph(nodes)
            def relabel_func(old_label):
                idx = nodes.index(old_label)
                prefix = "BW." if "BW" in old_label else "Comm."
                return "{}{}".format(prefix, idx)
            sub_dag = nx.relabel_nodes(sub_dag, relabel_func)
            nx.drawing.nx_pydot.write_dot(sub_dag, "/home/tiger/small_dag.txt")
            with open("/home/tiger/small_dag_idmapping.txt", 'w') as fp:
                json.dump(dict(enumerate(nodes)), fp, indent=4)

        ### Dump the default tensor fusion pattern
        comm_set = set([parse_op_name(n) for n in self.init_dfg.nodes if "Comm" in n and self.cord_pid in n])
        with open(os.path.join(self.spec_dir, "tensor_grp.json"), 'w') as fp:
            json.dump({"mapping": list(comm_set)}, fp, indent=4)

        return G, PKG, trajectory

    def load_ckpt(self):
        if os.path.isfile(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                node_attr_cache = pickle.load(f)
                self.node_attr_cache = node_attr_cache

        if os.path.isfile(self.iter_time_single_gpu_path):
            with open(self.iter_time_single_gpu_path, "rb") as f:
                self.iter_time_single_gpu = json.load(f)

    def checkpoint(self):
        with open(self.ckpt_path, "wb") as f:
            pickle.dump(self.node_attr_cache, f)
        
        with open(self.iter_time_single_gpu_path, "w") as f:
            json.dump(self.iter_time_single_gpu, f)

    def _load_cm(self):
        cost_models = {}
        models_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "_xla/.cost_model")

        cost_model_tmp_dir = os.path.join(self.root_path, "xla_cm_tmp")
        if not os.path.exists(cost_model_tmp_dir):
            os.makedirs(cost_model_tmp_dir)
        SingleLogger().info("Searching for XLA Cost Model dumps in {}".format(models_dir))
        cost_models["default"] = XLAModuleCostModel(models_dir, tmp_dir=os.path.join(cost_model_tmp_dir))
        return cost_models

    def _reduce_nx_size(self, G):
        ret_G = nx.DiGraph()
        edges_to_add = []
        for u, v in G.edges:
            if u == v:
                continue
            if u.startswith(self.cord_pid) and v.startswith(self.cord_pid):
                # we also remove FW -> UPDATE egdes here since now we have 
                # removed communication nodes, postorder_contract will try to
                # fuse UPDATE with FW
                if not (("FW" in u or "BW" in u) and "UPDATE" in v):
                    edges_to_add.append((u, v))
        ret_G.add_edges_from(edges_to_add)
        return ret_G

    def _sample_strategies(self, G, PKG, layer_num_limit):
        ''' Sample some operator fusion strategies by fusing operators layer by layer
            and generate cluster mapping files
        '''
        SingleLogger().info("Start to sample strategies, ... ")
        if layer_num_limit.isdigit():
            layer_num_limit = int(layer_num_limit)
            layer_num_list = layer_num_limit if layer_num_limit >= 0 else [1, 3, 5, 10, 20]
        else:
            layer_num_list = [int(n) for n in layer_num_limit.split(",")]

        first = True
        for _layer_num_limit in layer_num_list:
            partition_G = self._reduce_nx_size(G)
            partition_PKG = PKGraph(partition_G)
            source_nodes = sorted(
                [node for node in partition_G.nodes if node not in self.forbidden_list and "Comm" not in node], key=lambda x: partition_G.in_degree(x))
            # print(source_nodes)

            if first:
                layer_set = set()
                for n in partition_G.nodes():
                    if "BW" not in n:
                        continue
                    layer_name = _parse_tf_layer_names(n)
                    layer_set.add(layer_name[0])
                SingleLogger().info(bcolors.CYELLOWBG + "There are {} BW layers in total".format(len(layer_set)) + bcolors.ENDC)
                first = False

            # Run post order traversal on partition_G
            visited_nodes = set()
            for source in tqdm(source_nodes, total=len(source_nodes)):
                if source not in visited_nodes and source in partition_G.nodes:
                    _, _, partition_G = postorder_contract_nx(
                        partition_G, partition_PKG, source, visited_nodes,
                        forbidden_list = self.forbidden_list,
                        layer_num_limit = _layer_num_limit
                    )
            self._dump_cluster_mapping(partition_G, os.path.join(
                self.spec_dir, "cluster_mapping_layer_num_limit_{}.txt".format(_layer_num_limit)))
            
            G_copy = G.copy()
            PKG_copy = PKG.copy()
            G_copy, _ = self._create_clusters(G_copy, PKG_copy, partition_G)
            cost, _, _ = self.opt.evaluate(G_copy)
            SingleLogger().info(bcolors.CYELLOWBG +
                                "_layer_num_limit: {} ==> cost: {}".format(_layer_num_limit, cost) + bcolors.ENDC)

        SingleLogger().info("Done. Strategies are stored at {}".format(self.spec_dir))

    def _init_partition(self, G, PKG):
        ''' Initialize the graph with a default operator fusion strategy
            By default, this function tries to fuse all operators and avoids cycles
        '''
        partition_G = self._reduce_nx_size(G)
        partition_PKG = PKGraph(partition_G)

        if args_.layer_by_layer:
            if not args_.simulate:
                self.cost_models["default"].silent = True

            model_name = self.opt.clct.para_dict.parse_model_name()
            parse_layer_method = 2
            if parse_layer_method == 1:
                op2layer, layer2ops = _map_tf_op2layer(self.opt.clct.dag, model_name)
            elif parse_layer_method == 2:
                op2layer, layer2ops = _map_tf_op2layer(self.opt.clct.dag, model_name,
                    use_rough_layer=False, check_pred=False)
            else:
                raise ValueError()
            
            # import code
            # code.interact(local=locals())
            assert len([op for op in op2layer.keys() if "FW" in op]) == 0
            
            # from dag_utils import part_of_dag
            # node = "BW.gradient_tape/resnet50/avg_pool/Tile"
            # focus_nodes = ["host0.rank0->BW.gradient_tape/resnet50/conv5_block3_out/ReluGrad", "host0.rank0->BW.gradient_tape/resnet50/avg_pool/Tile", "host0.rank0->BW.gradient_tape/resnet50/avg_pool/Reshape", "host0.rank0->BW.gradient_tape/resnet50/avg_pool/truediv", "host0.rank0->BW.gradient_tape/resnet50/predictions/MatMul", "host0.rank0->BW.gradient_tape/resnet50/avg_pool/Tile/multiples", "host0.rank0->BW.gradient_tape/resnet50/conv5_block3_3_bn/FusedBatchNormGradV3", "host0.rank0->BW.gradient_tape/resnet50/conv5_block3_3_conv/Conv2D/Conv2DBackpropFilter", "host0.rank0->BW.gradient_tape/resnet50/avg_pool/Reshape/shape"]
            # small_dag = part_of_dag(self.opt.clct.dag, node,
            #     max_in_depth=3, max_out_depth=3,
            #     path = "/home/tiger/small_dag.txt",
            #     simple=False,
            #     focus_nodes=focus_nodes)
            # exit(0)

            ### We will handle bw operators seperately, so add bw ops to initial_forbidden_list first
            # self.initial_forbidden_list.union([gen_long_name(self.cord_pid, bw_op) for bw_op in op2layer.keys()])

        source_nodes = sorted(
            [node for node in partition_G.nodes if 
                node not in self.initial_forbidden_list and "Comm" not in node], 
            key=lambda x: partition_G.in_degree(x))
        # Run post order traversal on partition_G
        visited_nodes = set()
        SingleLogger().info("Start to postorder_contract_nx ... ")
        for source in tqdm(source_nodes, total=len(source_nodes)):
            if source not in visited_nodes and source in partition_G.nodes:
                _, _, partition_G = postorder_contract_nx(
                    partition_G, partition_PKG, source, visited_nodes,
                    forbidden_list=self.initial_forbidden_list)
        
        if args_.layer_by_layer:
            painted_timeline(self.opt.clct.traceM.traces, lambda event: op2layer.get(event["name"], event["name"]),
                os.path.join(self.root_path, "op2layer.json"))
            list_of_group = [[gen_long_name(self.cord_pid, n) for n in nodes_to_contract] for nodes_to_contract in layer2ops.values() if len(nodes_to_contract) > 1]
            SingleLogger().info("[Layer View] Start to contract {} layer ...".format(len(list_of_group)))
            partition_G = contract_groups(
                partition_G, partition_PKG, 
                forbidden_list=self.forbidden_list,
                list_of_group = list_of_group)
        
        G, initial_partitions_formed = self._create_clusters(G, PKG, partition_G)
        self._dump_cluster_mapping(G, os.path.join(
                self.spec_dir, "cluster_mapping_after_initialization.txt"), partition=True)

        if False:
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = '0'
            if "BYTEPS_TRACE_ON" in my_env:
                my_env.pop("BYTEPS_TRACE_ON")
            if "XLA_DUMP_DIR" in my_env:
                my_env.pop("XLA_DUMP_DIR")

            my_env["XLA_CLUSTER_SPEC"] = os.path.join(
                self.spec_dir, "cluster_mapping_after_initialization.txt")
            my_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
            init_fused_time = self._estimate_time_real_replay(self.base_cmd, my_env)
            if init_fused_time is None:
                SingleLogger().warn("Fail to run init")
                exit(1)

            if "baseline_single_gpu" not in self.iter_time_single_gpu:
                my_env.pop("XLA_CLUSTER_SPEC")
                my_env.pop("TF_XLA_FLAGS")
                avg = self._estimate_time_real_replay(self.base_cmd, my_env)
                assert avg is not None
                self.iter_time_single_gpu['baseline_single_gpu'] = avg
            SingleLogger().info("Origin time {} ==> fused time {} ".format(
                self.iter_time_single_gpu['baseline_single_gpu'], init_fused_time))
            exit(0)

        SingleLogger().info("Start to init partition graph ... ")
        return G, initial_partitions_formed

    def _create_clusters(self, G, PKG, partition_G):
        ''' Init partition on `G` based on the the pre-partitioned DFG `partition_G`
            **NOTE**: this modification is in place
        '''
        initial_partitions_formed = 0
        for node_name in tqdm(partition_G.nodes()):
            if node_name in self.initial_forbidden_list or "Comm" in node_name:
                continue
            
            if "+" in node_name:
                # fused node, test if compilable
                try:
                    avg = self._get_node_avg(node_name, verbose=False)
                    self._parse_node_attr(partition_G, node_name, avg)
                    compilable=True
                except (OptQueryCostModelError, ValueError):
                    compilable=False
                
                if compilable and avg is not None:
                    for _node_name in [node_name] + self.opt._debug_convert_to_other_machines(node_name):
                        ns = _node_name.split("+")

                        ### DEBUG
                        if self.cord_pid in _node_name:
                            origin_avg = 0
                            for _op in ns:
                                origin_avg += G.nodes[_op]["avg"]
                            if origin_avg > 0:
                                print("Fusion from {} nodes {:.3f} ms to {:.3f} ms {}".format(
                                    len(ns), origin_avg, avg, " <<< !!!" if avg > origin_avg else ""))

                        new_node_name = contract_nodes_nx(G, ns)
                        PKG.contract_nodes_unsafe(ns)
                        ### TODO (huhanpeng): since we assume data parallel
                        ### we directly use the fused time of the first GPU for all GPUs' fused nodes
                        self._parse_node_attr(G, new_node_name, avg) # type: ignore
                        initial_partitions_formed += 1
                        if args_.layer_by_layer:
                            if self.init_op2fused is None:
                                self.init_op2fused = {}
                            for op in ns:
                                self.init_op2fused[op] = new_node_name
        return G, initial_partitions_formed

    def _init_forbidden_list(self, G_prime=None):
        ''' Operators in the forbidden list will not be contacted with other operators
        `self.forbidden_list` is used through the search process
        `self.initial_forbidden_list` is only used when initalizing the fusion pattern.
        '''
        xla_candidates, _ = parse_xla_candidate_ops(args_.xla_candidate_path)
        # limit the range of nodes during search
        filtered_xla_candidates = set()
        for op in xla_candidates:
            should_ignore = False
            for ignore_type in IGNORE_OP_TYPES:
                if ignore_type in op:
                    should_ignore = True
                    break
            if not should_ignore:
                filtered_xla_candidates.add(op)
        
        dag = self.dag if G_prime is None else G_prime
        for node in dag.nodes:
            # ignore BW nodes and communication nodes
            if self.xla_init_forbid_bw and "BW" in node:
                self.initial_forbidden_list.add(node)

            try:
                op_name, pid = self.opt._get_original_name_pid_from_index(node)
            except:
                # not standard nodes, ignore
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                continue
            cat = parse_cat_from_name(node)
            if (not args_.simulate and op_name not in self._wrap_xla_operation_names(pid)) \
                    or "Assign" in op_name or cat == CatName.COMM.value \
                    or op_name not in filtered_xla_candidates:
                self.forbidden_list.add(node)
                self.initial_forbidden_list.add(node)
                
    def _get_node_attr(self, n, attr_):
        if attr_ in self.node_attr_cache[n]:
            return self.node_attr_cache[n][attr_]
        else:
            return 0

    def _cache_node_attr(self, n, attrs):
        ### TODO (huhanpeng): need .copy() ???
        self.node_attr_cache[n] = attrs

    def _dump_cluster_mapping(self, dag, output_path, partition=False):
        ''' Dump cluster mapping in `dag` at `output_path`
            If `partition` is set True, the dag will be partitioned
        '''
        if partition:
            dag = self._reduce_nx_size(dag)
        cluster_index = 0
        with open(output_path, "w") as f:
            for node in dag.nodes():
                if "+" in node and "Comm" not in node:
                    op_names, _ = self._get_original_name_pid_from_fused_node(node)
                    for orig_node_name in op_names:
                        f.write("{} {}\n".format(orig_node_name, cluster_index))
                    cluster_index += 1

    def _get_original_name_pid_from_fused_node(self, u_):
        single_pid = None
        op_names = []
        for node_name in self._get_defused_node_names(u_):
            op_name, pid = self.opt._get_original_name_pid_from_index(node_name)
            op_names.append(op_name)
            if single_pid is None:
                single_pid = pid
            else:
                if single_pid != pid:
                    raise RuntimeError(
                        "Fused DAG node {} contains ops from different machines.".format(u_))
        return op_names, single_pid

    def _get_defused_node_names(self, fused_node_):
        return fused_node_.split("+")

    def _wrap_xla_need_fuse(self, long_name):
        try:
            orig_name, pid = self.opt._get_original_name_pid_from_index(long_name)
        except (IndexError, KeyError):
            return False
        
        ### TODO (huhanpeng)
        # if orig_name.endswith("_Switch"):
        #     return True

        if args_.simulate:
            return long_name not in self.forbidden_list
        else:
            # if "_Switch" in orig_name and orig_name not in self._wrap_xla_operation_names(pid):
            #     print("{} not in xla_operation_names".format(orig_name))
            # if "_Switch" in orig_name and long_name in self.forbidden_list:
            #     print("{} in the forbidden list".format(orig_name))
            return (orig_name in self._wrap_xla_operation_names(pid)) and long_name not in self.forbidden_list

    def _wrap_xla_operation_names(self, pid):
        return self.cost_models["default"].graph_def_util.operation_names
        
    def _wrap_can_fuse_to_b(self, _pkg: PKGraph, a, b):
        ''' Return whether a can be fused with b
        '''
        if "Comm" in b or not _pkg.can_contract_edge(a, b):
            return False

        if "+" not in b and not self._wrap_xla_need_fuse(b):
            # if 'BW' in succ_:
            #     print("Ignore succ {}".format(succ_))
            return False
        
        if parse_pid_from_name(a) != parse_pid_from_name(b) or parse_cat_from_name(a) != parse_cat_from_name(b):
            return False

        return True

    def init_search_space_by_nodes(self, preds, node, _pkg):
        search_space = []
        if not self._wrap_xla_need_fuse(node):
            return []
        
        for pred in preds:
            if self._wrap_xla_need_fuse(pred) and self._wrap_can_fuse_to_b(_pkg, pred, node):
                search_space.append(("+", pred, node))

        return search_space

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        ### Based on the candidates, init the search space for the new dependency graph `_dag`
        ### TODO (huhanpeng): currently only consider fusion
        ###             Need to add quantization
        search_space = []
        weights = []
        prun_cnt = 0
        fusion_cnt = 0
        defusion_cnt = 0

        # TODO (huhanpeng): below code forces to search without the limitation of critical paths.
        if args_.no_crit:
            candidates = [(n, _dag.nodes[n]["avg"]) for n in _dag.nodes() if "BW" in n and "host0.rank0" in n]

        for cur_op, l in candidates:
            # node heat
            heat = self.opt._get_heat_from_history(cur_op)
            if "Comm" in cur_op:
                continue

            if "+" in cur_op and self.enable_partition:
                ### This a fused node
                if args_.layer_by_layer and "FW" in cur_op and "BW" not in cur_op:
                    continue
                ns = cur_op.split("+")
                cat = parse_cat_fine_grained(ns[0])
                pid = parse_pid_from_name(ns[0])
                if args_.layer_by_layer:
                    ### enforce that layer is the minimun unit
                    ns = set([self.init_op2fused.get(op, op) for op in ns])
                    subgraph = self.init_dfg.subgraph(ns)
                else:
                    ns = set(ns)
                    subgraph = self.dag.subgraph(ns)

                if len(ns) > 1:
                    # randomly split edges using spanning tree
                    ### a list of tupe(nodes_in_a, nodes_in_b)
                    valid_split_plans = subgraph_partition_connected_nx_using_topo(subgraph)
                    split_weights = []
                    for splits in valid_split_plans:
                        split_weights.append(gmean([len(nodes) for nodes in splits]))
                    split_weights = np.exp(5e-4*(len(ns) - 80)) * (np.array(split_weights) / np.sum(split_weights))
                    for split_index, splits in enumerate(valid_split_plans):
                        if args_.layer_by_layer:
                            splits = [list(itertools.chain(*[n.split("+") for n in nodes])) for nodes in splits]
                        search_space.append(("-", cur_op, splits))
                        defusion_cnt += 1
                        weights.append(self.opt._combine_weight(l, 1 / (heat + 1) - 1) * split_weights[split_index])
            else:
                ### Nodes that have never been fused
                cat = parse_cat_fine_grained(cur_op)
                pid = parse_pid_from_name(cur_op)
            
            ### TODO (huhanpeng): !!! NOTE: should modify below code back
            if args_.fusion_once and not self._wrap_xla_need_fuse(cur_op):
                continue
            elif not args_.fusion_once and "+" not in cur_op and not self._wrap_xla_need_fuse(cur_op):
                # if 'BW' in n:
                #     print("Ignore {}".format(n))
                continue

            for succ_ in _dag.successors(cur_op):
                if not self._wrap_can_fuse_to_b(_pkg, cur_op, succ_):
                    continue

                ### Assumption: for edge bw_u->bw_v, 
                # if comm_bw_u > bw_v, it can not bring any speedup if fusing u and v.
                if ENABLE_PRUNING:
                    comm_avg_u = self._op_to_coarse_comm_time(cur_op, _dag, pid)
                    effective_succ_bw = set()
                    for bw_v_succ in _dag.successors(succ_):
                        if "Comm" in bw_v_succ:
                            # check if the comm node's predecessor includes succ_
                            if cur_op in _dag.predecessors(bw_v_succ):
                                # OK to fuse
                                continue
                            else:
                                effective_succ_bw.union(set([node for node in _dag.predecessors(bw_u_succ) if "BW" in node]))
                    effective_avg_sum = 0
                    for node in effective_succ_bw:
                        effective_avg_sum += _dag.nodes[node]["avg"]

                    if comm_avg_u > effective_avg_sum:
                        prun_cnt += 1
                        SingleLogger().debug("Prune fusing {} and {} with comm time {}".format(cur_op, succ_, comm_t))
                        continue

                # calculate heat using max(heat(n), heat(succ_))
                heat_succ = self.opt._get_heat_from_history(succ_)
                heat_combined = (heat + heat_succ) / 2

                search_space.append(("+", cur_op, succ_))
                fusion_cnt += 1
                weights.append(self.opt._combine_weight(l, heat_combined))
                # weights.append(1)
        
        SingleLogger().info("Init search space from {} candidates, prune {}: {} fusion strategies, {} defusion strategies".format(
            len(candidates), prun_cnt if ENABLE_PRUNING else "(disabled)", fusion_cnt, defusion_cnt))

        if len(search_space) > 0:
            weights /= np.linalg.norm(weights)
            weights = list(weights)

        ### TODO (huhanpeng) delete
        bw_num_in_critical_path = len([n for n, _ in candidates if "BW" in n])
        bw_num_in_g = len([n for n in _dag.nodes() if "BW" in n and "host0.rank0" in n])
        SingleLogger().info(bcolors.CSELECTED +
                            "{}/{} BW nodes in the critical path".format(bw_num_in_critical_path, bw_num_in_g) + bcolors.ENDC)

        return search_space, weights

    def _concat_name(self, u_, v_):
        return "%s+%s" % (u_, v_)
        
    def _combine_gap(self, ug, vg):
        ### TODO (huhanpeng): key component
        ### Use max to avoid one input is zero,
        ### some how for the new gap x, ug < x < ug + vg, vg < x < ug + vg
        # return max(max((ug + vg) / 0.8, ug), vg)
        return max(ug, vg)

    def _combine_nodes_attr(self, _dag, target, u_, v_, avg):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target
        _dag.nodes[target]["avg"] = avg
        _dag.nodes[target][GAP_STR_OP2OP] = self._combine_gap(self._get_node_attr(u_, GAP_STR_OP2OP), self._get_node_attr(v_, GAP_STR_OP2OP))
        _dag.nodes[target][GAP_STR_OP2COMM] = self._combine_gap(self._get_node_attr(u_, GAP_STR_OP2COMM), self._get_node_attr(v_, GAP_STR_OP2COMM))

    def _combine_attr_except_avg(self, target, attr1, attr2):
        ### In graph _dag, combine the attributes of u_ and v_, store the results in _dag as the attributes of target

        if GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(attr1[GAP_STR_OP2OP], attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP not in attr1 and GAP_STR_OP2OP in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(0, attr2[GAP_STR_OP2OP])
        elif GAP_STR_OP2OP in attr1 and GAP_STR_OP2OP not in attr2:
            target[GAP_STR_OP2OP] = self._combine_gap(attr1[GAP_STR_OP2OP], 0)

        if GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(attr1[GAP_STR_OP2COMM], attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM not in attr1 and GAP_STR_OP2COMM in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(0, attr2[GAP_STR_OP2COMM])
        elif GAP_STR_OP2COMM in attr1 and GAP_STR_OP2COMM not in attr2:
            target[GAP_STR_OP2COMM] = self._combine_gap(attr1[GAP_STR_OP2COMM], 0)

    def _get_node_avg(self, new_name, verbose=True):
        if new_name not in self.node_attr_cache:
            assert "+" in new_name
            # query cost model for exec time of a fused node u
            nodes_in_u, u_pid = self._get_original_name_pid_from_fused_node(new_name)
            nodes_to_fuse = set(nodes_in_u)
            if verbose:
                if len(nodes_to_fuse) < 10:
                    SingleLogger().info("[COST MODEL QUERY] {} Nodes to fuse: {}".format(
                        len(nodes_to_fuse), nodes_to_fuse))
                else:
                    SingleLogger().info(
                        "[COST MODEL QUERY] {} Nodes to fuse ...".format(len(nodes_to_fuse)))

            predicted_time = self._wrap_xla_predict(u_pid, nodes_to_fuse, new_name, simulate=args_.simulate)
            if predicted_time is not None:
                if predicted_time < 0:
                    if args_.disable_estimate:
                        raise OptQueryCostModelError("Failed to query cost model.")
                    else:
                        predicted_time = self._wrap_xla_predict(
                            u_pid, nodes_to_fuse, new_name, simulate=True)
                        if verbose:
                            SingleLogger().warn(
                                "[COST MODEL QUERY] Exec time {}ESTIMATED{}: {}".format(bcolors.CYELLOWBG, bcolors.ENDC, predicted_time))
                else:
                    if verbose:
                        SingleLogger().info("[COST MODEL QUERY] Exec time predicted: {} (Avg. sum of origin: {}".format(
                            predicted_time, sum([self._get_node_avg(n) for n in new_name.split("+")])))
                    pass
            self.node_attr_cache[new_name] = {"avg": predicted_time}
        return self.node_attr_cache[new_name]["avg"]

    def _estimate_time_real_replay(self, cmd, env, time_limit=60, verbose=False):
        st = time.time()
        while True:
            try:
                ret = subprocess.check_output(cmd, 
                    stderr=subprocess.STDOUT, env=env).decode('utf-8')
            except:
                return None
            
            if re.search("tensorflow.python.framework.errors_impl.InvalidArgumentError: \d+ nodes in a cycle", ret):
                ### Introduce a cycle
                return None
            match = re.findall("iteration time (\d+[.\d]*) ms", ret)
            if len(match) == 0 or verbose:
                print(cmd)
                for line in ret.split("\n"):
                    print(line)
                subprocess.check_output(cmd, env=env
                    )
            if len(match) == 0:
                SingleLogger().error("Fail to match iteration time")
                exit(-1)
            match = np.array([float(itt) for itt in match])
            iter_time = np.average(match)
            stdev = np.std(match)
            if stdev / iter_time < 0.2:
                return iter_time
            if time.time() - st > time_limit:
                SingleLogger().warn("[XLA CM] return an unstable iter_time time {:.3f}(\u00B1{:.3f})ms, timelimit={:.1f} min".format(iter_time, stdev, int(time_limit/60)))
                return iter_time
    
    def _wrap_xla_predict(self, pid, nodes_to_fuse, fused_u_, simulate=False):
        ''' 
        nodes_to_fuse: a list of layer names to fuse
        fused_u_: a str of fused names with layer index
        '''
        if FORCE_ESTIMATE_FUSION_TIME or simulate:
            if NAIVE_SIMULATE:
                _sum = 0
                origin_nodes = self._get_defused_node_names(fused_u_)
                for idx in range(len(origin_nodes) - 1):
                    _sum += self.node_attr_cache[origin_nodes[idx]]["avg"]
                return _sum * FUSION_TIME_ESTIMATE_RATIO + self.node_attr_cache[origin_nodes[-1]]["avg"]
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = '0'
            if "BYTEPS_TRACE_ON" in my_env:
                my_env.pop("BYTEPS_TRACE_ON")
            if "XLA_DUMP_DIR" in my_env:
                my_env.pop("XLA_DUMP_DIR")

            key = "fuse-" + "+".join(sorted(nodes_to_fuse))
            sum_time = sum([self._get_node_avg(n) for n in fused_u_.split("+")])
            if sum_time == 0:
                return 0
            if key not in self.iter_time_single_gpu:
                with open("/tmp/xla_spec.txt", 'w') as fp:
                    for orig_node_name in nodes_to_fuse:
                        fp.write("{} {}\n".format(orig_node_name, 0))
                my_env["XLA_CLUSTER_SPEC"] = '/tmp/xla_spec.txt'
                my_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
                self.iter_time_single_gpu[key] = self._estimate_time_real_replay(self.base_cmd, my_env)

            iter_time_after_fuse = self.iter_time_single_gpu[key]
            if iter_time_after_fuse is None:
                SingleLogger().warn("Fail to fuse {}".format(nodes_to_fuse))
                return None

            if "baseline_single_gpu" not in self.iter_time_single_gpu:
                my_env.pop("XLA_CLUSTER_SPEC")
                my_env.pop("TF_XLA_FLAGS")
                avg = self._estimate_time_real_replay(self.base_cmd, my_env)
                assert avg is not None
                self.iter_time_single_gpu['baseline_single_gpu'] = avg
            elif self.iter_time_single_gpu_base_line is None:
                if "XLA_CLUSTER_SPEC" in my_env:
                    my_env.pop("XLA_CLUSTER_SPEC")
                if "TF_XLA_FLAGS" in my_env:
                    my_env.pop("TF_XLA_FLAGS")
                avg = self._estimate_time_real_replay(self.base_cmd, my_env)
                assert avg is not None
                self.iter_time_single_gpu_base_line = avg
                SingleLogger().info("Base Iter Time: Before {}, Now {}".format(
                    self.iter_time_single_gpu['baseline_single_gpu'], self.iter_time_single_gpu_base_line))
                if abs(avg - self.iter_time_single_gpu['baseline_single_gpu']) / self.iter_time_single_gpu['baseline_single_gpu'] > 0.1:
                    SingleLogger().warn("Inconsistent baseline, go on?")
                    name = input("Input your command[Y/n]: ")
                    if name.lower() in ["q", 'n', "no"]:
                        exit(0)
                
            before_fuse = self.iter_time_single_gpu['baseline_single_gpu']
            fused_time = max(0, sum_time + iter_time_after_fuse - before_fuse)
            SingleLogger().info("[OPFS CM] From {} to {}".format(sum_time, fused_time))
            if sum_time > 0.5 and fused_time > 10 * sum_time:
                SingleLogger().warn("Extremely large fused time from {} to {}, nodes {}".format(sum_time, fused_time, nodes_to_fuse))
                # my_env["XLA_CLUSTER_SPEC"] = '/tmp/xla_spec.txt'
                # my_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
                # self._estimate_time_real_replay(self.base_cmd, my_env, verbose=True)
                # exit(-1)  
            return fused_time
        else:
            # return self.cost_models[pid].predict(nodes_to_fuse)
            predicted_time, brkdn_dict = self.cost_models["default"].predict(nodes_to_fuse)
            return predicted_time / 1000

    def _parse_node_attr(self, _dag, new_name, avg):
        ''' Parse the fused node attribute corresponding to `new_name` and set _dag
        * If new_name has been cached, directly set _dag with the cached attributes
        * Otherwise, combine the attribution of all original nodes
            * If avg is not given, query the cost model
            * Otherwize, use the given avg (TODO Disabled, since we have cached attr in self.node_attr_cache)
        
        Return
        ------
        avg: average time
        '''
        if new_name in self.node_attr_cache:
            nx.set_node_attributes(_dag, {new_name: self.node_attr_cache[new_name]})
            # _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            ns = self._get_defused_node_names(new_name)
            attrs = self.node_attr_cache[ns[0]].copy()
            for idx in range(1, len(ns)):
                self._combine_attr_except_avg(attrs, attrs, self.node_attr_cache[ns[idx]])
            # combine attr avg
            attrs["avg"] = avg
            ### set and cache the attribute
            nx.set_node_attributes(_dag, {new_name: attrs})
            self._cache_node_attr(new_name, _dag.nodes[new_name])

        ### TODO (huhanpeng): apply to other GPUs, cache the same attribute for corresponding operators on other GPUs
        for other_name in self.opt._debug_convert_to_other_machines(new_name):
            if other_name in self.node_attr_cache:
                continue
            self._cache_node_attr(other_name, _dag.nodes[new_name])

        return self.node_attr_cache[new_name]["avg"]

    def _op_fusion(self, _dag, _pkg: PKGraph, u_, v_):
        # test if two nodes can be fused
        if _pkg.can_contract_edge(u_, v_):
            nodes_to_add = []
            nodes_to_remove = []
            MAX_FUSION_EXLPORE_DEPTH = 20
            for explore_idx in range(MAX_FUSION_EXLPORE_DEPTH):
                # pkg must contract after calling _fuse_pair since it can
                # throw errors
                SingleLogger().debug(bcolors.CBLUE + "[Fusion] The {} th exploration...".format(explore_idx) + bcolors.ENDC)
                avg = self._fuse_pair(_dag, u_, v_)
                _pkg.contract_edge(u_, v_)

                nodes_to_add.append(u_+"+"+v_)
                nodes_to_remove += [u_, v_]

                ### apply the same strategy to other GPUs
                ul = self.opt._debug_convert_to_other_machines(u_)
                vl = self.opt._debug_convert_to_other_machines(v_)
                for u__, v__ in zip(ul, vl):
                    assert _pkg.can_contract_edge(u__, v__)
                    ### TODO (huhanpeng): since we assume data parallel
                    ### use the same avg for the fused operators
                    self._fuse_pair(_dag, u__, v__, avg=avg)
                    _pkg.contract_edge(u__, v__)
                    nodes_to_add.append(u__+"+"+v__)
                    nodes_to_remove += [u__, v__]
                
                if not self.explore_fusion:
                    break

                is_end = True
                if avg > self._get_node_avg(u_) + self._get_node_avg(v_):
                    succs = [s for s in _dag.successors(
                        u_+"+"+v_) if (self._wrap_can_fuse_to_b(_pkg, u_+"+"+v_, s))] 
                    if len(succs) > 0:
                        heats = [self.opt._combine_weight(None, self.opt._get_heat_from_history(s)) for s in succs]
                        u_ = u_+"+"+v_
                        st_idx = self.opt.select_one_stategy(heats, range(len(succs)))
                        v_ = succs[st_idx]
                        is_end = False
                if is_end:
                    break
            nodes_to_add = set(nodes_to_add)
            nodes_to_remove = set(nodes_to_remove)
            return True, nodes_to_add.difference(nodes_to_remove), nodes_to_remove.difference(nodes_to_add)
        else:
            return False, None, None

    def _fuse_pair(self, _dag, u_, v_, avg=None):
        # print("fuse {} {}".format(u_, v_))
        ### Cache the node attributes in case they will be used when de-fuse
        if u_ not in self.node_attr_cache:
            self._cache_node_attr(u_, _dag.nodes[u_])
        if v_ not in self.node_attr_cache:
            self._cache_node_attr(v_, _dag.nodes[v_])

        SingleLogger().debug("Fuse {} ({}) and {} ({})".format(u_, self.node_attr_cache[u_]["avg"],
            v_, self.node_attr_cache[v_]["avg"]))

        new_name = self._concat_name(u_, v_)
        ### Add new nodes and get the attibute
        if new_name in self.node_attr_cache:
            _dag.add_node(new_name, **self.node_attr_cache[new_name])
        else:
            ### Calculate the new attribute
            if avg is None:
                # an error is thrown here if cannot combine
                # we must put all modification of dag after this line
                avg = self._get_node_avg(new_name, verbose=False)
            _dag.add_node(new_name)
            self._combine_nodes_attr(_dag, new_name, u_, v_, avg)
            ### cache the attribute
            self._cache_node_attr(new_name, _dag.nodes[new_name])

        ### Update edges
        for in_, _ in _dag.in_edges(u_):
            if in_ != v_:
                _dag.add_edge(in_, new_name)
        for in_, _ in _dag.in_edges(v_):
            if in_ != u_:
                _dag.add_edge(in_, new_name)

        for out_ in _dag.successors(u_):
            if out_ != v_:
                _dag.add_edge(new_name, out_)
        for out_ in _dag.successors(v_):
            if out_ != u_:
                _dag.add_edge(new_name, out_)

        ### Remove current nodes
        _dag.remove_node(u_)
        _dag.remove_node(v_)

        assert u_ not in _dag.nodes
        assert v_ not in _dag.nodes
        assert u_ in self.node_attr_cache and "avg" in self.node_attr_cache[u_]
        assert v_ in self.node_attr_cache and "avg" in self.node_attr_cache[v_]
        return self.node_attr_cache[new_name]["avg"]

    def _op_defusion(self, _dag, _pkg: PKGraph, target, components):
        nodes2add = []
        nodes2rm = []

        _, new_node_names = self._defuse_node(_dag, _pkg, target, components)
        nodes2add += new_node_names
        nodes2rm.append(target)

        ### apply the same strategy to other GPUs
        target_l = self.opt._debug_convert_to_other_machines(target)
        components_l = [tuple([self.opt._debug_convert_to_other_machines(node) for node in comp]) for comp in components]
        for idx, target_ in enumerate(target_l):
            components_ = [tuple([node_l[idx] for node_l in comp]) for comp in components_l]
            _, new_node_names_ = self._defuse_node(_dag, _pkg, target_, components_)
            nodes2add += new_node_names_
            nodes2rm.append(target_)

        return True, set(nodes2add), set(nodes2rm)

    def _defuse_node(self, _dag, _pkg, target, components):
        avgs = []
        component_names = get_concated_names(components)
        for new_node_name in component_names:
            avg = self._get_node_avg(new_node_name, verbose=False)
            avgs.append(avg)
        _pkg.split_node(target, components)

        # override successors for BW nodes if searching along with tensor fusion
        if "++" in self.opt.cst_md_mng.strategy2model:
            def succ_overide_func(_node):
                if "BW" not in _node:
                    return None
                else:
                    assert "+" not in _node
                    return self.opt.cst_md_mng.strategy2model["++"].get_current_comm_from_unfused_bw(_node)
            def pred_override_func(_node):
                if "UPDATE" not in _node:
                    return None
                else:
                    assert "+" not in _node
                    return self.opt.cst_md_mng.strategy2model["++"].get_current_comm_from_unfused_update(_node)
        else:
            succ_overide_func = None
            pred_override_func = None

        defuse_nodes_inplace_nx(_dag, _pkg, target, components, 
                                succ_override=succ_overide_func,
                                pred_override=pred_override_func)
        for idx, new_node_name in enumerate(component_names):
            self._parse_node_attr(_dag, new_node_name, avgs[idx])
        return True, component_names

    def apply(self, s, __dag, __pkg):
        op, target, next_ = s
        ### TODO (huhanpeng): need further add other optimization techiniques
        if op == "+":
            ### Fuse two nodes
            return self._op_fusion(__dag, __pkg, target, next_)
        elif op == "-":
            return self._op_defusion(__dag, __pkg, target, next_)

    def _check_dag_avg(self, G: nx.DiGraph):
        for n in G.nodes():
            if "Comm" in n or "host0.rank0" not in n:
                continue
            if "+" not in n:
                continue
            fused_avg = self.node_attr_cache[n]["avg"]
            avg_sum = 0
            all_fuse_nodes = n.split("+")
            for _n in all_fuse_nodes:
                avg_sum += self.node_attr_cache[_n]["avg"]
            print("Fuse {} nodes, predicted avg: {}, fused nodes avg sum: {}".format(
                len(all_fuse_nodes), fused_avg, avg_sum))
        raise RuntimeError()

    def is_fusion_better(self, long_name_u, long_name_v, _dag, _pkg, dp_state, no_theorem=False):
        if (long_name_u, long_name_v) not in _dag.edges() or not self._wrap_can_fuse_to_b(_pkg, long_name_u, long_name_v):
            return False, None
        
        fused_time = self._get_node_avg(long_name_u+"+"+long_name_v, verbose=False)
        if fused_time is None:
            return False, None

        if no_theorem:
            t_null = self.opt.estimate_time_related_to_comp([long_name_u, long_name_v], _dag, dump_path=None)

            G_star = _dag.copy()
            pkg_star = _pkg.copy()

            self.apply(("+", long_name_u, long_name_v), G_star, pkg_star)
            t_fuse = self.opt.estimate_time_related_to_comp([long_name_u+"+"+long_name_v], G_star, dump_path=None)

            if t_fuse < t_null:
                return True, fused_time
            else:
                return False, fused_time

        # ref_pid = parse_pid_from_name(long_name_u)
        # comm_dur_u = self.opt._op_to_coarse_grained_comm_time(long_name_u, _dag, ref_pid) ### q_{n-1}^d
        # comp_dur_v = self._get_node_avg(long_name_v, verbose=False) ### p_{n}^d
        # comp_dur_u = self._get_node_avg(long_name_u, verbose=False) ### p_{n-1}^d

        comm_dur_u = dp_state.q_d[-2]
        comp_dur_v = dp_state.p_d[-1]
        comp_dur_u = dp_state.p_d[-2]

        if comm_dur_u < comp_dur_v + comp_dur_u - fused_time:
            ### operator fusion leads to better performance
            return True, fused_time
        else:
            return False, fused_time