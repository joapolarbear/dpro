import networkx as nx
import time
import os
import pickle
import numpy as np
import ujson as json
from tqdm import tqdm

from ..arg_utils import SingleArg
from ..trace_utils import *
from ._xla.pk_graph import PKGraph
from .base import _BaseGraphPass
from ._mixed_precision.amp_pred import AMPPredictor

args_ = SingleArg().args

class AMPGraphPass(_BaseGraphPass):
    def __init__(self, opt):
        super().__init__(opt)
        ### AMP predictor
        self.amp_predictor = AMPPredictor(self.meta_info)
        self.token = [">", "<"]

    def init_search_space(self, candidates, _dag: nx.DiGraph, _pkg: PKGraph):
        search_space = []
        weights = []
        for n, l in candidates:
            # node heat
            # heat = self.opt._get_heat_from_history(n)
            # ### Nodes that have never been fused
            # cat = parse_cat_fine_grained(n)
            # pid = parse_pid_from_name(n)

            ### check if mixed precision can be used for this node
            if self.amp_predictor.is_need_amp(_dag, n):
                search_space.append((">", n, None))
                weights.append(l)

        # return [(">", "host1.rank0->BW.gradients/resnet50/conv2_block3_1_conv/Conv2D_grad/Conv2DBackpropFilter", None)], [1]
        SingleLogger().info("MP Cost Model init {} strategies.".format(len(search_space)))
        return search_space, weights

    def apply(self, s, __dag, __pkg):
        op, target, _ = s
        nodes_introduced = self.amp_predictor.quantize(__dag, target)
        ### apply this strategy to other GPUs' corresponding operators
        ### we assume data parallel, use the same model
        on_other_ranks = self.opt._debug_convert_to_other_machines(target)
        for target in on_other_ranks:
            nodes_introduced += self.amp_predictor.quantize(__dag, target)
        return True, nodes_introduced, []

    def checkpoint(self):
        self.amp_predictor.checkpoint()

    def load_ckpt(self):
        self.amp_predictor.load_ckpt()

    def load_init_ckpt(self):
        init_ckpt_path = os.path.join(ROOT_PATH, "amp_init_ckpt.pickle")
        if os.path.isfile(init_ckpt_path):
            with open(init_ckpt_path, "rb") as f:
                G, PKG, trajectory, _cast_cnt, _num_nonvar_casts_to_fp16, _op_status = pickle.load(f)
                self.amp_predictor.cast_cnt = _cast_cnt
                self.amp_predictor.num_nonvar_casts_to_fp16 = _num_nonvar_casts_to_fp16
                self.amp_predictor.op_status = _op_status
            SingleLogger().info("Reading init graph from cache.")
        else:
            G = self.dag.copy()
            PKG = PKGraph(G)

            source_nodes = [n for n in G.nodes() if "host0.rank0" in n]
            trajectory = []
            for n in tqdm(source_nodes, total=len(source_nodes)):
                if self.amp_predictor.is_need_amp(G, n):
                    s = (">", n, None)
                    trajectory.append(s)
                    self.apply(s, G, PKG)

            with open(init_ckpt_path, "wb") as f:
                pickle.dump([G, PKG, trajectory, self.amp_predictor.cast_cnt,
                             self.amp_predictor.num_nonvar_casts_to_fp16, self.amp_predictor.op_status], f)
            SingleLogger().info("Graph cache dumped to {}.".format(init_ckpt_path))

        SingleLogger().info("Successfully initialized mixed precision strategy with {} cast(s).".format(
            self.amp_predictor.num_nonvar_casts_to_fp16))
        return G, PKG, trajectory

    def flush(self, is_accept: bool):
        self.amp_predictor.flush(is_accept)
