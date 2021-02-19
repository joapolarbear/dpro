from functools import partial
from memory.gradient_accumulation import get_gradient_accumulation_edited_graph
from logger_utils import SingleLogger
from cost_model.base import _BaseCostModel
from .recomputation import get_recomputation_edited_graph
from replay import Replayer
from copy import deepcopy
from .utils import *


def get_execution_time(dag, clct):
    replayer = Replayer(dag=dag, _step_num=1,
                        leaf_dirs=clct.all_prefix_list(),
                        dump_path=clct.pm.path,
                        comm_backend=clct.comm_backend,
                        byteps_graph=clct.byteps_graph)
    step_end_time_ms = [
        t / 1000 for t in replayer.replayAndDelay(None).values()]
    return max(step_end_time_ms)

def has_recomputation(schedule):
    for op in schedule.operators:
        if op.requires_grad is False:
            return True
    return False

class MemoryCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["gradient_accumulation", "recomputation"]
        self.cnts = [0, 0]

    def init_search_space(self, candidates, dag, pkg):
        candidate_strategies = []
        candidate_weights = []
        for i, strategy in enumerate(self.token):
            if strategy == "recomputation":
                if has_recomputation(self.opt.memory_estimator.schedule):
                    continue

            func = self.func_factory(strategy)
            estimated_time, estimated_memory = func(dag, self.opt.clct)
            if estimated_memory > self.opt.memory_budget:
                continue
            candidate_strategies.append((strategy, None, None))
            candidate_weights.append(1./(self.cnts[0] + 1))

        return candidate_strategies, candidate_weights

    def apply(self, s, dag, pkg):
        if s[0] == "gradient_accumulation":
            if self.opt.memory_estimator.batch_size > 1:
                self.opt.memory_estimator.batch_size /= 2
            get_gradient_accumulation_edited_graph(dag)
            self.cnts[0] += 1
        elif s[0] == "recomputation":
            get_recomputation_edited_graph(
                dag, self.opt.memory_estimator.schedule, "speed")
            self.cnts[1] += 1
        else:
            raise NotImplementedError
        return True, [], []

    def func_factory(self, strategy):
        func_name = "_get_estimated_time_and_memory_of_" + strategy
        return getattr(self, func_name)

    def _get_estimated_time_and_memory_of_gradient_accumulation(self, dag, clct):
        dag_copy = deepcopy(dag)
        get_gradient_accumulation_edited_graph(dag_copy)
        estimated_time = get_execution_time(dag_copy, clct)

        self.opt.memory_estimator.batch_size /= 2
        estimated_memory = self.opt.memory_estimator.estimate(
            dag, clct.para_dict)
        self.opt.memory_estimator.batch_size *= 2  # restore

        SingleLogger().info("Estimated time and memory after applying gradient accumulation: {:.2f}ms, {:.2f}GB".format(
            estimated_time, estimated_memory
        ))
        return estimated_time, estimated_memory

    def _get_estimated_time_and_memory_of_recomputation(self, dag, clct):
        dag_copy = deepcopy(dag)
        prev_nodes = deepcopy(self.opt.memory_estimator.schedule.operators)
        get_recomputation_edited_graph(
            dag_copy, self.opt.memory_estimator.schedule, "speed")
        estimated_time = get_execution_time(dag_copy, clct)

        estimated_memory = self.opt.memory_estimator.estimate(
            dag, clct.para_dict)
        
        # dirty implementation ...
        for op, prev_op in zip(self.opt.memory_estimator.schedule.operators, prev_nodes):
            op.requires_grad = prev_op.requires_grad
        
        SingleLogger().info("Estimated time and memory after applying recomputation: {:.2f}ms, {:.2f}GB".format(
            estimated_time, estimated_memory
        ))
        return estimated_time, estimated_memory


class IncreasingBatchSizeCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["increase_batch_size"]
        self.cnt = 0

    def init_search_space(self, candidates, dag, pkg):
        candidate_strategies = []
        candidate_weights = []
        for strategy in self.token:
            func = self.func_factory(strategy)
            estimated_time, estimated_memory = func(dag, self.opt.clct)
            candidate_strategies.append((strategy, None, None))
            candidate_weights.append(1./(self.cnt + 1))

        return candidate_strategies, candidate_weights

    def apply(self, s, dag, pkg):
        # TODO(yuchen): determine batch size upper bound
        if self.opt.memory_estimator.batch_size < 1024:
            self.opt.memory_estimator.batch_size *= 2
            self._update_dag(dag)
            self.cnt += 1
        return True, [], []

    def func_factory(self, strategy):
        func_name = "_get_estimated_time_and_memory_of_" + strategy
        return getattr(self, func_name)

    def _update_dag(self, dag):
        computation_nodes = filter_out_comm_nodes(dag)
        update_time_by_scale(dag.subgraph(computation_nodes), 0.8)

    def _get_estimated_time_and_memory_of_increase_batch_size(self, dag, clct):
        dag_copy = deepcopy(dag)
        self._update_dag(dag_copy)

        estimated_time = get_execution_time(dag_copy, clct)

        self.opt.memory_estimator.batch_size *= 2
        estimated_memory = self.opt.memory_estimator.estimate(
            dag, clct.para_dict)
        self.opt.memory_estimator.batch_size /= 2  # restore

        SingleLogger().info("Estimated time and memory after applying increasing batch size: {:.2f}ms, {:.2f}GB".format(
            estimated_time, estimated_memory
        ))
        return estimated_time, estimated_memory

    def load_init_ckpt(self, G_prime=None):
        return None, None, []

    def load_ckpt(self):
        return

    def checkpoint(self):
        return

    def flush(self, is_accept):
        return
