from functools import partial
from logger_utils import SingleLogger
from cost_model.base import _BaseCostModel
from .recomputation import get_recomputation_edited_graph
from replay import Replayer
from copy import deepcopy


class MemoryCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["reduce_batch_size", "recomputation"]

    def init_search_space(self, candidates, dag, pkg):
        candidate_strategies = []
        candidate_weights = []
        for strategy in self.token:
            func = self.func_factory(strategy)
            estimated_time, estimated_memory = func(dag, self.opt.clct)
            if estimated_memory > self.opt.memory_budget:
                continue
            candidate_strategies.append((strategy, None, None))
            candidate_weights.append(1./estimated_time)

        return candidate_strategies, candidate_weights

    def apply(self, s, dag, pkg):
        if s[0] == "reduce_batch_size":
            if self.opt.memory_estimator.batch_size > 1:
                self.opt.memory_estimator.batch_size /= 2
        elif s[0] == "recomputation":
            get_recomputation_edited_graph(
                dag, self.opt.memory_estimator.schedule, "speed")
        else:
            raise NotImplementedError
        return True, [], []

    @staticmethod
    def _get_execution_time(dag, clct):
        replayer = Replayer(dag=dag, _step_num=1,
                            leaf_dirs=clct.all_prefix_list(),
                            dump_path=clct.pm.path,
                            comm_backend=clct.comm_backend,
                            byteps_graph=clct.byteps_graph)
        step_end_time_ms = [
            t / 1000 for t in replayer.replayAndDelay(None).values()]
        return max(step_end_time_ms)

    def func_factory(self, strategy):
        func_name = "_get_estimated_time_and_memory_of_" + strategy
        return getattr(self, func_name)

    def _get_estimated_time_and_memory_of_reduce_batch_size(self, dag, clct):
        estimated_time = self._get_execution_time(dag, clct) * 1.2

        self.opt.memory_estimator.batch_size /= 2
        estimated_memory = self.opt.memory_estimator.estimate(
            dag, clct.para_dict)
        self.opt.memory_estimator.batch_size *= 2  # restore

        SingleLogger().info("Estimated time and memory after applying reducing batch size: {:.2f}ms, {:.2f}GB".format(
            estimated_time, estimated_memory
        ))
        return estimated_time, estimated_memory

    def _get_estimated_time_and_memory_of_recomputation(self, dag, clct):
        dag_copy = deepcopy(dag)
        get_recomputation_edited_graph(
            dag_copy, self.opt.memory_estimator.schedule, "speed")
        estimated_time = self._get_execution_time(dag_copy, clct)

        estimated_memory = self.opt.memory_estimator.estimate(
            dag, clct.para_dict)

        SingleLogger().info("Estimated time and memory after applying recomputation: {:.2f}ms, {:.2f}GB".format(
            estimated_time, estimated_memory
        ))
        return estimated_time, estimated_memory
