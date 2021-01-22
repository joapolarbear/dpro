from cost_model.base import _BaseCostModel
from .recomputation import get_recomputation_edited_graph


class MemoryCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["reduce_batch_size", "amp", "recomputation"]

    def init_search_space(self, candidates, dag, pkg):
        # TODO(yuchen): calculate weight from replay's performance
        return [("reduce_batch_size", None, None), ("recomputation", None, None)], [1, 1]

    def apply(self, s, dag, pkg):
        if s[0] == "reduce_batch_size":
            if self.opt.memory_estimator.batch_size > 1:
                self.opt.memory_estimator.batch_size /= 2
        elif s[0] == "recomputation":
            status = get_recomputation_edited_graph(
                dag, self.opt.memory_estimator.schedule, "speed")
             
        else:
            raise NotImplementedError
        return True, [], []
