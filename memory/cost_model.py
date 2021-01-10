from cost_model.base import _BaseCostModel


class MemoryCostModel(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = ["reduce_batch_size", "amp", "recomputation"]

    def init_search_space(self, candidates, dag, pkg):
        return [("reduce_batch_size", None, None)], [1]

    def apply(self, s, __dag, __pkg):
        if s[0] == "reduce_batch_size":
            if self.opt.memory_estimator.batch_size > 1:
                self.opt.memory_estimator.batch_size /= 2
        else:
            raise NotImplementedError
        return True, [], []
