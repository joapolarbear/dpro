from cost_model.base import _BaseCostModel


class HalvingBatchSize(_BaseCostModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.token = []

    def init_search_space(self, *args, **kwargs):
        return [], []

    def apply(self, s, __dag, __pkg):
        self.opt.memory_estimator.batch_size /= 2
        return True, [], []
