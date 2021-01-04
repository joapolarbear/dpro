class _BaseCostModel:
    def __init__(self, opt):
        self.opt = opt
        self.dag = self.opt.dag
        ### token is the indendifier of each optimization technique
        self.token = None

    def init_search_space(self, *args, **kwargs):
        raise NotImplementedError()

    def apply(self, s, __dag, __pkg):
        raise NotImplementedError()

    def load_init_ckpt(self):
        ''' Load the init states BEFORE the search process, 
            reduce the preprocessing time, 
            e.g., XLA cost model need to init partition'''
        raise NotImplementedError()

    def load_ckpt(self):
        ''' Load checkponits during the search process '''
        raise NotImplementedError()

    def checkpoint(self):
        raise NotImplementedError()

