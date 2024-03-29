
class OptApplyStrategyError(Exception):
    pass


class OptNoValidStrategyError(Exception):
    pass


class OptQueryCostModelError(Exception):
    pass

class _BaseGraphPass:
    def __init__(self, opt):
        self.opt = opt
        self.dag = self.opt.dag
        ### token is the indendifier of each optimization technique
        self.token = None
        self.meta_info = self.opt.clct.para_dict

        self.ckpt_dir = self.opt.ckpt_dir
        self.spec_dir = self.opt.spec_dir

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
    
    def flush(self, is_accept):
        ''' A strategy may be rejected, so the internal states of 
            * cost model should not be changed in apply() 
            * but be changed when the strategy is accepted
        Each cost model may cache the change of the internal states, 
        and flush the change when this function is called
        '''
        raise NotImplementedError()

