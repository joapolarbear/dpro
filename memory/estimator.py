import ml_platform
import networkx as nx

class MemoryEstimator:
    """Memory Estimator


    """
    MEMORY_LIST_MOD = "memory_lists"
    LISTS = ["WHITE_LIST", "CWISE_LIST"]

    def __init__(self, platform):
        self.platform = getattr(ml_platform, platform.lower())
        if hasattr(self.platform, self.MEMORY_LIST_MOD):
            self.lists = getattr(self.platform, self.MEMORY_LIST_MOD)
        else:
            raise NotImplementedError("Memory Estimator Does Not Support %s" % platform)
        
        # check validity
        for list in self.LISTS:
            if not hasattr(self.lists, list):
                raise NotImplementedError("Platform %s lacks %s" % (platform, list))
        

    def estimate(self, dag, param_dict):
        """[summary]

        Args:
            dag ([type]): [description]
            param_dict ([type]): [description]
        """
        print("Nodes num:%d" % (len(dag)))
        nodes = list(nx.topological_sort(dag))
        for node in nodes:
            raw_name, _ = param_dict.metainfo.standarize_name(node)
            # print(raw_name)
            if param_dict.metainfo.in_metadata(raw_name):
                print(raw_name, param_dict.metainfo.ret_output_size_inB(raw_name))
                input()
        print("Done!")
        return 0

    def _estimate_model_size(self):
        pass
