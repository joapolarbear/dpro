import json
from pprint import pprint

import ml_platform
import networkx as nx
from trace_utils import FileName



    

class MemoryEstimator:

    MEMORY_LISTS_MODULE = "memory_lists"

    def __init__(self, platform, path_manager):
        self.memory_lists = self._get_platform_memory_lists(platform)
        self.graph_def = self._read_graph_def_from_json(path_manager)

    def _get_platform_memory_lists(self, platform):
        module = getattr(ml_platform, platform.lower())
        if hasattr(module, self.MEMORY_LISTS_MODULE):
            self.lists = getattr(module, self.MEMORY_LISTS_MODULE)
        else:
            raise NotImplementedError(
                "Memory Estimator Does Not Support %s" % platform)

        return self.lists

    def _read_graph_def_from_json(self, path_manager):
        json_path = path_manager.search(FileName.GRAPHDEF.value)
        with open(json_path, "r") as f:
            content = json.load(f)
        return content["node"]

    def estimate(self, dag, param_dict):
        """[summary]

        Args:
            dag ([type]): [description]
            param_dict ([type]): [description]
        """

        return 0
