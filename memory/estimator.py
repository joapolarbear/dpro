from pprint import pprint
from functools import partial

import ml_platform
import networkx as nx


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_rank0_forward_nodes(nodes):
    def _is_rank0(name):
        if name.startswith("traces_0.rank0"):
            return True
        return False

    def _is_forward(name):
        if "->FW." in name:
            return True
        return False

    return [remove_prefix(node, "traces_0.rank0->FW.") for node in nodes
            if _is_rank0(node) and _is_forward(node)]


class MemoryEstimator:

    MEMORY_LISTS_MODULE = "memory_lists"

    def __init__(self, platform):
        self.memory_lists = self._get_platform_memory_lists(platform)

    def _get_platform_memory_lists(self, platform):
        module = getattr(ml_platform, platform.lower())
        if hasattr(module, self.MEMORY_LISTS_MODULE):
            self.lists = getattr(module, self.MEMORY_LISTS_MODULE)
        else:
            raise NotImplementedError(
                "Memory Estimator Does Not Support %s" % platform)

        return self.lists

    def estimate(self, dag, param_dict):
        """[summary]

        Args:
            dag ([type]): [description]
            param_dict ([type]): [description]
        """
        nodes = dag.nodes
        selected_nodes = get_rank0_forward_nodes(nodes)
        print(selected_nodes)

        return 0
