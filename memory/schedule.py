import ml_platform

from node import Node


class Schedule:
    def __init__(self, platform):
        self._parameters = []
        self._operators = []
        self._node_collection = {}
        self.lists = self._get_platform_memory_lists(platform)

    def add(self, node):
        """add node into schedule and determine whether it runs in place 

        Args:
            node ([Node]): operator

        Returns:
            [bool]: Status
        """
        if not isinstance(node, Node):
            return False

        self._node_collection[node.name] = node

        if node.is_parameter():
            self._parameters.append(node)
        elif node.is_valid() and self._is_in_whitelist(node):
            self._set_inplace(node)
            self._operators.append(node)
        else:
            return False

        return True

    @property
    def parameters(self):
        return self._parameters

    @property
    def operators(self):
        return self._operators

    def _is_in_whitelist(self, node):
        if node.op not in self.lists.WHITE_LIST:
            return False
        return True

    def _should_inplace(self, input_node, output_node):
        if output_node.op not in self.lists.CWISE_LIST:
            return False

        if input_node.inplace:
            return False

        if input_node.dtype != output_node.dtype:
            return False

        if input_node.get_num_ele() != output_node.get_num_ele():
            return False

        return True

    def _set_inplace(self, node):
        input_names = node.input
        for input_name in input_names:
            input_node = self._node_collection.get(input_name)
            if input_node and self._should_inplace(input_node, node):
                node.inplace = True
                break

    def _get_platform_memory_lists(self, platform):
        module = getattr(ml_platform, platform.lower())
        if hasattr(module, "memory_lists"):
            self.lists = getattr(module, "memory_lists")
        else:
            raise NotImplementedError(
                "Memory Estimator Does Not Support %s" % platform)

        return self.lists
