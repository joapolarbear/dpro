import ml_platform

from node import Node


class Schedule:
    def __init__(self, platform):
        self._parameters = []
        self._operators = []
        self.lists = self._get_platform_memory_lists(platform)

    def add(self, node):
        """add node into schedule

        Args:
            node ([Node]): operator

        Returns:
            [bool]: Status
        """
        if not isinstance(node, Node):
            return False

        if node.is_parameter():
            self._parameters.append(node)
        elif node.is_valid() and self._is_in_whitelist(node):
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

    def _get_platform_memory_lists(self, platform):
        module = getattr(ml_platform, platform.lower())
        if hasattr(module, "memory_lists"):
            self.lists = getattr(module, "memory_lists")
        else:
            raise NotImplementedError(
                "Memory Estimator Does Not Support %s" % platform)

        return self.lists
