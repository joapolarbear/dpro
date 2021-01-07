from functools import partial

from node import Node
from schedule import Schedule

import networkx as nx

DEL = "->"
RANK0_PREFIX = "traces_0.rank0"
FORWARD_CAT = "FW."


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_forward_nodes(nodes):
    # TODO(yuchen): Not expandable
    def _is_forward(name):
        if name.startswith(DEL.join([RANK0_PREFIX, FORWARD_CAT]) + "bert"):
            return True
        return False

    return list(filter(_is_forward, nodes))


def remove_node_prefix(nodes, prefix):
    func = partial(remove_prefix, prefix=prefix)
    return list(map(func, nodes))


def get_leaf_nodes(dag):
    return [node for node in dag.nodes if dag.out_degree(node) == 1
            and dag.in_degree(node) == 0]


class MemoryEstimator:

    def __init__(self, platform):
        self.platform = platform
        self.default_batch_size = 32  # TODO(yuchen): should read from graph
        self.batch_size = self.default_batch_size

    def _compose_operator_schedule(self, dag, param_dict) -> Schedule:
        forward_nodes = get_forward_nodes(dag.nodes)
        forward_graph = dag.subgraph(forward_nodes).copy()

        leaf_nodes = get_leaf_nodes(forward_graph)
        forward_graph.remove_nodes_from(leaf_nodes)
        leaf_nodes = remove_node_prefix(
            leaf_nodes, DEL.join([RANK0_PREFIX, FORWARD_CAT]))

        sorted_forward_nodes = nx.topological_sort(forward_graph)
        sorted_forward_nodes = remove_node_prefix(
            sorted_forward_nodes, DEL.join([RANK0_PREFIX, FORWARD_CAT]))

        metadata = param_dict.metainfo.tf_meta
        operator_schedule = Schedule(self.platform)
        for name in leaf_nodes:
            node = Node.from_metadata(name, metadata)
            operator_schedule.add(node)

        for name in sorted_forward_nodes:
            node = Node.from_metadata(name, metadata)
            operator_schedule.add(node)

        return operator_schedule

    def _simulate_memory_allocation(self, operator_schedule) -> float:
        peak_size = 0
        total_activations = 0
        total_param_size = 0

        def _get_param_size():
            # including optimizer states, such as momentums
            nonlocal total_param_size
            for param in operator_schedule.parameters:
                total_param_size += param.get_output_size()

        def _simulate_forward_propagation():
            nonlocal total_activations, peak_size
            for op in operator_schedule.operators:
                if op.requires_grad:
                    total_activations += op.get_output_size()

                temp_size = op.get_temp_size()
                peak_size = max(peak_size, total_activations + temp_size)

        def _simulate_backward_propagation():
            nonlocal total_activations, peak_size
            for i, op in reversed(list(enumerate(operator_schedule.operators))):
                output_grad_size = op.get_output_size()

                j = i
                while j >= 0 and not operator_schedule.operators[j].requires_grad:
                    total_activations += operator_schedule.operators[j].get_output_size(
                    )
                    operator_schedule.operators[j].requires_grad = True
                    j -= 1

                temp_size = op.get_temp_size()
                peak_size = max(peak_size, total_activations +
                                output_grad_size + temp_size)
                total_activations -= op.get_output_size()

        def _byte_to_GB(size):
            return size / (1000**3)

        _get_param_size()
        _simulate_forward_propagation()
        _simulate_backward_propagation()

        peak_size, total_param_size = _byte_to_GB(
            peak_size), _byte_to_GB(total_param_size)

        peak_size *= self.batch_size / self.default_batch_size

        # TODO(yuchen): Not expandable. This is for Adam.
        return peak_size + total_param_size / 3 * 8

    def estimate(self, dag, param_dict):
        """Estimate memory usage based on computation graph

        Args:
            dag (nx.DiGraph): computation graph
            param_dict (ParameterDict): operator information

        Returns:
            [float]: memory usage in GB
        """
        operator_schedule = self._compose_operator_schedule(dag, param_dict)
        return self._simulate_memory_allocation(operator_schedule)
