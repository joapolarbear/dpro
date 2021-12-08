from .node import Node
from .schedule import Schedule
from .utils import *

import networkx as nx


class MemoryEstimator:

    def __init__(self, platform):
        self.platform = platform
        self.default_batch_size = 32  # TODO(yuchen): should read from graph
        self.batch_size = self.default_batch_size
        self._schedule = None
        self._cached_result = 0

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, val):
        self._schedule = val

    def _compose_operator_schedule(self, dag, param_dict) -> Schedule:
        forward_nodes = get_forward_nodes(dag.nodes)
        forward_graph = dag.subgraph(forward_nodes).copy()

        leaf_nodes = get_leaf_nodes(forward_graph)
        forward_graph.remove_nodes_from(leaf_nodes)
        leaf_nodes = remove_nodes_prefix(
            leaf_nodes, DEL.join([RANK0_PREFIX, FORWARD_CAT]))

        sorted_forward_nodes = nx.topological_sort(forward_graph)
        sorted_forward_nodes = remove_nodes_prefix(
            sorted_forward_nodes, DEL.join([RANK0_PREFIX, FORWARD_CAT]))

        metadata = param_dict.metainfo.tf_meta
        operator_schedule = Schedule(self.platform)
        trace_times = nx.get_node_attributes(dag, "avg")
        trace_times = {remove_node_prefix(k, DEL.join(
            [RANK0_PREFIX, FORWARD_CAT])): v for k, v in trace_times.items()}
        for node in leaf_nodes:
            op = Node.from_metadata(
                node, metadata, trace_times[node])
            operator_schedule.add(op)

        for node in sorted_forward_nodes:
            op = Node.from_metadata(node, metadata, trace_times[node])
            operator_schedule.add(op)

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
            restore_list = []
            for i, op in reversed(list(enumerate(operator_schedule.operators))):
                output_grad_size = op.get_output_size()

                j = i
                while j >= 0 and not operator_schedule.operators[j].requires_grad:
                    total_activations += operator_schedule.operators[j].get_output_size(
                    )
                    operator_schedule.operators[j].requires_grad = True
                    restore_list.append(operator_schedule.operators[j])
                    j -= 1

                temp_size = op.get_temp_size()
                peak_size = max(peak_size, total_activations +
                                output_grad_size + temp_size)
                total_activations -= output_grad_size

            # restore
            for op in restore_list:
                op.requires_grad = False

        def _byte_to_GB(size):
            return size / (1000**3)

        _get_param_size()
        _simulate_forward_propagation()
        _simulate_backward_propagation()

        peak_size, total_param_size = _byte_to_GB(
            peak_size), _byte_to_GB(total_param_size)

        peak_size *= self.batch_size / self.default_batch_size

        # TODO(yuchen): Not expandable. This is for Adam.
        total = peak_size + total_param_size / 3 * 8
        self._cached_result = total
        return total

    def estimate(self, dag, param_dict):
        """Estimate memory usage based on computation graph

        Args:
            dag (nx.DiGraph): computation graph
            param_dict (ParameterDict): operator information

        Returns:
            [float]: memory usage in GB
        """
        if not self._schedule:
            self._schedule = self._compose_operator_schedule(dag, param_dict)
        return self._simulate_memory_allocation(self._schedule)

    @property
    def cached_memory_estimation(self):
        return self._cached_result
