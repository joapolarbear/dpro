import re
import networkx as nx
from utils import *
from logger_utils import SingleLogger


class CheckpointsSelector:
    @classmethod
    def get_checkpoint_selector(cls, mode):
        if mode == "speed":
            return SpeedCheckpointsSelector()
        elif mode == "memory":
            return MemoryCheckpointsSelector()
        else:
            raise ValueError("%s is not found" % mode)

    @staticmethod
    def select_checkpoints(schedule):
        raise NotImplementedError


class SpeedCheckpointsSelector(CheckpointsSelector):
    @staticmethod
    def select_checkpoints(schedule):
        return list(filter(lambda n: len(re.findall("conv2d|conv|matmul", n.op))
                           > 0, schedule.operators))


class MemoryCheckpointsSelector(CheckpointsSelector):
    @staticmethod
    def select_checkpoints(schedule):
        # TODO(yuchen): https://arxiv.org/pdf/1604.06174.pdf
        raise NotImplementedError


def get_recomputation_edited_graph(dag, schedule, mode, verbose=True):
    selector = CheckpointsSelector.get_checkpoint_selector(mode)
    checkpoints = selector.select_checkpoints(schedule)
    if not checkpoints:
        SingleLogger().warn("No checkpoints found! Recomputation Aborted!")
        return False

    if verbose:
        names = [node.name for node in checkpoints]
        SingleLogger().info("select %d checkpoints: %s" %
                            (len(names), ', '.join(names)))

    _apply_recomputation(dag, schedule, checkpoints, verbose)

    return True


def _update_schedule(schedule, checkpoints):
    name_to_checkpoints = {node.name: node for node in checkpoints}
    for op in schedule.operators:
        if op.name in name_to_checkpoints:
            op.requires_grad = True
        else:
            op.requires_grad = False


def _apply_recomputation(dag, schedule, checkpoints, verbose):
    _update_schedule(schedule, checkpoints)
    _update_dag(dag, checkpoints, verbose)


def _compose_subgraph_between_two_nodes(dag, source, target):
    if not nx.has_path(dag, source, target):
        # it is possible. e.g. matmul in k, q, v
        return None

    paths_between_two_nodes = nx.shortest_simple_paths(dag, source, target)
    nodes_between_set = {
        node for path in paths_between_two_nodes for node in path}

    subgraph = dag.subgraph(nodes_between_set)

    # add suffix to avoid the same name in a graph
    mapping = {node: node+"_sg" for node in subgraph.nodes}
    return nx.relabel_nodes(subgraph, mapping)


def _insert_forward_nodes(dag: nx.DiGraph, subgraph: nx.DiGraph, target):
    # copy subgraph
    dag.add_nodes_from(subgraph.nodes)
    dag.add_edges_from(subgraph.edges)

    # connect subgraph output to target
    outputs = get_output_nodes(subgraph)
    for output in outputs:
        subgraph.add_edge(output, target)


def _get_last_forward_node(dag):
    forward_nodes = get_forward_nodes(dag.nodes)
    forward_graph = dag.subgraph(forward_nodes).copy()
    leaf_nodes = get_leaf_nodes(forward_graph)
    forward_graph.remove_nodes_from(leaf_nodes)
    sorted_forward_nodes = list(nx.topological_sort(forward_graph))
    sorted_forward_nodes = filter_out_node_by_name(
        sorted_forward_nodes, "read")
    return sorted_forward_nodes[-1]


def _get_target_backward_node(dag, target):
    target_bwp_op_name = target.replace("->FW.", "->BW.gradients/")
    target_bwp_op_name += "_grad/" + target_bwp_op_name.rsplit('/')[-1]
    if target_bwp_op_name in dag.nodes:
        return target_bwp_op_name
    return None


def _update_dag(dag, checkpoints, verbose):
    filtered_nodes = filter_out_comm_nodes(dag.nodes)
    # TODO(yuchen): deal with other ranks
    filtered_nodes = get_rank0_nodes(filtered_nodes)
    names_to_nodes = {get_node_name(node): node for node in filtered_nodes}
    checkpoints_to_nodes = {node.name: names_to_nodes[node.name]
                            for node in checkpoints if node.name in names_to_nodes}

    target = _get_last_forward_node(dag)  # last node in forward
    if verbose:
        SingleLogger().info("Get the last forward node %s." % target)

    for checkpoint in checkpoints[::-1]:
        source = checkpoints_to_nodes[checkpoint.name]
        print("source %s, target %s" % (source, target))
        subgraph = _compose_subgraph_between_two_nodes(dag, source, target)

        if subgraph:
            if verbose:
                SingleLogger().info("ops to be copied: %s" % (', '.join(subgraph.nodes)))

            target_bwp_op = _get_target_backward_node(dag, target)
            if verbose:
                SingleLogger().info("target backward op: %s" % (str(target_bwp_op)))

            # rewire
            _insert_forward_nodes(dag, subgraph, target_bwp_op)

        target = source
