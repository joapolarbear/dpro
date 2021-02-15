from functools import partial
import networkx as nx

DEL = "->"
RANK0_PREFIX = "host0.rank0"
FORWARD_CAT = "FW."
BACKWARD_CAT = "BW."


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_nodes_prefix(nodes, prefix):
    func = partial(remove_prefix, prefix=prefix)
    return list(map(func, nodes))


def remove_node_prefix(node, prefix):
    func = partial(remove_prefix, prefix=prefix)
    return func(node)


def filter_out_comm_nodes(nodes):
    def _is_comm(name):
        if name.startswith("server") or name.startswith("worker"):
            return False
        return True

    return list(filter(_is_comm, nodes))


def get_node_name(name):
    return name.rsplit(".")[-1]


def get_rank0_nodes(nodes):
    # TODO(yuchen): Not expandable
    def _is_rank0(name):
        if name.startswith(RANK0_PREFIX):
            return True
        return False

    return list(filter(_is_rank0, nodes))


def get_leaf_nodes(dag):
    return [node for node in dag.nodes if dag.out_degree(node) == 1
            and dag.in_degree(node) == 0]


def filter_out_node_by_name(nodes, name):
    return list(filter(lambda node: False if name in node else True, nodes))


def get_forward_nodes(nodes):
    # TODO(yuchen): Not expandable
    def _is_forward(name):
        if name.startswith(DEL.join([RANK0_PREFIX, FORWARD_CAT]) + "bert"):
            return True
        return False

    return list(filter(_is_forward, nodes))


def get_forward_backward_nodes(nodes):
    # TODO(yuchen): Not expandable
    def _is_forward(name):
        if name.startswith(DEL.join([RANK0_PREFIX, FORWARD_CAT]) + "bert") or \
                name.startswith(DEL.join([RANK0_PREFIX, BACKWARD_CAT]) + "bert"):
            return True
        return False

    return list(filter(_is_forward, nodes))


def get_input_nodes(dag):
    return [u for u, deg in dag.in_degree() if not deg]


def get_output_nodes(dag):
    return [u for u, deg in dag.out_degree() if not deg]


def insert_nodes(dag, subgraph, target):
    if not target:
        return
    # copy subgraph
    dag.add_nodes_from(subgraph.nodes.data())
    dag.add_edges_from(subgraph.edges.data())

    # remove previous nodes
    prev_nodes = list(dag.predecessors(target))
    for prev_node in prev_nodes:
        dag.remove_edge(prev_node, target)

    # connect subgraph output to target
    outputs = get_output_nodes(subgraph)
    dag.add_edge(outputs[0], target)


def update_time_by_scale(dag, scale):
    trace_times = nx.get_node_attributes(dag, "avg")
    for k, v in trace_times.items():
        trace_times[k] = v * scale
    nx.set_node_attributes(dag, trace_times, "avg")
