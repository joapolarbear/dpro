from .utils import *


def get_gradient_accumulation_edited_graph(dag, verbose=False):
    _apply_gradient_accumulation(dag, verbose)
    return True


def _apply_gradient_accumulation(dag, verbose):
    _update_dag(dag, verbose)


def _update_dag(dag, verbose):
    computation_nodes = filter_out_comm_nodes(dag)
    update_time_by_scale(dag.subgraph(computation_nodes), 0.8)

    # TODO(yuchen): deal with other ranks
    filtered_nodes = get_forward_backward_nodes(dag.nodes)
    subgraph = dag.subgraph(filtered_nodes)

    target = filtered_nodes[0]  # first node

    mapping = {node: node+"_ga" for node in subgraph.nodes}
    subgraph = nx.relabel_nodes(subgraph, mapping)

    insert_nodes(dag, subgraph, target)
