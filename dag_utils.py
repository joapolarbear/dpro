import networkx as nx
import matplotlib.pyplot as plt
from trace_utils import read_traces, return_stat

QueueType = [
  "COORDINATE_REDUCE",
  "REDUCE",
  "COPYD2H",
  "PCIE_REDUCE",
  "COORDINATE_PUSH",
  "PUSH",
  "PULL",
  "COPYH2D",
  "COORDINATE_BROADCAST",
  "BROADCAST",
  "QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST"
]

def visualize_gml(graph, layout="circular"):
  if layout == "spectral":
    pos = nx.spectral_layout(graph, dim=2, scale=0.5)
  elif layout == "circular":
    pos = nx.circular_layout(graph)
  elif layout == "random":
    pos = nx.random_layout(graph)
  nx.draw(graph, pos, with_labels=True, font_size=6)
  plt.show()
  # import matplotlib.pyplot as plt; plt.ion()
  # import netgraph
  # netgraph.draw(graph)
  # plot_instance = netgraph.InteractiveGraph(graph, node_positions=pos)
  # node_positions = plot_instance.node_positions

def dag_longest_path(G, local_rank, logger, weight='weight', default_weight=0):
  critical_path = nx.algorithms.dag.dag_longest_path(G, weight=weight, default_weight=default_weight)
  prefix = "Critical Path of " + ("the Entire Graph: " if local_rank == -1 else "GPU-%d: " % local_rank)
  logger.info(prefix + " => ")
  path_length = 0
  for (u, v) in nx.utils.pairwise(critical_path):
    path_length += G[u][v].get(weight, default_weight)
    logger.info("%s -> %s: %f ms" % (u, v, G[u][v].get(weight, default_weight)))
  # logger.info(prefix + str(critical_path) + " => " + prefix + "%12.4f ms" % path_length)
  logger.info("Length of the " + prefix + "%12.4f ms\n" % path_length)


def gen_dag_from_gml_and_traces(name2sta, gml_path, rank, del_queue, logger):
  '''
  Return: A dag, containing FW, BW, OUTPUT, Comm, I/O and Sync nodes
    node names start with 'rank{id}.'
  '''
  mygraph = nx.read_gml(gml_path)
  dag = nx.DiGraph()
  def add_prefix(name):
    return "rank%d."%rank + name
  def _read_stat(node_name, _assert=False):
    return name2sta[node_name]["avg"] if node_name in name2sta else 0.0

  for u, v in mygraph.edges:
    if "Comm" in u:
      if del_queue == True:
        prev_nodes = [_u for _u, _ in mygraph.in_edges(u)]
        assert len(prev_nodes) == 1
        prev_node = prev_nodes[0]
        for suffix in QueueType[-1:]:
          cur_node = u + '.' + suffix
          if _read_stat(cur_node) == 0:
            continue
          dag.add_edge(add_prefix(prev_node), add_prefix(cur_node), weight=_read_stat(prev_node))
          prev_node = cur_node
        dag.add_edge(add_prefix(prev_node), "Sync", weight=_read_stat(prev_node))
      else:
        dag.add_edge(add_prefix(u), "Sync", weight=_read_stat(u))
    else:
      dag.add_edge(add_prefix(u), add_prefix(v), weight= _read_stat(u)) 
  for e in dag.edges.data("weight"):
    logger.debug(e)
  # visualize_gml(dag, layout="circular")
  return dag