from random import sample
import networkx as nx
import random

from trace_utils import _parse_tf_layer_names, parse_op_name
from cost_model._xla.utils import parse_xla_candidate_ops

class PKGraphCycleError(Exception):
    pass

def get_all_pred_succ_nx(_dag: nx.DiGraph, nodes_to_contract):
    all_nodes = set(nodes_to_contract)
    all_predecessors = set()
    all_successors = set()
    new_node_name = ""
    for node_idx, node in enumerate(nodes_to_contract):
        new_node_name += node
        if node_idx != len(nodes_to_contract) - 1:
            new_node_name += "+"
        for pred in _dag.predecessors(node):
            if pred not in all_nodes:
                all_predecessors.add(pred)
        for succ in _dag.successors(node):
            if succ not in all_nodes:
                all_successors.add(succ)
    return all_predecessors, all_successors, new_node_name

def get_concated_names(components):
    return ["+".join(component) for component in components]

def defuse_nodes_inplace_nx(_dag: nx.DiGraph, pkg, target, components, 
                            succ_override=None, pred_override=None):
    _dag.remove_node(target)
    # build a node -> component reverse index
    component_names = get_concated_names(components)
    node2components = {}
    for idx, component in enumerate(components):
        for _, node in enumerate(component):
            node2components[node] = idx
        _dag.add_node(component_names[idx])

    for comp_idx, component in enumerate(components):
        component_predecessors = set()
        component_succsessors = set()
        overridden_pred = False
        overridden_succ = False
        for node in component:
            if pred_override is not None:
                preds = pred_override(node)
                if preds:
                    overridden_pred = True
                    for pred in preds:
                        component_predecessors.add(pred)
            if not overridden_pred:
                for pred in pkg.nx_graph_reference.predecessors(node):
                    if pred in node2components:
                        pred = component_names[node2components[pred]]
                    elif pred in pkg.nodename2fusednode:
                        pred = pkg.nodename2fusednode[pred]
                    if pred != component_names[comp_idx]:
                        component_predecessors.add(pred)
            if succ_override is not None:
                succs = succ_override(node)
                if succs:
                    overridden_succ = True
                    for succ in succs:
                        component_succsessors.add(succ)
            if not overridden_succ:
                for succ in pkg.nx_graph_reference.successors(node):
                    if succ in node2components:
                        succ = component_names[node2components[succ]]
                    elif succ in pkg.nodename2fusednode:
                        succ = pkg.nodename2fusednode[succ]
                    if succ != component_names[comp_idx]:
                        component_succsessors.add(succ)
        for node in component_predecessors:
            _dag.add_edge(node, component_names[comp_idx])
        for node in component_succsessors:
            _dag.add_edge(component_names[comp_idx], node)
    return component_names

def contract_nodes_nx(graph: nx.DiGraph, nodes_to_contract: list):
    G = graph
    all_nodes = set(nodes_to_contract)
    all_predecessors, all_successors, new_node_name = get_all_pred_succ_nx(graph, nodes_to_contract)
    for node in nodes_to_contract:
        G.remove_node(node)
    G.add_node(new_node_name)
    for pred in all_predecessors:
        G.add_edge(pred, new_node_name)
    for succ in all_successors:
        G.add_edge(new_node_name, succ)
    return new_node_name

def subgraph_partition_connected_nx(subgraph, size_limit=100):
    split_plans = []
    pkg = PKGraph(subgraph)
    source_nodes = [node for node in subgraph.nodes if subgraph.in_degree(node) == 0]
    sp = nx.Graph()
    sp.add_edges_from(nx.dfs_edges(subgraph.to_undirected(), source=random.sample(source_nodes, k=1)[0]))
    if len(sp.edges) > size_limit:
        sampled_edges = random.sample(list(sp.edges), k=size_limit)
    else:
        sampled_edges = list(sp.edges)
    for (u, v) in sampled_edges:
        sp.remove_edge(u, v)
        a = nx.dfs_preorder_nodes(sp, source=u)
        b = nx.dfs_preorder_nodes(sp, source=v)
        split_plans.append((tuple(a), tuple(b)))
        sp.add_edge(u, v)
    valid_split_plans = []
    for splits in split_plans:
        plan_valid = True
        pkg_test = pkg.copy()
        for nodes in splits:
            if len(nodes) > 1:
                if not pkg_test.can_contract_nodes(nodes):
                    plan_valid = False
                    break
                else:
                    pkg_test.contract_nodes_unsafe(nodes)
        if plan_valid:
            valid_split_plans.append(splits)
    return valid_split_plans

def subgraph_partition_connected_nx_using_topo(subgraph, size=100, layer_by_layer=False):
    split_plans = set()
    topo_order_nodes = list(nx.topological_sort(subgraph))
   
    valid_split_point = []
    for idx, node in enumerate(topo_order_nodes):
        if idx == 0:
            continue
        prev_node = topo_order_nodes[idx-1]
        # layer-by-layer fusion is only applied to BW operators
        if layer_by_layer and "BW" in node and "BW" in prev_node:
            _cur_layer = _parse_tf_layer_names(node)
            _prev_layer = _parse_tf_layer_names(prev_node)
            assert len(_cur_layer) == 1
            if _cur_layer[0] != _prev_layer[0]:
                valid_split_point.append(idx)
        else:
            valid_split_point.append(idx)

    size = min(size, len(valid_split_point) - 1)
    for _ in range(size):
        # sp_idx = random.randint(1, len(topo_order_nodes) - 1)
        sp_idx = valid_split_point[random.randint(
            1, len(valid_split_point) - 1)]
        
        part_a = topo_order_nodes[:sp_idx]
        # part_b = topo_order_nodes[sp_idx:], but we do not care about that part
        subgraph_a = subgraph.subgraph(part_a)
        nodes_in_a = max(nx.weakly_connected_components(subgraph_a), key=len)
        nodes_in_b = [node for node in subgraph.nodes if node not in nodes_in_a]
        split_plans.add((tuple(nodes_in_a), tuple(nodes_in_b)))
        
    return list(split_plans)

class PKGraph(object):
    """
    An auxiliary data structure for detecting cycles when modifying DAG.
    The implementation follows the algorithm in the paper:
        A Dynamic Topological Sort Algorithm for Directed Acyclic Graphs,
        DAVID J. PEARCE and PAUL H. J. KELLY
    PKGraph should be used along with the original graph, i.e. it does not 
    directly apply modifications on the original DAG. All methods modify the
    state of PKGraph only.
    Works for networkx DiGraphs.
    """

    def __init__(self, nx_graph=None, nx_graph_reference=None, _init_copy=False):
        if _init_copy:
            return
        if not isinstance(nx_graph, nx.DiGraph):
            raise RuntimeError("nx_graph must be an networkx DiGraph.")
        self.nx_graph = nx_graph.copy(as_view=False)
        if nx_graph_reference is None:
            self.nx_graph_reference = nx_graph.copy(as_view=False)
        else:
            if not isinstance(nx_graph_reference, nx.DiGraph):
                raise RuntimeError("nx_graph_reference must be an networkx DiGraph.")
            self.nx_graph_reference = nx_graph_reference
        index2nodename = list(nx.topological_sort(self.nx_graph))
        self.nodename2idx = dict([(name, idx) for idx, name in enumerate(index2nodename)])
        self.ord = list(range(len(index2nodename)))
        self.free_indexes = set()
        self.nodename2fusednode = {}

    def _get_node_order(self, u):
        idx = self.nodename2idx[u]
        return self.ord[idx]

    def add_edge(self, u, v, **kwargs):
        self.nx_graph.add_edge(u, v, **kwargs)
        if u not in self.nx_graph or v not in self.nx_graph:
            raise RuntimeError("u ({}) and v ({}) must in the original networkx graph.".format(u, v))
        lower_bound = self._get_node_order(v)
        upper_bound = self._get_node_order(u)
        if lower_bound < upper_bound:
            # discovery
            delta_f = []
            exists_cycle = not self._dfs_forward(v, upper_bound, delta_f)
            if exists_cycle:
                self.nx_graph.remove_edge(u, v)
                # print(delta_f)
                # print(list(nx.find_cycle(self.nx_graph, orientation="original")))
                from dag_utils import part_of_dag
                node = u
                focus_nodes = [v]
                small_dag = part_of_dag(self.nx_graph, node,
                    max_in_depth=3, max_out_depth=3,
                    path = "/home/tiger/small_dag.txt",
                    simple=True,
                    focus_nodes=focus_nodes,
                    use_std_name=False)
                raise PKGraphCycleError("Cannot add edge ({}, {}) because it will introduce a cycle.".format(u, v))
            delta_b = []
            self._dfs_backward(u, lower_bound, delta_b)
            # reassignment
            self._reorder(delta_f, delta_b)

    def can_contract_edge(self, u, v):
        ''' Check whether we can contract the edge from cluster u to cluster v
                * > // Check if contracting this edge will break the resource variable concurrency
                    // semantics.  In theory this is quadratic in the number of nodes, but seems
                    // to not be a problem in practice so far.   --- from TensorFlow
                * if the graph has cycles after contracting u and v
        '''
        ### check cycles
        if u not in self.nx_graph or v not in self.nx_graph:
            raise RuntimeError("u ({}) and v ({}) must in the original networkx graph.".format(u, v))
        if (u, v) not in self.nx_graph.edges():
            return True
        self.nx_graph.remove_edge(u, v)
        is_reachable = self._is_reachable(u, v)
        self.nx_graph.add_edge(u, v)
        return not is_reachable
    
    def can_contract_nodes(self, nodes_to_contract: list):
        try:
            pkg_copy = self.copy()
            pkg_copy.contract_nodes_unsafe(nodes_to_contract)
            return True
        except PKGraphCycleError as e:
            # cannot conract
            return False
    
    def contract_nodes_unsafe(self, nodes_to_contract: list):
        # NOTE: this function assumes after contraction there will be no cycle. 
        # If a cycle is found, an error will be thrown and the inernal graph will break.
        all_predecessors, all_successors, new_node_name = get_all_pred_succ_nx(self.nx_graph, nodes_to_contract)

        # update indexes
        self.nodename2idx[new_node_name] = self.nodename2idx[nodes_to_contract[-1]]
        self.nodename2idx.pop(nodes_to_contract[-1])
        for node in nodes_to_contract[:-1]:
            self.free_indexes.add(self.nodename2idx.pop(node))

        for node in nodes_to_contract:
            self.nx_graph.remove_node(node)
        self.nx_graph.add_node(new_node_name)
        for pred in all_predecessors:
            self.add_edge(pred, new_node_name)
        for succ in all_successors:
            self.add_edge(new_node_name, succ)
        # update mapping
        for node in nodes_to_contract:
            self.nodename2fusednode[node] = new_node_name
        return new_node_name
    
    def contract_edge(self, u, v):
        if (u, v) in self.nx_graph.edges():
            self.nx_graph.remove_edge(u, v)
        # if self._is_reachable(u, v):
        #     self.nx_graph.add_edge(u, v)
        #     return False

        predecessors = set()
        for node in self.nx_graph.predecessors(u):
            predecessors.add(node)
        for node in self.nx_graph.predecessors(v):
            predecessors.add(node)
        successors = set()
        for node in self.nx_graph.successors(u):
            successors.add(node)
        for node in self.nx_graph.successors(v):
            successors.add(node)
        new_node_name = u + "+" + v

        self.nodename2idx[new_node_name] = self.nodename2idx[v]
        self.free_indexes.add(self.nodename2idx.pop(u))
        self.nodename2idx.pop(v)
        
        self.nx_graph.remove_node(u)
        self.nx_graph.remove_node(v)

        self.nx_graph.add_node(new_node_name)
        for node in predecessors:
            self.add_edge(node, new_node_name)
        for node in successors:
            self.add_edge(new_node_name, node)
        
        ns = u.split("+") + v.split("+")
        for node in ns:
            self.nodename2fusednode[node] = new_node_name
        return True

    def split_node(self, u, components):
        # u: the name of the composite node to split,
        # components: a list of list containing the names of each 
        # connected components in the splitted graph
        # Note: we assume the split is valid and only throw errors 
        # if add edge fails
        self.nx_graph.remove_node(u)
        self.free_indexes.add(self.nodename2idx.pop(u))

        # build a node -> component reverse index
        component_names = []
        node2components = {}
        for idx, component in enumerate(components):
            component_name = ""
            for node_idx, node in enumerate(component):
                node2components[node] = idx
                component_name += node
                if node_idx != len(component) - 1:
                    component_name += "+"
            component_names.append(component_name)
            self.nx_graph.add_node(component_name)

            # ! important: assign a free index to the component
            free_idx = self.free_indexes.pop()
            self.nodename2idx[component_name] = free_idx

        # print("Component names: {}".format(component_names))

        for comp_idx, component in enumerate(components):
            component_predecessors = set()
            component_succsessors = set()
            for node in component:
                for pred in self.nx_graph_reference.predecessors(node):
                    if pred in node2components:
                        pred = component_names[node2components[pred]]
                    elif pred in self.nodename2fusednode:
                        pred = self.nodename2fusednode[pred]
                    if pred != component_names[comp_idx]:
                        component_predecessors.add(pred)
                for succ in self.nx_graph_reference.successors(node):
                    if succ in node2components:
                        succ = component_names[node2components[succ]]
                    elif succ in self.nodename2fusednode:
                        succ = self.nodename2fusednode[succ]
                    if succ != component_names[comp_idx]:
                        component_succsessors.add(succ)
            for node in component_predecessors:
                self.add_edge(node, component_names[comp_idx])
            for node in component_succsessors:
                self.add_edge(component_names[comp_idx], node)
        
        for comp_idx, component in enumerate(components):
            if len(component) == 1:
                node = component[0]
                self.nodename2fusednode.pop(node)
            else:
                for node in component:
                    self.nodename2fusednode[node] = component_names[comp_idx]

    def copy(self):
        new_instance = self.__class__(_init_copy=True)
        new_instance.nx_graph = self.nx_graph.copy()
        new_instance.nx_graph_reference = self.nx_graph_reference.copy()
        new_instance.nodename2idx = self.nodename2idx.copy()
        new_instance.ord = self.ord.copy()
        new_instance.free_indexes = self.free_indexes.copy()
        new_instance.nodename2fusednode = self.nodename2fusednode.copy()
        return new_instance

    def check_invariant(self):
        assert len(self.free_indexes) + len(self.nodename2idx) == len(self.nx_graph_reference.nodes)
    
    def check_identical(self, g):
        assert g.edges() == self.nx_graph.edges()

    def _is_reachable(self, u, v):
        if u == v:
            return True
        ord_v = self._get_node_order(v)
        if self._get_node_order(u) > ord_v:
            return False
        forward_nodes = []
        if not self._dfs_forward(u, ord_v, forward_nodes):
            return True
        else:
            return False

    def _dfs_forward(self, u, upper_bound, forward_nodes, visited=None):
        if visited is None:
            visited = set()
        visited.add(u)
        forward_nodes.append(u)
        for v in self.nx_graph.successors(u):
            ord_v = self._get_node_order(v)
            if ord_v == upper_bound:
                return False
            if v not in visited and ord_v < upper_bound:
                exist_cycles = not self._dfs_forward(v, upper_bound, forward_nodes=forward_nodes, visited=visited)
                if exist_cycles:
                    return False
        return True
    
    def _dfs_backward(self, v, lower_bound, backward_nodes, visited=None):
        if visited is None:
            visited = set()
        visited.add(v)
        backward_nodes.append(v)
        for u in self.nx_graph.predecessors(v):
            if u not in visited and self._get_node_order(u) > lower_bound:
                self._dfs_backward(u, lower_bound, backward_nodes=backward_nodes, visited=visited)
    
    def _reorder(self, delta_f, delta_b):
        delta_f_idx = sorted([self.nodename2idx[node] for node in delta_f], key=lambda x: self.ord[x])
        delta_f_ord = [self.ord[idx] for idx in delta_f_idx]
        delta_b_idx = sorted([self.nodename2idx[node] for node in delta_b], key=lambda x: self.ord[x])
        delta_b_ord = [self.ord[idx] for idx in delta_b_idx]
        L = []
        L += delta_b_idx
        L += delta_f_idx
        all_orders = sorted(delta_f_ord + delta_b_ord)
        for i in range(len(L)):
            self.ord[L[i]] = all_orders[i]

def postorder_contract_nx(_dag: nx.DiGraph, _pkg: PKGraph, source_node, visitied_nodes, 
                          forbidden_list=None, size_limit=None):
    # print("postorder_contract_nx for {}".format(source_node))
    graph_changed_outer = False
    while True:
        should_break = True
        for node in _dag.successors(source_node):
            if forbidden_list is not None and node in forbidden_list:
                continue
            if node in _dag.successors(source_node) and node not in visitied_nodes:
                visitied_nodes.add(node)
                new_node_name, graph_changed, _dag = postorder_contract_nx(
                    _dag, _pkg, node, visitied_nodes,
                    forbidden_list = forbidden_list,
                    size_limit = size_limit
                )
                if graph_changed:
                    should_break = False
                    graph_changed_outer = True
                    visitied_nodes.add(new_node_name)
                    break
        if should_break:
            break
    self_size = len(source_node.split("+"))
    for succ in list(_dag.successors(source_node)):
        if forbidden_list is not None:
            if source_node in forbidden_list or succ in forbidden_list:
                continue
        
        # ### fuse BW layer by layer
        # ### only operators in the same `layer_num_limit` layer(s) can be contracted
        # if layer_num_limit is not None and "BW" in source_node and "BW" in succ:
        #     u_layer_names = _parse_tf_layer_names(source_node)
        #     v_layer_names = _parse_tf_layer_names(succ)
        #     fused_layer_names = set(u_layer_names).union(v_layer_names)
        #     # print(set(u_layer_names), set(v_layer_names))
        #     if len(fused_layer_names) > layer_num_limit:
        #         continue

        if _pkg.can_contract_edge(source_node, succ):
            succ_size = len(succ.split("+"))
            if size_limit and self_size + succ_size > size_limit:
                continue
            _pkg.contract_edge(source_node, succ)
            new_node_name = contract_nodes_nx(_dag, [source_node, succ])
            source_node = new_node_name
            self_size = self_size + succ_size
            graph_changed_outer = True
    return source_node, graph_changed_outer, _dag

def contract_groups(_dag: nx.DiGraph, _pkg: PKGraph, forbidden_list=None, list_of_group=None):
    for nodes_to_contract in list_of_group:
        # if forbidden_list is not None:
        #     nodes_to_contract = [n for n in nodes_to_contract if n not in forbidden_list]
        
        assert len(nodes_to_contract) > 1
        assert len([n for n in nodes_to_contract if "FW" in n or "Comm" in n]) == 0
        # print("DEBUG: {}".format(nodes_to_contract))

        _pkg.contract_nodes_unsafe(nodes_to_contract)
        new_node_name = contract_nodes_nx(_dag, nodes_to_contract)
    return _dag