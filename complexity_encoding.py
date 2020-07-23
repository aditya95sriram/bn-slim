#!/bin/python3.6

import sys
from samer_veith import SvEncoding
from typing import Dict, List, Any
import subprocess
from time import time as now
from utils import read_model, TreeDecomposition, compute_complexity_width, \
    weights_from_domain_sizes, replicate
from collections import defaultdict, OrderedDict
import networkx as nx

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


DRAWING = False


def make_decision_diagram(bound: int, weights: Dict[Any, int]):
    # vertices of the form (node, level), 0 <= level <= bound
    dd = nx.DiGraph(bound=bound, weights=weights)
    YES, NO = "YES", "NO"
    dd.add_nodes_from([YES, NO])  # sinks of decision diagram
    nodes = list(sorted(weights.keys(), key=weights.get, reverse=True))
    first_node, last_node = nodes[0], nodes[-1]
    next_node = dict(zip(nodes, nodes[1:]))
    dd.graph['root'] = root = (first_node, 0)
    queue = [root]
    while queue:
        cur_node, cur_level = cur_vertex = queue.pop(0)
        if cur_node is last_node:
            # case if cur_node
            if weights[cur_node] + cur_level <= bound:
                dd.add_edge(cur_vertex, YES, case="if")
            else:
                dd.add_edge(cur_vertex, NO, case="if")
            # case if not cur_node
            dd.add_edge(cur_vertex, YES, case="else")
        else:
            child = next_node[cur_node]
            # case if cur_node
            new_level = cur_level + weights[cur_node]
            if new_level > bound:
                dd.add_edge(cur_vertex, NO, case="if")
            else:
                dd.add_edge( cur_vertex, (child, new_level), case="if")
                queue.append((child, new_level))
            # case if not cur_node
            dd.add_edge( cur_vertex, (child, cur_level), case="else")
            queue.append((child, cur_level))
    return dd


class SvEncodingWithComplexity(SvEncoding):

    use_dd = False

    def __init__(self, stream, graph, weights: Dict[Any, int] = None, debug=False):
        super().__init__(stream, graph)
        if not hasattr(self, 'debug'): self.debug = debug
        self.weights: Dict[int, int] = weights

        # for debugging purposes
        if self.debug:
            self.cardinality_counter_vars = []
            self.current_cardinality_outer_var = None
            self.current_cardinality_inner_vars = []

    def set_weights(self, weights: Dict[Any, int]):
        self.weights = weights

    def _add_comment(self, comment):
        if self.debug: self.stream.write(f"c {comment}\n")

    def encode_single_cardinality(self, bound, variables: OrderedDict):
        """Enforces cardinality constraint on list of variables (repeats allowed)"""
        if bound <= 0:
            if self.debug:
                print(f"warning: non-positive bound {bound} provided, forcing UNSAT")
            self._add_clause(-1)
            self._add_clause(1)
            return

        variables = replicate(variables)
        jlim = len(variables)
        ctr = [[self._add_var() for _ in range(0, min(j+1, bound))] for j in range(jlim)]

        for j in range(1, jlim):
            # never decrements
            for l in range(0, len(ctr[j-1])):
                self._add_clause(-ctr[j-1][l], ctr[j][l])

            # increment if variable and ctr
            for l in range(1, len(ctr[j])):
                self._add_clause(-variables[j], -ctr[j-1][l-1], ctr[j][l])

        # initialize first counter if corr variable is true
        for j in range(jlim):
            self._add_clause(-variables[j], ctr[j][0])

        # conflict if target exceeded
        for j in range(bound, jlim):
            self._add_clause(-variables[j], -ctr[j-1][bound-1])

        if self.debug:
            for j in range(jlim):
                outer_var = self.current_cardinality_outer_var
                inner_var = self.current_cardinality_inner_vars[j]
                self.cardinality_counter_vars.append((outer_var, inner_var, ctr[j]))

    def encode_cardinality_sat(self, bound, variables: Dict[int, Dict[int, int]]):
        """
        Enforce weighted cardinality constraint on 2-d structure of variables
        * weights are read from self.weights
        * bound is adjusted by weight of the outer variable
        * inner variable is replicated as many times as its weight to form final
          list of variables to be cardinally constrained
        """
        old = self.num_clauses
        for i in range(len(variables)):
            node = self.node_reverse_lookup[i]
            varcounts = OrderedDict()
            if self.debug:
                self.current_cardinality_outer_var = node
                self.current_cardinality_inner_vars = []
            for j in range(len(variables[i])):
                var = variables[i][j]
                other = self.node_reverse_lookup[j]
                if other == node: continue
                varcounts[var] = self.weights[other]
                # vararray.extend([var]*self.weights[other])
                if self.debug: self.current_cardinality_inner_vars.extend([other]*self.weights[other])
            if self.use_dd:
                self.encode_single_cardinality_with_dd(bound - self.weights[node], varcounts)
            else:
                self.encode_single_cardinality(bound - self.weights[node], varcounts)
        print("#cardinality clauses:", self.num_clauses - old)

    def debug_counters(self, model, elim_order):
        if not self.debug: return
        arc_counts = defaultdict(int)
        prevu = self.cardinality_counter_vars[0][0]
        for u, v, ctrs in self.cardinality_counter_vars:
            if u != prevu:
                print(f"{prevu} total arcs: {arc_counts[prevu]}\n")
                prevu = u
            appears_before = (elim_order.index(u) < elim_order.index(v))
            arc_exists = appears_before and self.arc[self.node_lookup[u]][self.node_lookup[v]] in model
            arc_counts[u] += arc_exists
            print(f"{u}->{v}" + ("*" if arc_exists else " "), end="\t:\t")
            print(*(int(var in model) for var in ctrs))
        print(f"{prevu} total arcs: {arc_counts[prevu]}\n")
        print(dict(arc_counts))

    def encode_single_cardinality_with_dd(self, bound, variables: OrderedDict):
        """
        Enforces weighted cardinality constraint on list of variables
        (repeats allowed)
        """
        if bound <= 0:
            if self.debug:
                print(f"warning: non-positive bound {bound} provided, forcing UNSAT")
            self._add_clause(-1)
            self._add_clause(1)
            return

        dd = make_decision_diagram(bound, variables)
        dd_lookup = {vertex: self._add_var() for vertex in dd.nodes}
        dd_reverse_lookup = {idx: vertex for (vertex, idx) in dd_lookup.items()}

        self._add_clause(-dd_lookup['NO'])  # if NO is ever true, this results in contradiction
        self._add_clause(dd_lookup[dd.graph['root']])  # initialize root node as true
        for u, v, case in dd.edges(data="case"):
            if v == 'YES':
                pass  # nothing to do
            else:
                varid, level = u
                if case == "if":
                    # clause: u & var => v
                    self._add_comment(f"({u} & arc_{varid}) -> {v}")
                    self._add_clause(-dd_lookup[u], -varid, dd_lookup[v])
                else:
                    # clause: u & !var => v
                    self._add_comment(f"({u} & !arc_{varid}) -> {v}")
                    self._add_clause(-dd_lookup[u], varid, dd_lookup[v])

        if self.debug:
            self.cardinality_counter_vars.append((self.current_cardinality_outer_var, dd, dd_lookup, dd_reverse_lookup))

    def debug_dd_counter(self, model, elim_order):
        if not self.debug: return
        arc_counts = dict()
        for u, dd, dd_lookup, dd_reverse_lookup in self.cardinality_counter_vars:
            arcs = set()
            remap = dict()
            for vpair in dd.nodes:
                if vpair == 'YES' or vpair == 'NO': continue
                arcid, level = vpair
                v = None
                for a, aid in self.arc[u].items():
                    if aid == arcid:
                        v = a
                remap[arcid] = v
                appears_before = (elim_order.index(u) < elim_order.index(v))
                arc_exists = appears_before and self.arc[self.node_lookup[u]][self.node_lookup[v]] in model
                if arc_exists:
                    arcs.add((u,v))
            print(arcs)
            draw_dd(dd, remap=remap)
            arc_counts[u] = len(arcs)
            print(f"{u} total arcs: {len(arcs)}\n")
        print(dict(arc_counts))


def draw_dd(dd, true_arcs=None, remap=None):
    bound = dd.graph['bound']
    weights = dd.graph['weights']
    nodes = list(sorted(weights.keys(), key=weights.get, reverse=True))
    nodeidx = {node: i for i, node in enumerate(nodes)}
    # pos = graphviz_layout(dd, prog='dot')
    pos = dict()
    for pair in dd.nodes:
        if isinstance(pair, tuple):
            node, level = pair
            pos[pair] = nodeidx[node], -level
    pos["YES"] = (len(nodes)-1, -(bound + 1))
    pos["NO"] = (0, -(bound + 1))
    # pos = {(nodeidx[node], level) for node, level in dd.nodes}
    nx.draw_networkx_nodes(dd, pos)
    if remap is not None:
        node_labels = dict()
        for pair in dd.nodes:
            if isinstance(pair, tuple):
                v, level = pair
                node_labels[pair] = remap[v], level
            else:
                node_labels[pair] = pair
        nx.draw_networkx_labels(dd, pos, node_labels)
    else:
        nx.draw_networkx_labels(dd, pos)
    if_edges, else_edges = [], []
    for u,v, case in dd.edges(data="case"):
        if case == "if":
            if_edges.append((u, v))
        else:
            else_edges.append((u, v))
    nx.draw_networkx_edges(dd, pos, if_edges, edge_color='g',
                           connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_edges(dd, pos, else_edges, edge_color='r',
                           connectionstyle='arc3,rad=-0.2')
    if true_arcs is not None:
        nx.draw_networkx_edge_labels(dd, pos, {e: '*' for e in true_arcs})
    plt.show()


def solve_graph(graph, weights, complexity_width, timeout: int = 10, debug=False):
    cnfpath = "temp-cw.cnf"
    with open(cnfpath, 'w') as cnffile:
        enc = SvEncodingWithComplexity(cnffile, graph, weights, debug)
        enc.encode_sat(complexity_width)
    print(f"enc: {enc.__class__.__name__}, ({enc.num_clauses} clauses)")
    if debug: print("encoding done")
    base_cmd = ["glucose", "-verb=0", "-model"]
    cmd = base_cmd + [cnfpath, f"-cpu-lim={timeout}"]
    start = now()
    proc = subprocess.run(cmd, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime = now() - start
    output = proc.stdout
    model = read_model(output)
    dec = TwbnDecoder(enc, -1, model, "")
    elim_order = dec.get_elim_order()
    # if USE_DD:
    #     enc.debug_dd_counter(model, elim_order)
    # else:
    #     enc.debug_counters(model, elim_order)
    tri = dec.get_triangulated().to_undirected()
    td = TreeDecomposition(tri, elim_order, width=-1)
    if DRAWING:
        pos = graphviz_layout(dec.get_triangulated(), prog='dot')
        nx.draw(dec.get_triangulated(), pos, with_labels=True)
        plt.show()
        td.draw()
    return td


if __name__ == '__main__':
    # weights = {'a':3, 'b':2, 'c':1, 'd': 1}
    # bound = 4
    # dd = make_decision_diagram(bound, weights)
    # draw_dd(dd)
    import random
    from berg_encoding import TwbnDecoder
    SEED = 3
    if len(sys.argv) >= 2:
        SEED = int(sys.argv[1])
    print("seed:", SEED)
    random.seed(SEED)
    if len(sys.argv) >= 3:
        use_dd = bool(int(sys.argv[2]))
    else:
        use_dd = input("use dd? y/[n]:") == "y"
    SvEncodingWithComplexity.use_dd = use_dd

    # g = nx.Graph()
    # g.add_edges_from("ac af bc bh cd eg eh fg gh".split())
    g = nx.fast_gnp_random_graph(20, p=0.3, seed=SEED)
    ds = {node: random.randint(2,16) for node in g.nodes}
    print("ds:", ds)
    weights = weights_from_domain_sizes(ds)
    # g.remove_edge(0, 3)
    # g.add_edge(0, 4)
    # weights[4] = 2

    # weights = {node: 1 for node in g.nodes}
    # weights['h'] = 2
    # weights['a'] = 3

    print("weights:", weights)
    nx.draw(g, with_labels=True)
    if DRAWING: plt.show()
    td = solve_graph(g, weights, complexity_width=15, timeout=30, debug=False)
    td.verify()
    print(td.elim_order)
    cw = compute_complexity_width(td, ds)
    print("final cw:", cw)
    # td.draw()