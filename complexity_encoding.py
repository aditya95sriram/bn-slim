#!/bin/python3.6

import os
import sys
from samer_veith import SvEncoding
from berg_encoding import TwbnEncoding, TwbnDecoder, SOLVER_DIR, TIMEOUT, TOP
from typing import Dict, Any, List
import subprocess
from time import time as now
import shutil
from utils import NoSolutionException, read_model, TreeDecomposition, compute_complexity_width
from math import ceil, log2
from collections import defaultdict

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class SvEncodingWithComplexity(SvEncoding):

    _add_comment = TwbnEncoding._add_comment

    def __init__(self, stream, graph, weights: Dict[int, int], debug=False):
        super().__init__(stream, graph)
        self.debug = debug
        self.weights: Dict[int, int] = weights

        # for debugging purposes
        if self.debug:
            self.cardinality_counter_vars = []
            self.current_cardinality_outer_var = None
            self.current_cardinality_inner_vars = []

    def encode_single_cardinality(self, bound, variables: List[int]):
        """Enforces weighted cardinality constraint on 1-d structure of variables"""
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
        for i in range(len(variables)):
            node = self.node_reverse_lookup[i]
            vararray = []
            if self.debug:
                self.current_cardinality_outer_var = node
                self.current_cardinality_inner_vars = []
            for j in range(len(variables[i])):
                var = variables[i][j]
                other = self.node_reverse_lookup[j]
                if other == node: continue
                vararray.extend([var]*self.weights[other])
                if self.debug: self.current_cardinality_inner_vars.extend([other]*self.weights[other])
            self.encode_single_cardinality(bound - self.weights[node], vararray)

    def show_counters(self, model, elim_order):
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


def solve_graph(graph, weights, complexity_width, timeout: int = TIMEOUT,
                debug=False):
    cnfpath = "temp-cw.cnf"
    # logpath = "temp-cw.log"
    with open(cnfpath, 'w') as cnffile:
        enc = SvEncodingWithComplexity(cnffile, graph, weights, debug)
        # enc = SvEncoding(cnffile, graph)
        enc.encode_sat(complexity_width)
    print("enc:", enc.__class__.__name__)
    if debug: print("encoding done")
    # base_cmd = [os.path.join(SOLVER_DIR, "uwrmaxsat"), "-m", "-v0"]
    # cmd = base_cmd + [cnfpath, f"-cpu-lim={timeout}"]
    base_cmd = ["glucose", "-verb=0", "-model"]
    cmd = base_cmd + [cnfpath, f"-cpu-lim={timeout}"]
    start = now()
    try:
        proc = subprocess.run(cmd, universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        # copy over problematic cnf file
        errcnf = "error.cnf"
        shutil.copy(cnfpath, errcnf)
        if err.stdout is not None:
            errfilename = "uwrmaxsat-err.log"
            with open(errfilename, 'w') as errfile:
                errfile.write(err.stdout)
                #raise RuntimeError(f"error while running uwrmaxsat on {errcnf}"
                #                   f"\nrc: {err.returncode}, check {errfilename}")
                print(f"error while running uwrmaxsat on {errcnf}"
                      f"\nrc: {err.returncode}, check {errfilename}")
                raise NoSolutionException("nonzero returncode")
        else:
            #raise RuntimeError(f"error while running uwrmaxsat on {errcnf}"
            #                   f"\nrc: {err.returncode}, no stdout captured")
            print(f"error while running uwrmaxsat on {errcnf}"
                  f"\nrc: {err.returncode}, no stdout captured")
            raise NoSolutionException("nonzero returncode")
    else:  # if no error while maxsat solving
        runtime = now() - start
        output = proc.stdout
        model = read_model(output)
        dec = TwbnDecoder(enc, -1, model, "")
        elim_order = dec.get_elim_order()
        # enc.show_counters(model, elim_order)
        tri = dec.get_triangulated().to_undirected()
        td = TreeDecomposition(tri, elim_order, width=-1)
        pos = graphviz_layout(dec.get_triangulated(), prog='dot')
        nx.draw(dec.get_triangulated(), pos, with_labels=True)
        plt.show()
        td.draw()
        return td


COMPLEXITY_WIDTH = 10


def get_weights(domain_sizes):
    return {node: int(ceil(log2(size))) for node, size in domain_sizes.items()}


if __name__ == '__main__':
    import networkx as nx
    import random
    SEED = 3
    if len(sys.argv) >= 2:
        SEED = int(sys.argv[1])
    random.seed(SEED)

    # g = nx.Graph()
    # g.add_edges_from("ac af bc bh cd eg eh fg gh".split())
    g = nx.fast_gnp_random_graph(5, p=0.5, seed=SEED)
    ds = {node: random.randint(2,16) for node in g.nodes}
    print("ds:", ds)
    weights = get_weights(ds)

    # g.remove_edge(0, 3)
    # g.add_edge(0, 4)
    # weights[4] = 2

    # weights = {node: 1 for node in g.nodes}
    # weights['h'] = 2
    # weights['a'] = 3

    print("weights:", weights)
    nx.draw(g, with_labels=True)
    plt.show()
    td = solve_graph(g, weights, complexity_width=10, timeout=30, debug=True)
    print(td.elim_order)
    cw = compute_complexity_width(td, ds)
    print("final cw:", cw)
    # td.draw()
