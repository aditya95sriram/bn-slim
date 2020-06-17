#!/bin/python3.6

import os
import sys
from samer_veith import SvEncoding
from berg_encoding import TwbnEncoding, TwbnDecoder, SOLVER_DIR, TIMEOUT, TOP
from typing import Dict
import subprocess
from time import time as now
import shutil
from utils import NoSolutionException, read_model, TreeDecomposition, compute_complexity_width
from math import ceil, log2

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class SvEncodingWithComplexity(SvEncoding):

    _add_comment = TwbnEncoding._add_comment

    def __init__(self, stream, graph, shadow_counts: Dict[int, int], debug=False):
        super().__init__(stream, graph)
        self.debug = debug
        self.shadow_counts: Dict[int, int] = shadow_counts
        self.shadow_lookup = {i: dict() for i in range(self.num_nodes)}
        self.shadow_reverse_lookup = dict()
        ctr = self.num_nodes
        for _u, counts in self.shadow_counts.items():
            u = self.node_lookup[_u]
            for i in range(counts):
                self.shadow_lookup[u][i] = ctr
                self.shadow_reverse_lookup[ctr] = (_u, i)
                ctr += 1

        # for debugging purposes
        self.cardinality_counter_vars = []

    def encode_single_cardinality(self, bound, variables):
        """Enforces cardinality constraint on 1-d structure of variables"""
        jvals = list(sorted(variables.keys()))
        jlim = len(jvals)
        ctr = [[self._add_var() for _ in range(0, min(j+1, bound))] for j in range(jlim)]

        for j in range(1, jlim):
            # never decrements
            for l in range(0, len(ctr[j-1])):
                self._add_clause(-ctr[j-1][l], ctr[j][l])

            # increment if variable and ctr
            for l in range(1, len(ctr[j])):
                self._add_clause(-variables[jvals[j]], -ctr[j-1][l-1], ctr[j][l])

        # initialize first counter if corr variable is true
        for j in range(jlim):
            self._add_clause(-variables[jvals[j]], ctr[j][0])

        # conflict if target exceeded
        for j in range(bound, jlim):
            self._add_clause(-variables[jvals[j]], -ctr[j-1][bound-1])

        self.cardinality_counter_vars.append(ctr)

    def encode_cardinality_sat(self, bound, variables):
        for i in range(len(variables)):
            self.encode_single_cardinality(bound, variables[i])

    def encode_sat(self, target, cardinality=True):
        # encode everything but cardinality
        super().encode_sat(target, cardinality=False)
        self.stream.seek(0, os.SEEK_END)

        # encode complexity cardinality using shadow variables
        # own shadows
        for _u, counts in self.shadow_counts.items():
            u = self.node_lookup[_u]
            # clauses: forced arcs from u to shadows
            if counts > 0:
                self._add_comment(f"begin forced arcs from {_u}->shadows({_u}) {set(self.shadow_lookup[u].values())}")
            for i in range(counts):
                shadow = self.shadow_lookup[u][i]
                self._add_clause(self.arc[u][shadow])
            if counts > 0:
                self._add_comment(f"end forced arcs from {_u}->shadows({_u}) {set(self.shadow_lookup[u].values())}")

        # other's shadows
        for _u in self.shadow_counts.keys():
            for _v, counts in self.shadow_counts.items():
                if _u == _v: continue
                # clauses: shadowing main arcs u->v
                if counts > 1: self._add_comment(f"begin {_u}->{_v} shadowing")
                u, v = self.node_lookup[_u], self.node_lookup[_v]
                for i in range(1, counts):
                    shadow = self.shadow_lookup[v][i]
                    self._add_clause(-self.arc[u][v], self.arc[u][shadow])
                if counts > 1: self._add_comment(f"end {_u}->{_v} shadowing")

        # now encode cardinality (accounting for shadow arcs)
        self._add_comment("begin encode cardinality")
        self.encode_cardinality_sat(target, self.arc)
        self._add_comment("end encode cardinality")

        # update header
        self.replace_sat_placeholder()

    def show_counters(self, model: set, elim_order):
        def name(idx):
            if idx in self.node_reverse_lookup:
                return self.node_reverse_lookup[idx]
            else:
                var, ct = self.shadow_reverse_lookup[idx]
                return f"{var}_{ct}"
        variables = self.arc
        ctr_var_list = self.cardinality_counter_vars
        for i in range(len(ctr_var_list)):
            iname = name(i)
            ctr_vars = ctr_var_list[i]
            arc_count = 0
            # for j, jvar in zip(ctr_vars, sorted(variables[i].keys())):
            # for j in range(len(ctr_vars)):
            for j, jvar in enumerate(sorted(variables[i].keys())):
                jname = name(jvar)
                try:
                    appears_before = (elim_order.index(iname) < elim_order.index(jname))
                except ValueError:
                    appears_before = True
                arc_exists = appears_before and variables[i][jvar] in model
                arc_count += arc_exists
                print(f"{iname}->{jname}" + ("*" if arc_exists else " "), end="\t:\t")
                print(*(int(var in model) for var in ctr_vars[j]))
            print(f"({arc_count} arcs)\n")


def solve_graph(graph, shadow_counts, complexity_width, timeout: int = TIMEOUT,
                debug=False):
    cnfpath = "temp-cw.cnf"
    # logpath = "temp-cw.log"
    with open(cnfpath, 'w') as cnffile:
        enc = SvEncodingWithComplexity(cnffile, graph, shadow_counts, debug)
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


def get_shadow_counts(domain_sizes):
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
    shadow_counts = get_shadow_counts(ds)

    print("ds:", ds)
    print("shadow count:", shadow_counts)

    # g.remove_edge(0, 3)
    # g.add_edge(0, 4)
    # shadow_counts[4] = 2

    # shadow_counts = {node: 1 for node in g.nodes}
    # shadow_counts['h'] = 2
    # shadow_counts['a'] = 2
    td = solve_graph(g, shadow_counts, complexity_width=10, timeout=30, debug=True)
    print(td.elim_order)
    cw = compute_complexity_width(td, ds)
    print("final cw:", cw)
    # td.draw()
