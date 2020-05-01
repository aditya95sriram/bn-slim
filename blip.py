#!/bin/python3.6

import subprocess
import os
import networkx as nx
import random
from typing import Optional

from utils import filled_in, TreeDecomposition, pairs, stream_bn, get_bn_stats, filter_read_bn

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import pygraphviz_layout

SOLVER_DIR = "../solvers"


class BayesianNetwork(object):
    def __init__(self, input_file, data=None, parents=None):
        self.parents = dict() if parents is None else parents
        self.input_file = input_file
        self.data = data
        self.sum_scores, self.offsets = get_bn_stats(self.input_file)
        self._score = None
        self._dag = None

    def _clear_cached(self):
        self._score = self._dag = None

    def compute_score(self, subset: set=None) -> float:
        """recompute score based on parents sets"""
        if self.data is None:
            if subset is None:
                data = stream_bn(self.input_file, normalize=True)
            else:
                data = filter_read_bn(self.input_file, subset, normalize=True).items()
        else:
            data = self.data.items()  # data is assumed to be normalized
        score = 0
        for node, psets in data:
            if node in self.parents:  # could be a bn with a subset of nodes
                if subset is None or node in subset:
                    score += psets[self.parents[node]] + self.offsets[node]
        return score

    @property
    def score(self) -> float:
        if self._score is None:
            self._score = self.compute_score()
        return self._score

    def add(self, node, parents):
        self.parents[node] = frozenset(parents)
        self._clear_cached()

    def recompute_dag(self):
        dag = nx.DiGraph()
        for node in self.parents.keys():
            dag.add_node(node)
            for parent in self.parents[node]:
                assert not dag.has_edge(node, parent), f"cyclic parent set {node}<->{parent}"
                dag.add_edge(parent, node)
        self._dag = dag

    @property
    def dag(self) -> nx.DiGraph:
        if self._dag is None:
            self.recompute_dag()
        return self._dag

    def verify(self):
        assert nx.is_directed_acyclic_graph(self.dag), "network is not acyclic"

    def get_moralized(self):
        moral = self.dag.to_undirected()
        for node in self.parents.keys():
            for p1, p2 in pairs(self.parents[node]):
                moral.add_edge(p1, p2)
        return moral

    def draw(self):
        dag = self.dag
        pos = pygraphviz_layout(dag, prog='dot')
        nx.draw(dag, pos, with_labels=True)
        plt.show()

    def replace(self, newbn: 'BayesianNetwork'):
        for node, new_parents in newbn.parents.items():
            self.parents[node] = new_parents
        self._clear_cached()


class TWBayesianNetwork(BayesianNetwork):
    def __init__(self, input_file, tw=0, elim_order=None, td=None, *args, **kwargs):
        super().__init__(input_file, *args, **kwargs)
        self.tw = tw
        self.elim_order = elim_order
        self._td: Optional[TreeDecomposition] = td

    @property
    def td(self) -> TreeDecomposition:
        if self._td is None:
            raise RuntimeError("td requested before finalizing (call `.done`)")
        return self._td

    def done(self):
        """
        compute and store tree decomp and width based on
        elim_order(default: reverse topological order of the dag)
        """
        if self.elim_order is None:
            self.elim_order = list(nx.topological_sort(self.dag))[::-1]
        self._td = TreeDecomposition(self.get_moralized(), self.elim_order, self.tw)
        if self.tw <= 0:
            self.tw = self.td.width

    def verify(self):
        super().verify()
        self.td.verify()

    def get_triangulated(self, elim_order=None):
        if elim_order is None: elim_order = self.elim_order
        assert elim_order is not None, "elim order not specified"
        triangulated, max_degree = filled_in(self.get_moralized(), self.elim_order)
        _, max_degree2 = filled_in(self.get_moralized(), self.elim_order[::-1])
        # print("tw", self.tw, "max degrees", max_degree, max_degree2)
        assert max_degree <= self.tw, "rev top order failed"
        return triangulated


def run_blip(filename, treewidth, outfile="temp.out", timeout=10, seed=0,
             solver="kg", logfile="temp.log", debug=False):
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"),
               f"solver.{solver}", "-v", "1"]
    args = ["-j", filename, "-w", str(treewidth+1), "-r", outfile,
            "-t", str(timeout), "-seed", str(seed), "-l", logfile]
    # blip width convention is off by one (trees have width 2)
    cmd = basecmd + args
    if debug: print("running blip, cmd:", cmd)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    if proc.returncode == 0:  # success
        bn = TWBayesianNetwork(tw=treewidth, input_file=filename)
        with open(outfile) as out:
            for line in out:
                if not line.strip(): continue
                elif line.startswith("Score:"):
                    continue
                else:
                    vertex, rest = line.split(":")
                    rest = rest.strip().split()
                    if len(rest) == 1:
                        local_score = float(rest[0])
                        parents = []
                    else:
                        local_score, parents = rest
                        parents = parents.strip("()").split(",")
                    bn.add(int(vertex), map(int, parents))
                    if debug: print(f"v:{vertex}, sc:{local_score}, par:{parents})")
        bn.done()
        return bn
    else:  # error
        print("error encountered, returncode:", proc.returncode)
        return None


if __name__ == '__main__':
    # res = run_blip("../past-work/blip-publish/data/child-5000.jkl", 10, timeout=2, seed=4)
    # dag = res.get_dag()
    # print("acyclic", nx.is_directed_acyclic_graph(dag))
    # fig, axes = plt.subplots(1, 3)
    # nx.draw(dag, with_labels=True, ax=axes[0])
    # moral = res.get_moralized()
    # nx.draw(moral, with_labels=True, ax=axes[1])
    # tri = res.get_triangulated()
    # nx.draw(tri, with_labels=True, ax=axes[2])
    # fig.show()
    for seed in range(1,100):
        print(seed)
        res = run_blip("../past-work/blip-publish/data/child-5000.jkl", 10, timeout=2, seed=seed, debug=False)
        tri = res.get_triangulated()
    print("done")

