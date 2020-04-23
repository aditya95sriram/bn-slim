#!/bin/python3.6

import networkx as nx
import itertools
import random
from typing import Union, Tuple, Dict, List, Iterator
import sys, os
from operator import itemgetter

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


# general purpose functions

def first(obj):
    """return first element from object
    (also consumes it if obj is an iterator)"""
    return next(iter(obj))


def pick(obj):
    """randomly pick an element from obj"""
    return random.sample(obj, 1)[0]


def pairs(obj):
    """return all unordered pairs made from
    distinct elements of obj"""
    return itertools.combinations(obj, 2)


def ord_triples(obj):
    """return all ordered triples made from
    distinct elements of obj"""
    return itertools.permutations(obj, 3)


def posdict_to_ordering(positions: dict):
    ordering = [-1]*len(positions)
    for elem, pos in positions.items():
        ordering[pos] = elem
    return ordering


# i/o utility functions

class FileReader(object):  # todo[safety]: add support for `with` usage
    def __init__(self, filename:str, ignore="#"):
        self.file = open(filename)
        self.ignore = ignore

    def readline(self):
        line = self.file.readline()
        while line.startswith(self.ignore):
            line = self.file.readline()
        return line

    def readints(self):
        return map(int, self.readline().split())

    def readint(self):
        return int(self.readline().strip())

    def close(self):
        self.file.close()

# bn datatypes
Psets = List[Tuple[float, frozenset]]
BNStream = Iterator[Tuple[int, Psets]]
BNData = Dict[int, Psets]


def stream_jkl(filename: str, normalize=True):
    reader = FileReader(filename, ignore="#")
    n = reader.readint()
    for i in range(n):
        psets = []
        minscore = 1e9
        node, numsets = reader.readints()
        for j in range(numsets):
            score, parents = reader.readline().split(sep=" ", maxsplit=1)
            score = float(score)
            parents = frozenset(map(int, parents.split()[1:]))
            psets.append((score, parents))
            minscore = min(score, minscore)
        if normalize:
            psets = [(score - minscore, parents) for score, parents in psets]
        yield node, psets
    reader.close()


def read_jkl(filename: str, normalize=True):
    return dict(stream_jkl(filename, normalize))


def num_nodes_jkl(filename: str):
    reader = FileReader(filename, ignore="#")
    n = reader.readint()
    reader.close()
    return n


def write_jkl(data, filename):
    n = len(data)
    with open(filename, 'w') as outfile:
        outfile.write(f"{n}\n")
        for node, psets in sorted(data.items(), key=itemgetter(0)):
            outfile.write(f"{node} {len(psets)}\n")
            for score, parents in psets:
                outfile.write(f"{score:.4f} {len(parents)}")
                for parent in sorted(parents):
                    outfile.write(f" {parent}")
                outfile.write("\n")


def stream_bn(filename: str, normalize=True) -> BNStream:
    path, ext = os.path.splitext(filename)
    if ext == ".jkl":
        return stream_jkl(filename, normalize)
    else:
        print(f"unknown file format '{ext}'")


def read_bn(filename: str, normalize=True) -> BNData:
    return dict(stream_bn(filename, normalize))


def num_nodes_bn(filename: str) -> int:
    path, ext = os.path.splitext(filename)
    if ext == ".jkl":
        return num_nodes_jkl(filename)
    else:
        print(f"unknown file format '{ext}'")


def filter_stream_bn(filename: str, filterset, normalize=True) -> BNStream:
    for node, psets in stream_bn(filename, normalize):
        if node in filterset:
            filtered = list(filter(lambda a: a[1].issubset(filterset), psets))
            yield node, filtered


def filter_read_bn(filename: str, filterset, normalize=True) -> BNData:
    return dict(filter_stream_bn(filename, filterset, normalize))


def read_model(filename: str) -> set:
    with open(filename) as out:
        for line in out:
            if line.startswith("v"):
                return set(map(int, line.split()[1:]))
    print("model not found (no line starting with 'v')")
    return set()


# treewidth related functions

def filled_in(graph, order) -> Tuple[nx.Graph, int]:
    fgraph = graph.copy()
    cur_nodes = set(graph.nodes)
    max_degree = -1
    for u in order:
        trunc_graph = fgraph.subgraph(cur_nodes)
        max_degree = max(max_degree, trunc_graph.degree(u))
        neighbors = trunc_graph.neighbors(u)
        fgraph.add_edges_from(itertools.combinations(neighbors, 2))
        cur_nodes.remove(u)
    return fgraph, max_degree


def find_first_by_order(elements: set, order):
    for element in order:
        if element in elements: return element


class TreeDecomposition(object):
    def __init__(self, graph: nx.Graph, order, width=0):
        self.bags: Dict[int, frozenset] = dict()
        self.decomp = nx.Graph()
        self.graph = graph
        self.elim_order = order
        self.width = width
        self._bag_ctr = 0
        self.decomp_from_ordering(graph, order, width)

    def add_bag(self, nodes: Union[set, frozenset], parent: int = -1):
        nodes = frozenset(nodes)
        bag_idx = self._bag_ctr
        self._bag_ctr += 1
        self.bags[bag_idx] = nodes
        self.decomp.add_node(bag_idx)
        if parent >= 0:
            self.decomp.add_edge(parent, bag_idx)
        return bag_idx

    def decomp_from_ordering(self, graph, order, width):
        graph, max_degree = filled_in(graph, order)
        if width > 0:
            assert max_degree <= width, \
            f"Treewidth({width}) exceeded by ordering: {order}"
        self.width = max_degree
        revorder = order[::-1]
        cur_nodes = set(revorder[:width+1])
        root_bag = self.add_bag(cur_nodes)
        blame = {node: root_bag for node in cur_nodes}
        for u in revorder[width+1:]:
            cur_nodes.add(u)
            neighbors = set(graph.subgraph(cur_nodes).neighbors(u))
            if neighbors:
                first_neighbor = find_first_by_order(neighbors, order)
                parent = blame[first_neighbor]
            else:
                parent = root_bag
            bag_idx = self.add_bag(neighbors | {u}, parent)
            blame[u] = bag_idx

    def draw(self):
        labels = {bag_idx: f"{bag_idx}{list(bag)}" for (bag_idx, bag) in self.bags.items()}
        pos = graphviz_layout(self.decomp, prog='dot', root=0)
        nx.draw(self.decomp, pos)
        nx.draw_networkx_labels(self.decomp, pos, labels=labels)
        plt.show()

    def replace(self, selected, new_td: 'TreeDecomposition'):
        # delete old bags which are going to be replaced
        covered_nodes = set()
        for sel_idx in selected:
            covered_nodes.update(self.bags[sel_idx])
            del self.bags[sel_idx]
            self.decomp.remove_node(sel_idx)

        remap = dict()
        # add new bags
        for old_idx, bag in new_td.bags.items():
            new_idx = self.add_bag(bag)
            remap[old_idx] = new_idx
        # add new edges
        for b1, b2 in new_td.decomp.edges:
            self.decomp.add_edge(remap[b1], remap[b2])

        # todo[crit]: connect new bags to ones outside selected


if __name__ == '__main__':
    g = nx.bull_graph()
    g.add_edges_from([(4, 5), (2, 5)])
    ordering = list(range(len(g)))
    fgraph = filled_in(g, ordering)
    td = TreeDecomposition(g, ordering, 3)
    td.draw()
