#!/bin/python3.6

import networkx as nx
import itertools
import random
from typing import Union, Tuple, Dict, List, Iterator, FrozenSet, TextIO, Set
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


def elem_apply(fns, elems):
    """apply list of functions to list of elements,
    element-wise"""
    return map(lambda a, b: a(b), fns, elems)

def posdict_to_ordering(positions: dict):
    ordering = [-1]*len(positions)
    for elem, pos in positions.items():
        ordering[pos] = elem
    return ordering


# i/o utility functions

class FileReader(object):  # todo[safety]: add support for `with` usage
    def __init__(self, filename: str, ignore="#"):
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
Psets = Dict[FrozenSet[int], float]
BNStream = Iterator[Tuple[int, Psets]]
BNData = Dict[int, Psets]


def stream_jkl(filename: str, normalize=True):
    reader = FileReader(filename, ignore="#")
    n = reader.readint()
    for i in range(n):
        psets: Psets = dict()
        minscore = 1e9
        node, numsets = reader.readints()
        for j in range(numsets):
            score, parents = reader.readline().split(sep=" ", maxsplit=1)
            score = float(score)
            parents = frozenset(map(int, parents.split()[1:]))
            psets[parents] = score
            minscore = min(score, minscore)
        if normalize:
            psets = {pset: score-minscore for pset, score in psets.items()}
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
            for parents, score in psets.items():
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
            yield node, psets


def filter_read_bn(filename: str, filterset, normalize=True) -> BNData:
    return dict(filter_stream_bn(filename, filterset, normalize))


def get_bn_stats(filename: str) -> Tuple[float, float, Dict[int, float]]:
    """
    returns sum of all scores and node-wise offsets used for normalizing
    :param filename:
    :return:  (sum_score, Dict[node, min_score])
    """
    sum_score = 0
    best_score = 0
    offsets = dict()
    for node, psets in stream_bn(filename, normalize=False):
        scores = psets.values()
        sum_score += sum(scores)
        best_score += max(scores)
        offsets[node] = min(scores)
    return sum_score, best_score, offsets


class NoSolutionException(BaseException): pass


def read_model(output: Union[str, TextIO]) -> set:
    if isinstance(output, str):
        output = output.split("\n")
    for line in output:
        if line.startswith("v"):
            return set(map(int, line.split()[1:]))
    # if model found, this line should not be reached
    if isinstance(output, TextIO): output.seek(0)
    with open("err-output.log", 'w') as err_out:
        for line in output:
            err_out.write(line)
    raise NoSolutionException("model not found (no line starting with 'v'\n\t"
                              "output written to err-output.log")

def read_model_from_file(filename: str) -> set:
    with open(filename) as out:
        return read_model(out)


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


def check_subgraph(graph: nx.Graph, subgraph: nx.Graph):
    for edge in subgraph.edges:
        if not graph.has_edge(*edge):
            return False
    return True


def topsort(graph: nx.DiGraph, seed=None):
    #todo: complete
    rng = random.Random(seed)
    graph = graph.copy()
    sources = [v for v in graph if graph.in_degree(v) == 0]
    while graph:
        index = rng.randint(0, len(sources)-1)
        chosen = sources[index]
        yield chosen
        del sources[index]
        for nbr in graph.successors(chosen):
            if graph.in_degree(nbr) == 1:
                sources.append(nbr)
        graph.remove_node(chosen)


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
                f"Treewidth({width}) exceeded by ordering({max_degree}): {order}"
            self.width = width
        else:
            self.width = max_degree
        revorder = order[::-1]
        cur_nodes = set(revorder[:self.width+1])
        root_bag = self.add_bag(cur_nodes)
        blame = {node: root_bag for node in cur_nodes}
        for u in revorder[self.width+1:]:
            cur_nodes.add(u)
            neighbors = set(graph.subgraph(cur_nodes).neighbors(u))
            if neighbors:
                first_neighbor = find_first_by_order(neighbors, order)
                parent = blame[first_neighbor]
            else:
                parent = root_bag
            bag_idx = self.add_bag(neighbors | {u}, parent)
            blame[u] = bag_idx
        if __debug__: self.verify()

    def verify(self, graph: nx.Graph=None):
        if graph is None: graph = self.graph
        # check if tree
        assert nx.is_tree(self.decomp), "decomp is not a tree"
        # check width
        max_bag_size = max(map(len, self.bags.values()))
        assert max_bag_size <= self.width + 1, \
            f"decomp width too high ({max_bag_size} > {self.width + 1})"
        # check vertex connected subtree
        for node in graph.nodes:
            bags_containing = [bag_id for bag_id in self.bags
                               if node in self.bags[bag_id]]
            assert nx.is_connected(self.decomp.subgraph(bags_containing)), \
                f"subtree for vertex {node} is not connected"
        # check if every edge covered
        for edge in graph.edges:
            for bag in self.bags.values():
                if bag.issuperset(edge):
                    break
            else:
                raise AssertionError(f"edge {edge} not covered by decomp")
            continue
        new_elim_order = self.recompute_elim_order()
        fgraph, max_degree = filled_in(graph, new_elim_order)
        assert max_degree <= self.width, f"newly computed elim order invalid" \
                                         f"{max_degree} > {self.width}"

    def get_boundary_intersections(self, selected) -> Dict[int, Dict[int, frozenset]]:
        intersections = {bag_id: dict() for bag_id in selected}
        for bag_id, nbr_id in nx.edge_boundary(self.decomp, selected):
            assert bag_id in selected, "edge boundary pattern assumption failed"
            intersections[bag_id][nbr_id] = self.bags[bag_id] & self.bags[nbr_id]
        return intersections

    def draw(self, subset=None):
        if subset is None:
            decomp = self.decomp
        else:
            decomp = self.decomp.subgraph(subset)
        pos = graphviz_layout(decomp, prog='dot')
        labels = {bag_idx: f"{bag_idx}{list(bag)}"
                  for (bag_idx, bag) in self.bags.items() if bag_idx in pos}
        nx.draw(decomp, pos)
        nx.draw_networkx_labels(decomp, pos, labels=labels)
        plt.show()

    def replace(self, selected, forced_cliques, new_td: 'TreeDecomposition'):
        remap = dict()
        # add new bags
        for old_id, bag in new_td.bags.items():
            new_id = self.add_bag(bag)
            remap[old_id] = new_id
        # add new edges
        for b1, b2 in new_td.decomp.edges:
            self.decomp.add_edge(remap[b1], remap[b2])

        # connect new bags to those outside selected
        # todo[opt]: smart patching (avoid brute force, use elim ordering td)
        for nbr_id, intersection in forced_cliques.items():
            # find bag in new_td which contains intersection
            req_bag_id = new_td.bag_containing(intersection)
            assert req_bag_id != -1,\
                f"required bag containing {set(intersection)} not found"
            self.decomp.add_edge(remap[req_bag_id], nbr_id)

        # noinspection PyUnreachableCode
        if __debug__:
            covered_nodes = set()
            for bag in new_td.bags.values():
                covered_nodes.update(bag)
            existing_nodes = set()
            for sel_idx in selected:
                existing_nodes.update(self.bags[sel_idx])
            assert covered_nodes == existing_nodes, \
                f"replacement td mismatch, " \
                f"existing: {existing_nodes}\tcovered: {covered_nodes}"

        # delete old bags which have been replaced
        for sel_idx in selected:
            del self.bags[sel_idx]
            self.decomp.remove_node(sel_idx)

    def bag_containing(self, members: Union[set, frozenset],
                       exclude: Set[int] = None) -> int:
        """
        returns the id of a bag containing given members
        if no such bag exists, returns -1
        """
        exclude = set() if exclude is None else exclude
        for bag_id, bag in self.bags.items():
            if bag_id in exclude: continue
            if bag.issuperset(members):
                return bag_id
        return -1

    def recompute_elim_order(self) -> list:
        """
        recomputes elimination ordering based on possibly modified
        decomposition bags

        :return: new elim_order as list
        """
        rootbag = first(self.bags)  # arbitrarily choose a root bag
        elim_order = list(self.bags[rootbag])  # initialize eo with rootbag
        for u, v in nx.dfs_edges(self.decomp, source=rootbag):
            forgotten = self.bags[v] - self.bags[u]
            elim_order.extend(forgotten)
        elim_order.reverse()
        return elim_order


if __name__ == '__main__':
    g = nx.bull_graph()
    g.add_edges_from([(4, 5), (2, 5)])
    ordering = list(range(len(g)))
    fgraph = filled_in(g, ordering)
    td = TreeDecomposition(g, ordering, 3)
    td.draw()
