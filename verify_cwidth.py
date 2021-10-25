#!/bin/python3.6

import sys
from pprint import pprint, pformat
import networkx as nx
from itertools import combinations

from utils import TreeDecomposition


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        eprint("invalid number of arguments")
        print("usage: python verify_cwidth.py <resfile> <datfile> <cwidth>")
        sys.exit()

resfile = sys.argv[1]
datfile = sys.argv[2]
cwidth = int(sys.argv[3])

elim_order = []
spset = dict()

with open(resfile) as rf:
    for line in rf:
        if not line.strip(): continue
        key, value = line.strip().split(":")
        if key == "elim-order":
            value = value.strip().strip("()")
            elim_order = list(map(int, value.split(",")))
        elif key.isdigit():
            try:
                score, pset = value.split()
            except ValueError:
                pset = set()
                score = value
            else:
                pset = set(map(int, pset.strip("()").split(",")))
            finally:
                score = float(score)
                spset[int(key)] = (score, pset)

eprint("elim-order:", elim_order)
eprint(pformat(spset))

with open(datfile) as df:
    line = df.readline()
    domain_sizes = dict(zip(range(100), map(int, df.readline().strip().split())))


def bag_complexity(pset, node=None):
    complexity = domain_sizes[node] if node is not None else 1
    for p in pset: complexity *= domain_sizes[p]
    return complexity


rootbag_complexity, rootbag_size = 1, 0
for node in elim_order[::-1]:
    rootbag_complexity *= domain_sizes[node]
    if rootbag_complexity > cwidth: break
    rootbag_size += 1
eprint("rootbag size:", rootbag_size)

bn = nx.DiGraph()
mbn = nx.Graph()
for node, (score, pset) in spset.items():
    for p in pset:
        bn.add_edge(p, node)
        mbn.add_edge(p, node)
    mbn.add_edges_from(combinations(pset, 2))  # moral edges

td = TreeDecomposition(mbn, elim_order)

actual_cwidth = max(map(bag_complexity, td.bags.values()))
print("treewidth:", td.width)
print("complexity-width:", actual_cwidth)

assert nx.algorithms.is_directed_acyclic_graph(bn), "BN is not DAG"
assert actual_cwidth <= cwidth, f"complexity-width({actual_cwidth}) exceeds {cwidth}"

print("verified!")
