#!/bin/python3.6

# external
import os, sys
import networkx as nx
import random

# internal
from blip import run_blip, BayesianNetwork
from utils import TreeDecomposition, pick, pairs, read_bn, write_jkl

# optional
from networkx.drawing.nx_agraph import pygraphviz_layout
import matplotlib.pyplot as plt

BUDGET = 7


class SubTreeDecomposition(object):
    def __init__(self, td: TreeDecomposition, selected):
        pass


def find_subtree(td: TreeDecomposition, debug=False):
    start_bag_id = pick(td.bags.keys())
    selected = {start_bag_id}
    seen = set(td.bags[start_bag_id])
    if debug: print(f"starting bag {start_bag_id}: {td.bags[start_bag_id]}")
    for layer, bag_ids in nx.bfs_successors(td.decomp, start_bag_id):
        no_new_inclusion = True
        for bag_id in bag_ids:
            bag = td.bags[bag_id]
            if len(seen.union(bag)) > BUDGET:
                continue
            else:
                selected.add(bag_id)
                seen.update(bag)
                if debug: print(f"added bag {bag_id}: {td.bags[bag_id]}")
                no_new_inclusion = False
        if no_new_inclusion:
            break
    if debug: print(f"final seen: {seen}")
    return selected, seen


def handle_acyclicity(bn: BayesianNetwork, seen: set, leaf_nodes: set, debug=False):
    dag = bn.get_dag()
    inner_nodes = seen - leaf_nodes
    outer_nodes = dag.nodes - inner_nodes
    subdag = nx.subgraph_view(dag, outer_nodes.__contains__,
                              lambda x,y: not ((x in seen) and (y in seen)))
    if debug:
        nx.draw(subdag, pygraphviz_layout(subdag), with_labels=True)
        plt.show()
    forced_parents = {node: set() for node in leaf_nodes}
    if debug: print(f"nodes leaf:{leaf_nodes}\tinner:{inner_nodes}\touter:{outer_nodes}")
    for src, dest in pairs(leaf_nodes):
        if nx.has_path(subdag, src, dest):
            forced_parents[dest].add(src)
            if debug: print(f"added forced {dest}<-{src}")
        else:
            # only check if prev path not found
            if nx.has_path(subdag, dest, src):
                forced_parents[src].add(dest)
                if debug: print(f"added forced {src}<-{dest}")
    return forced_parents


def prepare_subtree(bn: BayesianNetwork, td: TreeDecomposition, bag_ids: set,
                    seen: set, debug=False):
    # compute leaf bag ids
    # compute leaf nodes (based on intersection of leaf bags with outside)
    leaf_bag_ids = set()
    leaf_nodes = set()
    forced_cliques = []
    # for bag_id in bag_ids:
    #     is_leaf = False
    #     for nbr_id in td.decomp.neighbors(bag_id):
    #         if nbr_id in bag_ids: continue
    #         is_leaf = True
    #         common = td.bags[bag_id] & td.bags[nbr_id]
    #         leaf_nodes.update(common)
    #     if is_leaf: leaf_bag_ids.add(bag_id)
    for bag_id, nbr_id in nx.edge_boundary(td.decomp, bag_ids):
        assert bag_id in bag_ids, "edge boundary pattern assumption failed"
        leaf_bag_ids.add(bag_id)
        common = td.bags[bag_id] & td.bags[nbr_id]
        forced_cliques.append(common)
        leaf_nodes.update(common)

    # compute forced parent data for leaf nodes
    if debug: print(f"leaf bags: {leaf_bag_ids}\tleaf nodes:{leaf_nodes}")
    forced_parents = handle_acyclicity(bn, seen, leaf_nodes, debug)
    if debug: print("forced parents", forced_parents)

    # copy over bn data for inner nodes respecting forced parents
    # input_data = read_bn(start_bn.input_file)
    # new_input = {node: input_data[node] for node in seen - leaf_nodes}
    # for node in leaf_nodes:
    #     forced = forced_parents[node]
    #     # new_parents = list(filter(lambda p: forced.issubset(p[1]), input_data[node]))
    #     new_parents = []
    #     for score, parents in input_data[node]:
    #         if forced.issubset(parents):
    #             new_parents.append((score, parents))
    #     new_input[node] = new_parents

    # construct forced clique edges on leaf nodes per bag
    if debug: print(f"clique sets: {forced_cliques}")
    return forced_parents, forced_cliques




if __name__ == '__main__':
    filename = "../past-work/blip-publish/data/child-5000.jkl"
    print("running blip...")
    start_bn = run_blip(filename, 5, timeout=2, seed=9)
    # start_bn.draw()
    print("done")
    start_td = start_bn.td
    # start_td.draw()
    random.seed(9)
    selected, seen = find_subtree(start_td, debug=True)
    print("found subtree", "sel:", selected, "\tseen:", seen)
    forced_parents, forced_cliques = prepare_subtree(start_bn, start_td, selected, seen, True)
    # write_jkl(new_input, "testnew.jkl")

