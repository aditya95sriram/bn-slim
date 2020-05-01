#!/bin/python3.6

# external
import os, sys
import networkx as nx
import random
from typing import Dict, List, Tuple, Set, FrozenSet
from pprint import pprint
from time import time as now

# internal
from blip import run_blip, BayesianNetwork, TWBayesianNetwork
from utils import TreeDecomposition, pick, pairs, filter_read_bn, BNData, stream_bn
from berg_encoding import solve_bn

# optional
from networkx.drawing.nx_agraph import pygraphviz_layout
import matplotlib.pyplot as plt

BUDGET = 7
TIMEOUT = 5


class SubTreeDecomposition(object):
    def __init__(self, td: TreeDecomposition, selected):
        pass


def find_subtree(td: TreeDecomposition, budget:int, debug=False):
    """
    finds a subtree that fits within the budget

    :param td: tree decomposition in which to find the subtree
    :param budget: max number of vertices allowed in the union of selected bags
    :param debug: debug mode
    :return: (selected_bag_ids, seen_vertices)
    """
    start_bag_id = pick(td.bags.keys())
    selected = {start_bag_id}
    seen = set(td.bags[start_bag_id])
    if debug: print(f"starting bag {start_bag_id}: {td.bags[start_bag_id]}")
    for layer, bag_ids in nx.bfs_successors(td.decomp, start_bag_id):
        no_new_inclusion = True
        for bag_id in bag_ids:
            bag = td.bags[bag_id]
            if len(seen.union(bag)) > budget:
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
    dag = bn.dag
    inner_nodes = seen - leaf_nodes
    outer_nodes = dag.nodes - inner_nodes
    subdag = nx.subgraph_view(dag, outer_nodes.__contains__,
                              lambda x,y: not ((x in seen) and (y in seen)))
    forced_arcs = []
    if debug: print(f"nodes leaf:{leaf_nodes}\tinner:{inner_nodes}")
    for src, dest in pairs(leaf_nodes):
        if nx.has_path(subdag, src, dest):
            forced_arcs.append((src, dest))
            if debug: print(f"added forced {src}->{dest}")
        else:
            # only check if prev path not found
            if nx.has_path(subdag, dest, src):
                forced_arcs.append((dest, src))
                if debug: print(f"added forced {dest}->{src}")
    return forced_arcs


def prepare_subtree(bn: TWBayesianNetwork, bag_ids: set, seen: set, debug=False):
    # compute leaf nodes (based on intersection of leaf bags with outside)
    boundary_nodes = set()
    forced_cliques: Dict[int, frozenset] = dict()
    for bag_id, nbrs in bn.td.get_boundary_intersections(bag_ids).items():
        for nbr_id, intersection in nbrs.items():
            boundary_nodes.update(intersection)
            forced_cliques[nbr_id] = intersection
    if debug: print("clique sets:", [set(c) for c in forced_cliques.values()])

    # compute forced arc data for leaf nodes
    if debug: print("boundary nodes:", boundary_nodes)
    forced_arcs = handle_acyclicity(bn, seen, boundary_nodes, debug)
    if debug: print("forced arcs", forced_arcs)

    data, substitutions = get_data_for_subtree(bn, boundary_nodes, seen)

    return forced_arcs, forced_cliques, data, substitutions


def get_data_for_subtree(bn: TWBayesianNetwork, boundary_nodes: set, seen: set) \
        -> Tuple[BNData, Dict[int, Dict[frozenset, frozenset]]]:
    # compute downstream global green vertices
    downstream = set()
    for node in boundary_nodes:
        for layer, successors in nx.bfs_successors(bn.dag, node):
            downstream.update(successors)
    downstream -= seen  # ignore local red vertices

    # construct score function data for local instance
    data: BNData = {node: dict() for node in seen}
    substitutions = {node: dict() for node in seen}
    for node, psets in filter_read_bn(bn.input_file, seen).items():
        if node in boundary_nodes:
            for pset, score in psets.items():
                pset_in = pset.intersection(seen)
                # all internal vertices, so allowed
                if len(pset_in) < len(pset):
                    pset_out = pset - pset_in
                    if pset_out.intersection(downstream):
                        continue  # reject because pset contains downstream verts
                    bag_id = bn.td.bag_containing(pset | {node})  # todo[opt]: exclude selected
                    if bag_id == -1:
                        continue  # reject because required bag doesnt already exist in td
                if pset_in not in data[node] or data[node][pset_in] < score:
                    data[node][pset_in] = score
                    if len(pset_in) < len(pset):
                        substitutions[node][pset_in] = pset
        else:  # internal node
            for pset, score in psets.items():
                # internal vertices are not allowed outside parents
                if pset.issubset(seen):
                    data[node][pset] = score
    return data, substitutions


def slimpass(bn: TWBayesianNetwork, budget: int, timeout: int, debug=False):
    td = bn.td
    tw = td.width
    selected, seen = find_subtree(td, budget, debug=False)
    forced_arcs, forced_cliques, data, subs = prepare_subtree(bn, selected, seen, debug)
    if debug:
        print("filtered data:-")
        pprint(data)
    old_score = bn.compute_score(seen)
    pos = pygraphviz_layout(bn.dag, prog='dot')
    if debug:
        nx.draw(bn.dag, pos, with_labels=True)
        plt.suptitle("entire dag")
        plt.show()
        print("old parents:-")
        pprint({node: par for node, par in bn.parents.items() if node in seen})
        nx.draw(bn.dag.subgraph(seen), pos, with_labels=True)
        plt.suptitle("subdag before improvement")
        plt.show()
    replbn = solve_bn(data, tw, bn.input_file, forced_arcs, forced_cliques, timeout, debug)
    # perform substitutions
    for node, pset in replbn.parents.items():
        if pset in subs[node]:
            if debug: print(f"performed substition {pset}->{subs[node][pset]}")
            replbn.parents[node] = subs[node][pset]
    replbn.recompute_dag()
    new_score = replbn.compute_score()
    if debug:
        print("new parents:-")
        pprint(replbn.parents)
        nx.draw(replbn.dag, pos, with_labels=True)
        plt.suptitle("replacement subdag")
        plt.show()
    if debug: print(f"score change: {old_score:.3f} -> {new_score:.3f}")
    if new_score >= old_score:  # replacement condition
        td.replace(selected, forced_cliques, replbn.td)
        # update bn with new bn
        bn.replace(replbn)
        bn.verify()


def slim(filename: str, treewidth: int, budget: int, timeout: int,
         num_passes=10, debug=False):
    bn = run_blip(filename, treewidth, timeout=2, seed=9)
    prev_score = bn.score
    print(f"Starting score: {prev_score:.5f}")
    start = now()
    for i in range(num_passes):
        slimpass(bn, budget, timeout, debug=False)
        new_score = bn.score
        if new_score > prev_score:
            print(f"New improvement! {new_score:.5f} (after {now()-start:.1f} s.)")
            prev_score = new_score
        if debug: print(f"Iteration {i} score:\t{bn.score:.5f}\n")
    print(f"done (after {now()-start:.1f} s.)")


if __name__ == '__main__':
    # filename = "../past-work/blip-publish/data/child-5000.jkl"
    filename = "child-norm.jkl"
    random.seed(5)
    slim(filename, 5, BUDGET, TIMEOUT, 25, debug=False)
    sys.exit()
    print("running blip...")
    start_bn = run_blip(filename, 5, timeout=2, seed=9)
    # start_bn.draw()
    print("done")
    start_td = start_bn.td
    # start_td.draw()
    selected, seen = find_subtree(start_td, debug=True)
    print("found subtree", "sel:", selected, "\tseen:", seen)
    forced_arcs, forced_cliques = prepare_subtree(start_bn, start_td, selected, seen, True)
    # write_jkl(new_input, "testnew.jkl")

