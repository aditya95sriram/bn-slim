#!/bin/python3.6

# external
import os, sys
import networkx as nx
import random
from typing import Dict, List, Tuple, Set, FrozenSet, Callable
from pprint import pprint
from time import time as now
import argparse
import signal
from collections import Counter

# internal
from blip import run_blip, BayesianNetwork, TWBayesianNetwork, monitor_blip
from utils import TreeDecomposition, pick, pairs, filter_read_bn, BNData
from berg_encoding import solve_bn

# optional
import wandb

# check if drawing/plotting is available
CLUSTER = os.environ["WANDB_PLATFORM"] == "cluster"
if not CLUSTER:
    from networkx.drawing.nx_agraph import pygraphviz_layout
    import matplotlib.pyplot as plt


# default parameter values
BUDGET = 7
TIMEOUT = 5
MAX_PASSES = 100
MAX_TIME = 1800
HEURISTIC = 'kmax'
OFFSET = 2
SEED = 9
LAZY_THRESHOLD = 0.0


def find_start_bag(td: TreeDecomposition, history: Counter=None, debug=False):
    return pick(td.bags.keys())  # randomly pick a bag


def find_subtree(td: TreeDecomposition, budget:int, history: Counter=None,
                 debug=False):
    """
    finds a subtree that fits within the budget

    :param td: tree decomposition in which to find the subtree
    :param budget: max number of vertices allowed in the union of selected bags
    :param history: tally of bags picked in previous iterations
    :param debug: debug mode
    :return: (selected_bag_ids, seen_vertices)
    """
    start_bag_id = find_start_bag(td, history, debug)
    selected = {start_bag_id}
    seen = set(td.bags[start_bag_id])
    if debug: print(f"starting bag {start_bag_id}: {td.bags[start_bag_id]}")
    queue = [start_bag_id]
    visited = set()
    while queue:
        bag_id = queue.pop(0)
        visited.add(bag_id)
        bag = td.bags[bag_id]
        if len(seen.union(bag)) > budget:
            continue
        else:  # can include bag in local instance
            selected.add(bag_id)
            seen.update(bag)
            # add neighboring bags to queue
            for nbr_id in td.decomp.neighbors(bag_id):
                if nbr_id not in visited:
                    queue.append(nbr_id)
            if debug: print(f"added bag {bag_id}: {td.bags[bag_id]}")
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
    for _, nbrs in bn.td.get_boundary_intersections(bag_ids).items():
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


def compute_max_score(data: BNData, bn: BayesianNetwork) -> float:
    max_score = 0
    for node in data:
        max_score += max(data[node].values()) + bn.offsets[node]
    return max_score


class Solution(object):
    def __init__(self, value=None, logger: Callable=None):
        self.value: TWBayesianNetwork = value
        self.logger = logger
        # other metrics to track
        self.start_score = 0
        self.num_passes = 0
        self.num_improvements = 0
        self.skipped = 0

    def update(self, new_value):
        if self.logger is not None:
            self.logger({'score': new_value.score})
        self.value = new_value

    def log_potential(self, old, new, max, offset, best):
        if CLUSTER: return
        old = (old - offset)/best
        new = (new - offset)/best
        max = (max - offset)/best
        with open("potentials.csv", "a") as f:
            f.write(f"{old:.5f},{new:.5f},{max:.5f}\n")


SOLUTION = Solution()  # placeholder global solution variable


def slimpass(bn: TWBayesianNetwork, budget: int=BUDGET, timeout: int=TIMEOUT,
             history: Counter=None, debug=False):
    td = bn.td
    tw = td.width
    selected, seen = find_subtree(td, budget, history, debug=False)
    forced_arcs, forced_cliques, data, subs = prepare_subtree(bn, selected, seen, debug)
    # if debug:
    #     print("filtered data:-")
    #     pprint(data)
    old_score = bn.compute_score(seen)
    max_score = compute_max_score(data, bn)
    if max_score < old_score:
        print("#"*80)
        print("found culprit")
        sys.exit()
    cur_offset = sum(bn.offsets[node] for node in seen)
    if debug: print(f"potential max: {(max_score - cur_offset)/bn.best_norm_score:.5f}", end="")
    if (max_score - cur_offset)/bn.best_norm_score <= LAZY_THRESHOLD:
        if debug: print(" skipping ####")
        SOLUTION.skipped += 1
        return
    print()
    if not CLUSTER and debug:
        pos = pygraphviz_layout(bn.dag, prog='dot')
        nx.draw(bn.dag, pos, with_labels=True)
        plt.suptitle("entire dag")
        plt.show()
        nx.draw(bn.dag.subgraph(seen), pos, with_labels=True)
        plt.suptitle("subdag before improvement")
        plt.show()
    if debug:
        print("old parents:-")
        pprint({node: par for node, par in bn.parents.items() if node in seen})
    replbn = solve_bn(data, tw, bn.input_file, forced_arcs, forced_cliques, timeout, debug)
    # perform substitutions
    for node, pset in replbn.parents.items():
        if pset in subs[node]:
            if debug: print(f"performed substition {pset}->{subs[node][pset]}")
            replbn.parents[node] = subs[node][pset]
    replbn.recompute_dag()
    new_score = replbn.compute_score()
    SOLUTION.log_potential(old_score, new_score, max_score, cur_offset, bn.best_norm_score)
    recorded, potential = new_score-old_score, max_score-old_score
    if potential == 0:
        if debug: print("already at potential")
    else:
        if debug: print(f"percentage potential matched: {recorded/potential:.2%}")
    if not CLUSTER and debug:
        nx.draw(replbn.dag, pos, with_labels=True)
        plt.suptitle("replacement subdag")
        plt.show()
    if debug:
        print("new parents:-")
        pprint(replbn.parents)
    if debug: print(f"score change: {old_score:.3f} -> {new_score:.3f}")
    SOLUTION.num_passes += 1
    if new_score >= old_score:  # replacement condition
        SOLUTION.num_improvements += 1
        td.replace(selected, forced_cliques, replbn.td)
        # update bn with new bn
        bn.replace(replbn)
        bn.verify()


def slim(filename: str, treewidth: int, budget: int=BUDGET,
         sat_timeout: int=TIMEOUT, max_passes=MAX_PASSES, max_time: int=MAX_TIME,
         heuristic=HEURISTIC, offset: int=OFFSET, seed=SEED, debug=False):
    start = now()
    def elapsed(): return f"(after {now()-start:.1f} s.)"
    if debug: print(f"running initial heuristic for {offset}s")
    bn = run_blip(filename, treewidth, timeout=offset, seed=seed, solver=heuristic)
    bn.verify()
    SOLUTION.update(bn)
    prev_score = bn.score
    print(f"Starting score: {prev_score:.5f}")
    SOLUTION.start_score = prev_score
    history = Counter()
    if seed: random.seed(seed)
    for i in range(max_passes):
        slimpass(bn, budget, sat_timeout, history, debug=False)
        new_score = bn.score
        if new_score > prev_score:
            print(f"*** New improvement! {new_score:.5f} {elapsed()} ***")
            prev_score = new_score
            SOLUTION.update(bn)
        if debug: print(f"* Iteration {i}:\t{bn.score:.5f} {elapsed()}")
        if now()-start > max_time:
            if debug: print("time limit exceeded, quitting")
            break
    print(f"done {elapsed()}")


class SolverInterrupt(BaseException): pass


def term_handler(signum, frame):
    print(f"#### received signal {signum}, stopping...")
    raise SolverInterrupt


def register_handler():
    signums = [signal.SIGHUP, signal.SIGINT, signal.SIGTERM,
               signal.SIGUSR1, signal.SIGUSR2]
    for signum in signums:
        signal.signal(signum, term_handler)


def wandb_configure(wandb: wandb, args):
    basename, ext = os.path.splitext(os.path.basename(args.file))
    instance, samples = basename.split("-")
    wandb.config.instance = instance
    wandb.config.samples = int(samples)
    wandb.config.treewidth = args.treewidth
    wandb.config.budget = args.budget
    wandb.config.sat_timeout = args.sat_timeout
    wandb.config.heuristic = args.heuristic
    wandb.config.offset = args.offset
    wandb.config.threshold = args.lazy_threshold
    wandb.config.seed = args.random_seed
    wandb.config.method = 'heur' if args.compare else 'slim'


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("file", help="path to input file")
parser.add_argument("treewidth", help="bound for treewidth", type=int)
parser.add_argument("-b", "--budget", type=int, default=BUDGET,
                    help="budget for size of local instance")
parser.add_argument("-s", "--sat-timeout", type=int, default=TIMEOUT,
                    help="timeout per MaxSAT call")
parser.add_argument("-p", "--max-passes", type=int, default=MAX_PASSES,
                    help="max number of passes of SLIM to run")
parser.add_argument("-t", "--max-time", type=int, default=MAX_TIME,
                    help="max time for SLIM to run")
parser.add_argument("-u", "--heuristic", default=HEURISTIC,
                    choices=["kg", "ka", "kmax"], help="heuristic solver to use")
parser.add_argument("-o", "--offset", type=int, default=OFFSET,
                    help="duration after which slim takes over")
parser.add_argument("-c", "--compare", action="store_true",
                    help="run only heuristic to gather stats for comparison")
parser.add_argument("-z", "--lazy-threshold", type=float, default=LAZY_THRESHOLD,
                    help="threshold below which to not try to improve local instances")
parser.add_argument("-r", "--random-seed", type=int, default=SEED,
                    help="random seed (set 0 for no seed)")
parser.add_argument("-l", "--logging", action="store_true", help="wandb logging")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = os.path.abspath(args.file)
    LAZY_THRESHOLD = args.lazy_threshold
    if args.logging:
        wandb.init(project='twbnslim-test')
        wandb_configure(wandb, args)
        SOLUTION = Solution(logger=wandb.log)
    else:
        SOLUTION = Solution()

    # compare and exit if only comparison requested
    if args.compare:
        if args.logging:
            logger = wandb.log
        else:
            logger = lambda x: x
        monitor_blip(filepath, args.treewidth, logger, timeout=args.max_time,
                     seed=args.random_seed, solver=args.heuristic, debug=args.verbose)
        sys.exit()

    register_handler()
    try:
        slim(filepath, args.treewidth, args.budget, args.sat_timeout,
             args.max_passes, args.max_time, args.heuristic, args.offset,
             args.random_seed, args.verbose)
    except SolverInterrupt:
        print("solver interrupted")
    finally:
        if SOLUTION.value is None:
            print("no solution computed so far")
        else:
            metrics = {"start_score": SOLUTION.start_score,
                       "num_passes": SOLUTION.num_passes,
                       "num_improvements": SOLUTION.num_improvements,
                       "success_rate": SOLUTION.num_improvements/SOLUTION.num_passes,
                       "skipped": SOLUTION.skipped}
            if args.logging:
                wandb.log(metrics)
            else:
                print("final metrics:")
                pprint(metrics)
                print(f"final score: {SOLUTION.value.score:.5f}")
