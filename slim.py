#!/bin/python3.6

# external
import os
import sys
import networkx as nx
import random
from typing import Dict, List, Tuple, Set, FrozenSet, Callable
from pprint import pprint
from time import sleep, time as now
import argparse
import signal
from collections import Counter
import statistics

# internal
from blip import run_blip, BayesianNetwork, TWBayesianNetwork, monitor_blip, \
    parse_res, start_blip_proc, check_blip_proc, stop_blip_proc
from utils import TreeDecomposition, pick, pairs, filter_read_bn, BNData, NoSolutionException
from berg_encoding import solve_bn, PSET_ACYC

# optional
import wandb

# check if drawing/plotting is available
CLUSTER = os.environ["WANDB_PLATFORM"] == "cluster"
if not CLUSTER:
    from networkx.drawing.nx_agraph import pygraphviz_layout
    import matplotlib.pyplot as plt

# score comparison epsilon
EPSILON = 1e-10

# default parameter values
BUDGET = 7
TIMEOUT = 5
MAX_PASSES = 100
MAX_TIME = 1800
HEURISTIC = 'kmax'
OFFSET = 2
SEED = 9
LAZY_THRESHOLD = 0.0
START_WITH = None
RELAXED_PARENTS = False
TRAV_STRAT = "random"
MIMIC = False


def find_start_bag(td: TreeDecomposition, history: Counter = None, debug=False):
    if TRAV_STRAT == "random":
        return pick(td.bags.keys())  # randomly pick a bag
    else:  # pick bag with least count and earliest in order
        if TRAV_STRAT == "post":
            trav_order = list(nx.dfs_postorder_nodes(td.decomp))
        elif TRAV_STRAT == "pre":
            trav_order = list(nx.dfs_postorder_nodes(td.decomp))
        else:
            raise ValueError(f"invalid traversal strategy {TRAV_STRAT}")
        mincount = min(history.values())
        # todo[opt]: avoid retraversing every time
        for bag_id in trav_order:
            if history[bag_id] == mincount:
                return bag_id


def find_subtree(td: TreeDecomposition, budget: int, history: Counter = None,
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
    subdag = nx.subgraph_view(dag, lambda x: True,
                              lambda x, y: not ((x in seen) and (y in seen)))
    forced_arcs = []
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

    data, pset_acyc = get_data_for_subtree(bn, boundary_nodes, seen, forced_arcs)

    return forced_arcs, forced_cliques, data, pset_acyc


def get_data_for_subtree(bn: TWBayesianNetwork, boundary_nodes: set, seen: set,
                         forced_arcs: List[Tuple[int, int]]) -> Tuple[BNData, PSET_ACYC]:
    # store downstream relations in a graph
    downstream_graph = nx.DiGraph()
    downstream_graph.add_nodes_from(boundary_nodes)
    for node in boundary_nodes:
        for _, successors in nx.bfs_successors(bn.dag, node):
            downstream_graph.add_edges_from((node, succ) for succ in successors
                                      if succ not in seen)
    #downstream.remove_nodes_from(seen - boundary_nodes)  # ignore inner red nodes
    assert seen.intersection(downstream_graph.nodes).issubset(boundary_nodes), \
        "downstream connectivity graph contains inner nodes"
    downstream_graph.add_edges_from(forced_arcs)

    downstream = set()
    if not RELAXED_PARENTS:
        downstream = set(downstream_graph.nodes())-seen

    # construct score function data for local instance
    data: BNData = {node: dict() for node in seen}
    pset_acyc: PSET_ACYC = dict()
    for node, psets in filter_read_bn(bn.input_file, seen).items():
        if node in boundary_nodes:
            if RELAXED_PARENTS:
                downstream = set(downstream_graph.successors(node))
            for pset, score in psets.items():
                if pset.intersection(downstream):
                    continue  # reject because pset contains downstream verts
                pset_in = pset.intersection(seen)
                if len(pset_in) < len(pset):
                    # not all internal vertices, so check if bag exists
                    # todo[opt]: exclude selected while searching for bag
                    bag_id = bn.td.bag_containing(pset | {node})
                    if bag_id == -1:
                        continue  # reject because required bag doesnt already exist in td
                    rem_parents = pset - pset_in
                    req_acyc = set()
                    for parent in rem_parents:
                        if parent in downstream_graph:
                            req_acyc.update(downstream_graph.predecessors(parent))
                    pset_acyc[(node, pset)] = req_acyc
                data[node][pset] = score
        else:  # internal node
            for pset, score in psets.items():
                # internal vertices are not allowed outside parents
                if pset.issubset(seen):
                    data[node][pset] = score
    return data, pset_acyc


def compute_max_score(data: BNData, bn: BayesianNetwork) -> float:
    max_score = 0
    for node in data:
        max_score += max(data[node].values()) + bn.offsets[node]
    return max_score


METRICS = ("start_score", "num_passes", "num_improvements", "skipped",
           "nosolution", "restarts")

class Solution(object):
    def __init__(self, value=None, logger: Callable = None):
        self.value: TWBayesianNetwork = value
        # other metrics to track
        self.data = dict.fromkeys(METRICS, 0)
        # mute logger for code completion hack
        self.logger = None
        # for code completion
        self.start_score = self.num_passes = self.num_improvements = \
        self.skipped = self.nosolution = self.restarts = 0
        # set proper value for logger now
        self.logger = logger

    def update(self, new_value):
        if self.logger is not None:
            self.logger({'score': new_value.score})
        self.value = new_value


def log_potential(old, new, max, offset, best):
    if CLUSTER: return
    old = (old - offset)/best
    new = (new - offset)/best
    max = (max - offset)/best
    with open("potentials.csv", "a") as f:
        f.write(f"{old:.5f},{new:.5f},{max:.5f}\n")


def _getter_factory(metric: str):
    def getter(self: Solution):
        return self.data[metric]
    return getter


def _setter_factory(metric: str):
    def setter(self: Solution, val):
        self.data[metric] = val
        if self.logger is not None:
            self.logger({metric: val})
    return setter


for metric in METRICS:
    setattr(Solution, metric,
            property(_getter_factory(metric), _setter_factory(metric)))


SOLUTION = Solution()  # placeholder global solution variable


def slimpass(bn: TWBayesianNetwork, budget: int = BUDGET, timeout: int = TIMEOUT,
             history: Counter = None, debug=False):
    td = bn.td
    tw = td.width
    selected, seen = find_subtree(td, budget, history, debug=False)
    history.update(selected)
    forced_arcs, forced_cliques, data, pset_acyc = prepare_subtree(bn, selected, seen, debug)
    # if debug:
    #     print("filtered data:-")
    #     pprint(data)
    old_score = bn.compute_score(seen)
    max_score = compute_max_score(data, bn)
    if RELAXED_PARENTS:
        assert max_score + EPSILON >= old_score, "max score less than old score"
        if max_score < old_score:
            print("#### max score smaller than old score modulo epsilon")
    cur_offset = sum(bn.offsets[node] for node in seen)
    if debug: print(f"potential max: {(max_score - cur_offset)/bn.best_norm_score:.5f}", end="")
    if (max_score - cur_offset)/bn.best_norm_score <= LAZY_THRESHOLD:
        if debug: print(" skipping ####")
        SOLUTION.skipped += 1
        return
    pos = dict()  # placeholder layout
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
    try:
        replbn = solve_bn(data, tw, bn.input_file, forced_arcs, forced_cliques,
                          pset_acyc, timeout, debug)
    except NoSolutionException as err:
        SOLUTION.nosolution += 1
        print(f"no solution found by maxsat, skipping (reason: {err})")
        return
    new_score = replbn.compute_score()
    log_potential(old_score, new_score, max_score, cur_offset, bn.best_norm_score)
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
    if new_score >= old_score:  # replacement condition
        td.replace(selected, forced_cliques, replbn.td)
        # update bn with new bn
        bn.replace(replbn)
        if __debug__: bn.verify()


def slim(filename: str, treewidth: int, budget: int = BUDGET,
         sat_timeout: int = TIMEOUT, max_passes=MAX_PASSES, max_time: int = MAX_TIME,
         heuristic=HEURISTIC, offset: int = OFFSET, seed=SEED, debug=False):
    start = now()
    def elapsed(): return f"(after {now()-start:.1f} s.)"
    heur_proc = outfile = None  # placeholder
    if START_WITH is not None:
        if debug: print(f"starting with {START_WITH}, not running heuristic")
        # todo[safety]: handle case when no heuristic solution so far
        bn = parse_res(filename, treewidth, START_WITH)
    else:
        if MIMIC:
            if debug: print("starting heuristic proc for mimicking")
            outfile = "temp-mimic.res"
            heur_proc = start_blip_proc(filename, treewidth, outfile=outfile,
                                        timeout=max_time, seed=seed,
                                        solver=heuristic, debug=False)
            if debug: print(f"waiting {offset}s")
            sleep(offset)
            # todo[safety]: make more robust by wrapping in try except (race condition)
            bn = parse_res(filename, treewidth, outfile)
        else:
            if debug: print(f"running initial heuristic for {offset}s")
            bn = run_blip(filename, treewidth, timeout=offset, seed=seed,
                          solver=heuristic)
    if __debug__: bn.verify()
    SOLUTION.update(bn)
    if LAZY_THRESHOLD > 0:
        print(f"lazy threshold: {LAZY_THRESHOLD} i.e. "
              f"minimum delta required: {bn.best_norm_score*LAZY_THRESHOLD}")
    prev_score = bn.score
    print(f"Starting score: {prev_score:.5f}")
    SOLUTION.start_score = prev_score
    history = Counter(dict.fromkeys(bn.td.decomp.nodes, 0))
    if seed: random.seed(seed)
    for i in range(max_passes):
        slimpass(bn, budget, sat_timeout, history, debug=False)
        SOLUTION.num_passes += 1
        new_score = bn.score
        if new_score > prev_score:
            print(f"*** New improvement! {new_score:.5f} {elapsed()} ***")
            prev_score = new_score
            SOLUTION.update(bn)
            SOLUTION.num_improvements += 1
        if MIMIC:
            heur_score = check_blip_proc(heur_proc, debug=False)
            if heur_score > bn.score:
                if debug: print(f"heuristic solution better {heur_score:.5f} > {bn.score:.5f}, mimicking")
                SOLUTION.restarts += 1
                newbn = parse_res(filename, treewidth, outfile)
                new_score = newbn.score
                assert abs(new_score >= heur_score - 1e-5), \
                    f"score exaggerated, reported: {heur_score}\tactual score: {new_score}"
                bn = newbn
                prev_score = new_score
                SOLUTION.update(bn)
                # reset history because fresh tree decomposition
                history = Counter(dict.fromkeys(bn.td.decomp.nodes, 0))
        if debug: print(f"* Iteration {i}:\t{bn.score:.5f} {elapsed()}")
        if now()-start > max_time:
            if debug: print("time limit exceeded, quitting")
            break
    if MIMIC:
        if debug: print("stopping heur proc")
        stop_blip_proc(heur_proc)
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
    wandb.config.instance = basename
    wandb.config.treewidth = args.treewidth
    wandb.config.budget = args.budget
    wandb.config.sat_timeout = args.sat_timeout
    wandb.config.heuristic = args.heuristic
    wandb.config.offset = args.offset
    wandb.config.threshold = args.lazy_threshold
    wandb.config.seed = args.random_seed
    wandb.config.method = 'heur' if args.compare else 'slim'
    wandb.config.traversal = args.traversal_strategy
    wandb.config.relaxed = int(args.relaxed_parents)
    wandb.config.mimic = int(args.mimic)
    # process config
    wandb.config.platform = "cluster" if CLUSTER else "workstation"
    wandb.config.jobid = os.environ.get("MY_JOB_ID", -1)
    wandb.config.taskid = os.environ.get("MY_TASK_ID", -1)


# noinspection PyTypeChecker
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
parser.add_argument("-x", "--relaxed-parents", action="store_true",
                    help="relax allowed parent sets for maxsat encoding\n"
                         "[warning: use at own risk, could terminate abruptly]")
parser.add_argument("-y", "--traversal-strategy", default=TRAV_STRAT,
                    choices=["random", "post", "pre"],
                    help="td traversal strategy")
parser.add_argument("-m", "--mimic", action="store_true",
                    help="mimic heuristic if it outperforms")
parser.add_argument("-r", "--random-seed", type=int, default=SEED,
                    help="random seed (set 0 for no seed)")
parser.add_argument("-l", "--logging", action="store_true", help="wandb logging")
parser.add_argument("--project-name", default="twbnslim-test", help="wandb project name")
parser.add_argument("--start-with", default=None,
                    help="optionally skip heuristic and start with this solution")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = os.path.abspath(args.file)
    LAZY_THRESHOLD = args.lazy_threshold
    RELAXED_PARENTS = args.relaxed_parents
    MIMIC = args.mimic
    if args.budget <= args.treewidth:
        print("budget smaller than treewidth bound, quitting")
        sys.exit()
    print("not"*__debug__, "running optimized")
    if args.start_with is not None:
        START_WITH = os.path.abspath(args.start_with)
    TRAV_STRAT = args.traversal_strategy
    logger = lambda x: x  # no op
    # logger = lambda x: print(f"log: {x}")  # local log
    if args.logging:
        wandb.init(project=args.project_name)
        wandb_configure(wandb, args)
        logger = wandb.log
    SOLUTION = Solution(logger=logger)

    # compare and exit if only comparison requested
    if args.compare:
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
            SOLUTION.value.verify()  # verify final bn
            print("verified")
            success_rate = SOLUTION.num_improvements / (SOLUTION.num_passes - SOLUTION.skipped)
            if args.logging:
                wandb.log({"success_rate": success_rate})
            else:
                print("final metrics:")
                pprint(SOLUTION.data)
                print(f"success_rate: {success_rate:.2%}")
                print(f"final score: {SOLUTION.value.score:.5f}")