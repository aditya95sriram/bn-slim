#!/bin/python3.6

# external
import os
import sys
import networkx as nx
import random
from typing import Dict, List, Tuple, Set, FrozenSet, Callable, Optional
from pprint import pprint
from time import sleep, time as now
import argparse
import signal
from collections import Counter
from operator import itemgetter
from functools import reduce
from heapq import heappop, heappush
import statistics

# internal
from blip import run_blip, BayesianNetwork, TWBayesianNetwork, monitor_blip, \
    parse_res, start_blip_proc, check_blip_proc, stop_blip_proc, write_res, activate_checkpoints
from utils import TreeDecomposition, pick, pairs, filter_read_bn, BNData, NoSolutionException,\
    get_domain_sizes, log_bag_metrics, compute_complexity_width, weight_from_domain_size, \
    compute_complexity, compute_complexities, shuffled
from berg_encoding import solve_bn, PSET_ACYC

# optional
import wandb

# check if drawing/plotting is available
CLUSTER = os.environ["WANDB_PLATFORM"] == "cluster"
if not CLUSTER:
    from networkx.drawing.nx_agraph import pygraphviz_layout
    import matplotlib.pyplot as plt

# score comparison epsilon
EPSILON = 1e-7

# default parameter values
BUDGET = 10
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
DOMAIN_SIZES: Dict[int, int] = None
DATFILE = ""
USE_COMPLEXITY_WIDTH = False
USING_COMPLEXITY_WIDTH = False
COMPLEXITY_BOUND = -1
CW_TARGET_REACHED = False
CW_TRAV_STRAT = "max-rand"  # one of max, max-min, max-rand, tw-max-rand
CW_EXP_STRAT = "min-int"  # one of max, min, min-int
SAVE_AS = ""  # pattern to use while saving networks


def find_start_bag(td: TreeDecomposition, history: Counter = None, debug=False):
    if USING_COMPLEXITY_WIDTH:  # pick bag with highest complexity
        complexities = compute_complexities(td, DOMAIN_SIZES)
        if CW_TRAV_STRAT == "max":
            bag_order = sorted(complexities, key=complexities.get, reverse=True)
        elif CW_TRAV_STRAT == "max-min":
            bag_order = sorted(complexities, key=complexities.get,
                               reverse=not CW_TARGET_REACHED)
        else:  # max-rand or tw-max-rand
            if CW_TARGET_REACHED:
                bag_order = shuffled(complexities.keys())
            else:
                bag_order = sorted(complexities, key=complexities.get, reverse=True)
        mincount = min(history[bag_id] for bag_id in bag_order)
        for bag_id in bag_order:
            if history[bag_id] == mincount:
                return bag_id
        # maxbagidx, _ = max(complexities.items(), key=itemgetter(1))
        # if history[maxbagidx] == 0:
        #     return maxbagidx
        # else:  # cw target already met, return random bag
        #     print("randomly picking bag:", end="")
        #     return pick(td.bags.keys())
    elif TRAV_STRAT == "random":  # randomly pick a bag
        return pick(td.bags.keys())
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
    queue = [(0, start_bag_id)]  # (sorting metric, bag_id)
    visited = set()
    while queue:
        _, bag_id = heappop(queue)
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
                    nbr_bag = td.bags[nbr_id]
                    if not USING_COMPLEXITY_WIDTH:
                        expansion_metric = 0  # no sorting, so set same value for all bags
                        # todo[feature]: set to random value to randomize expansion
                    else:
                        if CW_EXP_STRAT == "max":
                            expansion_metric = -compute_complexity(nbr_bag, DOMAIN_SIZES)
                        elif CW_EXP_STRAT == "min":
                            expansion_metric = compute_complexity(nbr_bag, DOMAIN_SIZES)
                        else:
                            expansion_metric = bag.intersection(nbr_bag)
                    heappush(queue, (expansion_metric, nbr_id))
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
            if USING_COMPLEXITY_WIDTH and intersection:
                clique_cw = reduce(lambda x, y: x * y, map(DOMAIN_SIZES.get, intersection))
                if clique_cw >= COMPLEXITY_BOUND:
                    if debug: print(" skipping because marker clique exceeds cw bound")
                    SOLUTION.skipped += 1
                    return None
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
           "nosolution", "restarts", "start_width")


class Solution(object):
    def __init__(self, value=None, logger: Callable = None):
        self.value: TWBayesianNetwork = value
        # other metrics to track
        self.data = dict.fromkeys(METRICS, 0)
        # mute logger for code completion hack
        self.logger = None
        # for code completion
        self.start_score = self.num_passes = self.num_improvements = \
        self.skipped = self.nosolution = self.restarts = self.start_width = 0
        # set proper value for logger now
        self.logger = logger

    def update(self, new_value):
        if self.logger is not None:
            self.logger({'score': new_value.score})
            if new_value.score - 10 > self.start_score:
                self.logger({'extremely_strong': True})
            if USE_COMPLEXITY_WIDTH:
                width = compute_complexity_width(new_value.td, DOMAIN_SIZES)
                approx_width = compute_complexity_width(new_value.td, DOMAIN_SIZES, approx=True)
                self.logger({'width': width, 'approx_width': approx_width})
        self.value = new_value


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
             history: Counter = None, width_bound: int = None, debug=False):
    td = bn.td
    if USING_COMPLEXITY_WIDTH:
        final_width_bound = weight_from_domain_size(width_bound)
    else:
        final_width_bound = width_bound
    selected, seen = find_subtree(td, budget, history, debug=False)
    history.update(selected)
    prep_tuple = prepare_subtree(bn, selected, seen, debug)
    if prep_tuple is None: return
    forced_arcs, forced_cliques, data, pset_acyc = prep_tuple
    # if debug:
    #     print("filtered data:-")
    #     pprint(data)
    old_score = bn.compute_score(seen)
    max_score = compute_max_score(data, bn)
    if RELAXED_PARENTS:
        # too strict
        # assert max_score + EPSILON >= old_score, "max score less than old score"
        assert round(max_score + EPSILON, 4) >= round(old_score, 4), "max score less than old score"
        if max_score < old_score:
            print("#### max score smaller than old score modulo epsilon")
    cur_offset = sum(bn.offsets[node] for node in seen)
    if debug: print(f"potential max: {(max_score - cur_offset)/bn.best_norm_score:.5f}", end="")
    if (max_score - cur_offset)/bn.best_norm_score <= LAZY_THRESHOLD:
        if debug: print(" skipping because lazy threshold not met")
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
    domain_sizes = DOMAIN_SIZES if USING_COMPLEXITY_WIDTH else None
    try:
        replbn = solve_bn(data, final_width_bound, bn.input_file, forced_arcs, forced_cliques,
                          pset_acyc, timeout, domain_sizes, debug)
    except NoSolutionException as err:
        SOLUTION.nosolution += 1
        print(f"no solution found by maxsat, skipping (reason: {err})")
        return
    new_score = replbn.compute_score()
    if not CLUSTER and debug:
        nx.draw(replbn.dag, pos, with_labels=True)
        plt.suptitle("replacement subdag")
        plt.show()
    if debug:
        print("new parents:-")
        pprint(replbn.parents)
    if debug: print(f"score change: {old_score:.3f} -> {new_score:.3f}")
    if USE_COMPLEXITY_WIDTH:
        old_cw = compute_complexity_width(td, DOMAIN_SIZES, include=selected)
        new_cw = compute_complexity_width(replbn.td, DOMAIN_SIZES)
        old_acw = compute_complexity_width(td, DOMAIN_SIZES, include=selected, approx=True)
        new_acw = compute_complexity_width(replbn.td, DOMAIN_SIZES, approx=True)
        print(f"old: {old_cw}|{old_acw}\tnew: {new_cw}|{new_acw}")
    # replacement criterion
    if USING_COMPLEXITY_WIDTH and old_cw > width_bound:
        if new_cw > width_bound:
            return False
    elif USING_COMPLEXITY_WIDTH and new_score == old_score and new_cw > old_cw:
        return False
    elif new_score < old_score:  # in case not using cw, then this is the only check
        return False
    print(f"score change: {old_score:.3f} -> {new_score:.3f}, replacing ...")
    td.replace(selected, forced_cliques, replbn.td)
    # update bn with new bn
    bn.replace(replbn)
    if __debug__: bn.verify(verify_treewidth=not USING_COMPLEXITY_WIDTH)
    return True


def slim(filename: str, start_treewidth: int, budget: int = BUDGET,
         sat_timeout: int = TIMEOUT, max_passes=MAX_PASSES, max_time: int = MAX_TIME,
         heuristic=HEURISTIC, offset: int = OFFSET, seed=SEED, debug=False):
    global USING_COMPLEXITY_WIDTH, COMPLEXITY_BOUND, CW_TARGET_REACHED
    start = now()
    if SAVE_AS: activate_checkpoints(lambda: SOLUTION.value, SAVE_AS)
    def elapsed(): return f"(after {now()-start:.1f} s.)"
    heur_proc = outfile = None  # placeholder
    if START_WITH is not None:
        if not os.path.isfile(START_WITH):
            print(f"specified start-with file doesn't exist, quitting {elapsed()}")
            return
        if debug: print(f"starting with {START_WITH}, not running heuristic")
        # todo[safety]: handle case when no heuristic solution so far
        bn = parse_res(filename, start_treewidth, START_WITH)
    else:
        if MIMIC:
            if debug: print("starting heuristic proc for mimicking")
            outfile = "temp-mimic.res"
            heur_proc = start_blip_proc(filename, start_treewidth, outfile=outfile,
                                        timeout=max_time, seed=seed,
                                        solver=heuristic, debug=False)
            if debug: print(f"waiting {offset}s")
            sleep(offset)
            # todo[safety]: make more robust by wrapping in try except (race condition)
            bn = parse_res(filename, start_treewidth, outfile)
        else:
            if debug: print(f"running initial heuristic for {offset}s")
            bn = run_blip(filename, start_treewidth, timeout=offset, seed=seed,
                          solver=heuristic)
    if __debug__: bn.verify()
    if USE_COMPLEXITY_WIDTH:
        start_cw = compute_complexity_width(bn.td, DOMAIN_SIZES)
        start_acw = compute_complexity_width(bn.td, DOMAIN_SIZES, approx=True)
        complexity_bound = start_cw // 2  # todo[opt]: maybe use weight as bound?
        print(f"start cw: {start_cw}\tacw:{start_acw}")
        print(f"setting complexity bound: {complexity_bound}|{weight_from_domain_size(complexity_bound)}")
        COMPLEXITY_BOUND = complexity_bound
    SOLUTION.update(bn)
    if DOMAIN_SIZES: log_bag_metrics(bn.td, DOMAIN_SIZES)
    if LAZY_THRESHOLD > 0:
        print(f"lazy threshold: {LAZY_THRESHOLD} i.e. "
              f"minimum delta required: {bn.best_norm_score*LAZY_THRESHOLD}")
    prev_score = bn.score
    print(f"Starting score: {prev_score:.5f}")
    #if debug and DATFILE: print(f"Starting LL: {eval_ll(bn, DATFILE):.6f}")
    SOLUTION.start_score = prev_score
    if USE_COMPLEXITY_WIDTH: SOLUTION.start_width = start_cw
    history = Counter(dict.fromkeys(bn.td.decomp.nodes, 0))
    if seed: random.seed(seed)
    for i in range(max_passes):
        if USE_COMPLEXITY_WIDTH:
            USING_COMPLEXITY_WIDTH = i >= 10 or CW_TRAV_STRAT != "tw-max-rand"
        width_bound = complexity_bound if USING_COMPLEXITY_WIDTH else start_treewidth
        replaced = slimpass(bn, budget, sat_timeout, history, width_bound, debug=False)
        SOLUTION.num_passes += 1
        new_score = bn.score
        if new_score > prev_score:
            print(f"*** New improvement! {new_score:.5f} {elapsed()} ***")
            prev_score = new_score
            SOLUTION.update(bn)
            SOLUTION.num_improvements += 1
        elif replaced:
            print("*** No improvement, but replacement performed ***")
            prev_score = new_score
            SOLUTION.update(bn)
        if MIMIC:
            heur_score = check_blip_proc(heur_proc, debug=False)
            if heur_score > bn.score:
                if debug: print(f"heuristic solution better {heur_score:.5f} > {bn.score:.5f}, mimicking")
                SOLUTION.restarts += 1
                newbn = parse_res(filename, start_treewidth, outfile)
                new_score = newbn.score
                assert abs(new_score >= heur_score - 1e-5), \
                    f"score exaggerated, reported: {heur_score}\tactual score: {new_score}"
                bn = newbn
                prev_score = new_score
                SOLUTION.update(bn)
                # reset history because fresh tree decomposition
                history = Counter(dict.fromkeys(bn.td.decomp.nodes, 0))
        if USE_COMPLEXITY_WIDTH:
            current_cw = compute_complexity_width(bn.td, DOMAIN_SIZES)
            if current_cw <= width_bound and not CW_TARGET_REACHED:
                if USING_COMPLEXITY_WIDTH and CW_TRAV_STRAT in ["max-min", "max-rand", "tw-max-rand"]:
                    print("*** cw target reached, flipping strategy ***")
                CW_TARGET_REACHED = True
        if debug and USE_COMPLEXITY_WIDTH: print("current cw:", current_cw)
        if debug: print(f"* Iteration {i}:\t{bn.score:.5f} {elapsed()}\n")
        if now() - start > max_time:
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
    wandb.config.method = "heur" if args.compare else f"slim_{args.heuristic}"
    wandb.config.traversal = args.traversal_strategy
    wandb.config.relaxed = int(args.relaxed_parents)
    wandb.config.mimic = int(args.mimic)
    wandb.config.datfile = args.datfile
    wandb.config.complexity_width = args.complexity_width
    wandb.config.cw_strategy = args.cw_strategy
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
                    choices=["kg", "ka", "kmax", "hc", "hcp"], help="heuristic solver to use")
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
parser.add_argument("-d", "--datfile", help="path to datfile, if omitted,"
                                            "complexity-width will not be tracked")
parser.add_argument("-w", "--complexity-width", action="store_true",
                    help="minimizing complexity width becomes main objective\n"
                         "[requires option -d|--datfile]")
parser.add_argument("--cw-strategy", default=CW_TRAV_STRAT,
                    choices=["max", "max-rand", "max-min", "tw-max-rand"],
                    help="complexity width reduction traversal strategy\n"
                         "[ignored if option -w|--complexity width not provided]")
parser.add_argument("-r", "--random-seed", type=int, default=SEED,
                    help="random seed (set 0 for no seed)")
parser.add_argument("-l", "--logging", action="store_true", help="wandb logging")
parser.add_argument("--project-name", default="twbnslim-test", help="wandb project name")
parser.add_argument("--start-with", default=None,
                    help="optionally skip running heuristic and start with this solution")
parser.add_argument("--save-as", default="", help="filename to save final network as")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = os.path.abspath(args.file)
    LAZY_THRESHOLD = args.lazy_threshold
    RELAXED_PARENTS = args.relaxed_parents
    MIMIC = args.mimic
    DATFILE = args.datfile
    DOMAIN_SIZES = get_domain_sizes(args.datfile) if args.datfile else None
    if args.complexity_width:
        USE_COMPLEXITY_WIDTH = True
        if DOMAIN_SIZES is None:
            parser.error("--complexity-width requires --datfile")
    CW_TRAV_STRAT = args.cw_strategy
    if args.budget <= args.treewidth and not args.compare:
        print("budget smaller than treewidth bound, quitting")
        sys.exit()
    print("not"*__debug__, "running optimized")
    if args.start_with is not None:
        START_WITH = os.path.abspath(args.start_with)
    TRAV_STRAT = args.traversal_strategy
    SAVE_AS = os.path.abspath(args.save_as) if args.save_as else ""
    logger = lambda x: x  # no op
    # logger = lambda x: print(f"log: {x}")  # local log
    if args.logging:
        wandb.init(project=args.project_name)
        wandb_configure(wandb, args)
        logger = wandb.log
    SOLUTION = Solution(logger=logger)

    # if comparison requested, compare then exit
    if args.compare:
        outfile = SAVE_AS or "temp.res"
        logger = lambda x: print(f"log: {x}")  # local log
        monitor_blip(filepath, args.treewidth, logger, timeout=args.max_time,
                     seed=args.random_seed, solver=args.heuristic,
                     domain_sizes=DOMAIN_SIZES, outfile=outfile, save_as=SAVE_AS,
                     debug=args.verbose)
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
            # verify final bn (not required if optimizing complexity width)
            # todo[req]: complexity width separate verification
            SOLUTION.value.verify(verify_treewidth=not USE_COMPLEXITY_WIDTH)
            print("verified")
            if SAVE_AS:
                save_fname = SAVE_AS.replace(".res", "-final.res")
                write_res(SOLUTION.value, save_fname, write_elim_order=True)
                obn = SOLUTION.value
                bn2 = parse_res(filepath, args.treewidth, save_fname)
                print("saving final network to", save_fname, "as final checkpoint")
            success_rate = SOLUTION.num_improvements / (SOLUTION.num_passes - SOLUTION.skipped)
            if args.logging:
                wandb.log({"success_rate": success_rate})
                if SOLUTION.num_improvements > 0:
                    wandb.log({"improved": True})
            else:
                print("final metrics:")
                pprint(SOLUTION.data)
                print(f"success_rate: {success_rate:.2%}")
                print(f"final score: {SOLUTION.value.score:.5f}")
                if DOMAIN_SIZES:
                    log_bag_metrics(SOLUTION.value.td, DOMAIN_SIZES, append=True)
                    print("complexity-width:", compute_complexity_width(SOLUTION.value.td,
                                                                        DOMAIN_SIZES))
