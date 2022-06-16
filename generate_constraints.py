#!/usr/bin/python3
import os.path
import sys
import networkx as nx
import random
import argparse
import re
from math import ceil, floor

from tqdm import tqdm

from utils import read_jkl

PROB_PROFILE = "default"
INSTANCE = "none"
CUSTOM = False


def debug(*args, **kwargs):
    print(f"[gencon|{INSTANCE}]", *args, **kwargs, file=sys.stderr)


def tqdebug(msg):
    tqdm.write(f"[gencon|{INSTANCE}] {msg}")


def net2digraph(net_path):
    digraph = nx.DiGraph()
    node_patn = re.compile(r"node\s+(?P<node>\S+)")
    par_patn  = re.compile(r"potential\s+\(\s+(?P<child>\S+)\s+(?P<parents>\|.+)?\)")
    with open(net_path) as f:
        for line in f:
            mo = node_patn.search(line)
            if mo is not None:
                label = mo.group("node")
                nodeid = len(digraph)
                digraph.add_node(label, nodeid=nodeid)
            mo = par_patn.search(line)
            if mo is not None:
                child = mo.group("child")
                parents = mo.group("parents")
                if parents is None:
                    parents = []
                else:
                    parents = parents.lstrip("|").split()
                for parent in parents:
                    digraph.add_edge(parent, child)
    return digraph


def extract_nodemap(digraph):
    return dict(digraph.nodes(data="nodeid"))


def randcons(digraph, num_constraints):
    nodes = tuple(digraph.nodes)
    succ = {n: set(digraph.successors(n)) for n in nodes}
    desc = {n: nx.descendants(digraph, n) for n in nodes}
    checker = {"arc": succ, "anc": desc}
    constraints = []
    for _ in range(num_constraints):
        p, q = random.sample(nodes, 2)
        typ = random.choice(["anc", "arc"])
        exists = q in checker[typ][p]
        if exists:
            qualifier = random.choice(["pos", "und"])
        else:
            if PROB_PROFILE == "default":
                qualifier = "neg"
            else:
                reverse = p in checker[typ][q]
                qualifier = "und" if reverse else "neg"
        yield (f"{qualifier}{typ}", p, q)
        #constraints.append((f"{qualifier}{typ}", p, q))
    #return constraints


def sample_percentage(population, percentage: float):
    required = int(ceil(len(population) * percentage))
    return random.sample(population, required)


def sample_count(population, count: int):
    return random.sample(population, count)


class LivanBeekGenerator:

    def __init__(self, digraph, score_function=None):
        self.digraph = digraph
        self.num_nodes = digraph.number_of_nodes()
        self.num_arcs = digraph.number_of_edges()
        self.present_arcs = self.all_present_arcs()
        self.absent_arcs = self.all_absent_arcs()
        self.present_ancestries, self.absent_ancestries = self.all_ancestries()

        if score_function is not None:
            self.filter_impossible(score_function)

    def all_present_arcs(self):
        return list(self.digraph.edges())

    def all_absent_arcs(self):
        absent = []
        for i in self.digraph.nodes:
            for j in self.digraph.nodes:
                if i == j: continue
                if not self.digraph.has_edge(i, j):
                    absent.append((i,j))
        return absent

    def all_ancestries(self):
        ancestry_adj = {node: set(nx.descendants(self.digraph, node))
                        for node in self.digraph}
        present = []
        absent = []
        for i in self.digraph.nodes:
            for j in self.digraph.nodes:
                if i == j: continue
                if j in ancestry_adj[i]:
                    present.append((i, j))
                else:
                    absent.append((i, j))
        return present, absent

    def generate_arcs(self, percentage: float):
        result = dict()
        result["negarc"] = sample_percentage(self.absent_arcs, percentage)
        result["posarc"] = []
        result["undarc"] = []
        for arc in sample_percentage(self.present_arcs, percentage):
            if random.choice([True, False]):
                result["posarc"].append(arc)
            else:
                result["undarc"].append(arc)
        return result

    def generate_ancestries(self, percentage: float):
        result = dict()
        result["neganc"] = sample_percentage(self.absent_ancestries, percentage)
        result["posanc"] = []
        result["undanc"] = []
        for ancestry in sample_percentage(self.present_ancestries, percentage):
            if random.choice([True, False]):
                result["posanc"].append(ancestry)
            else:
                result["undanc"].append(ancestry)
        return result

    def filter_impossible(self, score_function):
        # set up checker
        nodemap = extract_nodemap(self.digraph)
        checker = ConstraintPossibilityChecker(score_function, nodemap)

        # filter out impossible posarcs
        new_arcs = [arc for arc in self.present_arcs if checker.posarc_chcker(*arc)]

        # filter out impossible posancs
        new_ancs = [anc for anc in self.present_ancestries if checker.posanc_checker(*anc)]

        filtered_arcs = len(self.present_arcs) - len(new_arcs)
        filtered_ancs = len(self.present_ancestries) - len(new_ancs)
        if filtered_arcs + filtered_ancs > 0:
            debug(f"filtered out {filtered_arcs} arcs and "
                  f"{filtered_ancs} ancestries from universe")
        # else:
        # debug("nothing to filter out")

        self.present_arcs = new_arcs
        self.present_ancestries = new_ancs

    def generate(self, percentage: float, seed=None):
        if seed is not None: random.seed(seed)
        arcs = self.generate_arcs(percentage)
        ancestries = self.generate_ancestries(percentage)
        return dict(**arcs, **ancestries)  # union of both dicts


class CustomGenerator(LivanBeekGenerator):
    """
    generate fixed # constraints (i.e. absolute count),
    no undirected ancestry constraints
    """

    def adjust_count(self, count: int) -> int:
        total_arcs = len(self.present_arcs)
        half_total = int(floor(total_arcs / 2))
        if count > half_total:
            tqdebug(f"adjusted count: {count} -> {half_total}")
            count = half_total
        return count

    def generate_arcs(self, count: int):
        result = dict()
        result["negarc"] = sample_count(self.absent_arcs, count)
        result["posarc"] = []
        result["undarc"] = []
        posarc_done = undarc_done = False
        for arc in sample_count(self.present_arcs, 2*count):
            if (random.choice([True, False]) and not posarc_done) or undarc_done:
                result["posarc"].append(arc)
            else:
                result["undarc"].append(arc)
            if len(result["undarc"]) == count: undarc_done = True
            if len(result["posarc"]) == count: posarc_done = True
        return result

    def generate_ancestries(self, count: int):
        result = dict()
        result["neganc"] = sample_count(self.absent_ancestries, count)
        result["posanc"] = sample_count(self.present_ancestries, count)
        result["undanc"] = []
        return result

    def generate(self, count: int, seed=None):
        count = self.adjust_count(count)
        if seed is not None: random.seed(seed)
        arcs = self.generate_arcs(count)
        ancestries = self.generate_ancestries(count)
        return dict(**arcs, **ancestries)  # union of both dicts


def test(digraph, num_constraints, num_tests):
    from collections import Counter
    counts = Counter()
    for i in range(num_tests):
        count = Counter()
        random.seed(i)
        for constraint in randcons(digraph, num_constraints):
            count[constraint[0]] += 1
        counts.update(count)
    for key in sorted(counts.keys()):
        percentage = counts[key]/(num_constraints*num_tests)
        print(f"{percentage:2.2%}", end=",")
    print()


class ConstraintPossibilityChecker:

    def __init__(self, score_function, nodemap, debug=False):
        self.score_function = score_function
        self.nodemap = nodemap
        self.debug = debug

        score_func_dag = nx.DiGraph()
        score_func_dag.add_nodes_from(self.score_function)
        for child, psets in score_function.items():
            all_parents = set()
            for pset in psets:
                all_parents.update(pset)
            score_func_dag.add_edges_from((parent, child) for parent in all_parents)

        self.score_func_parents = {node: set(score_func_dag.predecessors(node))
                                   for node in score_func_dag}
        self.score_func_ancestries = {node: set(nx.ancestors(score_func_dag, node))
                                      for node in score_func_dag}

    def map(self, *args):
        return map(self.nodemap.get, args)

    def posarc_chcker(self, _u, _v):
        u, v = self.map(_u, _v)
        if u in self.score_func_parents[v]:
            return True
        if self.debug: debug(f"{u} -> {v} not possible")
        return False

    def posanc_checker(self, _u, _v):
        u, v = self.map(_u, _v)
        if u in self.score_func_ancestries[v]:
            return True
        if self.debug: debug(f"{u} ~> {v} not possible")
        return False

    @staticmethod
    def undirected(checker, u, v):
        return checker(u, v) or checker(v, u)

    def check_possible(self, constraints, early_exit=True):
        checkers = {"posarc": self.posarc_chcker,
                    "posanc": self.posanc_checker}
        failed = {typ: 0 for typ in constraints}
        for typ in constraints:
            if typ.startswith("pos"):
                checker = checkers[typ]
            elif typ.startswith("und"):
                basetyp = typ.replace("und", "pos")
                checker = lambda u, v: self.undirected(checkers[basetyp], u, v)
            else:  # negative constraints always possible, no check needed
                continue
            for u, v in constraints[typ]:
                if not checker(u, v):
                    if early_exit:
                        return False
                    else:
                        failed[typ] += 1
        if not early_exit:
            if sum(failed.values()):
                debug("failed possibility check:")
                for typ, count in failed.items():
                    debug(f"{typ}: {count}")
                return False
        return True


parser = argparse.ArgumentParser(description="generate constraints for given instance")
parser.add_argument("netpath", help="path to .net file")
parser.add_argument("-seed", type=int, nargs='*',
                    help="random seed (accepts multiple seeds)")
parser.add_argument("-num-con", type=int, nargs="*",
                    help="number of constraints (accepts multiple numbers)")
parser.add_argument("-percent-con", type=float, nargs='*',
                    help="number of constraints as a percentage of number of nodes"
                         "(accepts multiple percentages)")
parser.add_argument("-test", type=int, default=0,
                    help="count percentage of pos/und constraints over TEST runs")
parser.add_argument("-prob-profile", choices=['default', 'adjusted', 'livanbeek', 'custom'],
                    default=PROB_PROFILE)
parser.add_argument("-score-func", type=str,
                    help="path to score function cache (.jkl file)")
parser.add_argument("-ensure-possible", action="store_true",
                    help="ensure generated constraints feasible via score function cache"
                         "(requires -score-func argument)")
parser.add_argument("-out-directory", type=str,
                    help="directory to store output constraint files")
parser.add_argument("-verbose", action="store_true", help="verbose debug output")


if __name__ == "__main__":
    args = parser.parse_args()

    netpath = args.netpath
    digraph = net2digraph(netpath)
    nodemap = extract_nodemap(digraph)

    instance, ext = os.path.splitext(os.path.basename(netpath))
    INSTANCE = instance.split("-")[0]

    if args.seed is None and args.test is None:
        own_seed = random.randint(1, 1000000)
        debug("seed:", own_seed)
        seeds = [own_seed]
    else:
        seeds = args.seed

    PROB_PROFILE = args.prob_profile

    score_function = None
    if args.score_func is not None:
        score_function = read_jkl(args.score_func)

    multiple_seeds = len(seeds) > 1
    multiple_counts = args.num_con and len(args.num_con) > 1
    multiple_percents = args.percent_con and len(args.percent_con) > 1
    multiple_runs = multiple_seeds or multiple_counts or multiple_percents
    if PROB_PROFILE in ["livanbeek", "custom"]:
        if PROB_PROFILE == "custom": CUSTOM = True
        generator = CustomGenerator if CUSTOM else LivanBeekGenerator
        debug("setting up generator...")
        if args.ensure_possible:
            if score_function is None:
                parser.error("for -ensure-possible, -score-func must be supplied")
            gen = generator(digraph, score_function)
        else:
            gen = generator(digraph)
        if CUSTOM:
            if args.num_con is None:
                parser.error("for -prob-profile custom, -num-con must be supplied")
        else:
            if args.percent_con is None:
                parser.error("for -prob-profile livanbeek, -percent-con must be supplied")

        if score_function is not None and args.ensure_possible:
            checker = ConstraintPossibilityChecker(score_function, nodemap, debug=False)
        else:
            checker = None

        progressbar = tqdm if multiple_runs else lambda a: a
        consets = args.num_con if CUSTOM else args.percent_con
        for conset_id in progressbar(consets):
            for seed in progressbar(seeds):
                if CUSTOM:
                    count = conset_id
                    fname = f"{INSTANCE}-{count}-{seed}.con"
                    constraints = gen.generate(count, seed=seed)
                else:
                    percent = conset_id
                    fname = f"{INSTANCE}-{int(percent * 100)}-{seed}.con"
                    constraints = gen.generate(percent, seed=seed)

                if checker is not None:
                    result = checker.check_possible(constraints, early_exit=False)
                    assert result, "constraint possibility check failed"

                if not args.out_directory:
                    for typ, cons in constraints.items():
                        print(f"{typ}: {len(cons)}")
                else:
                    outpath = os.path.join(args.out_directory, fname)
                    if not multiple_runs: debug(f"writing out to {outpath}")
                    # debug(f"writing out to {outpath}")
                    with open(outpath, 'w') as outfile:
                        for typ in constraints:
                            for u, v in constraints[typ]:
                                outfile.write(f"{typ} {nodemap[u]} {nodemap[v]}\n")
    else:
        num_constraints = args.num_con
        for percent_con in args.percent_con:
            for seed in seeds:
                if percent_con >= 0:
                    num_constraints = percent_con * len(digraph)

                random.seed(seed)

                if args.test > 0:
                    test(digraph, num_constraints, args.test)
                else:
                    for constraints in randcons(digraph, num_constraints):
                        print(*constraints)

