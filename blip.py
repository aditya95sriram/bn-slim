#!/bin/python3.6

import subprocess
import os
import networkx as nx
import random
from typing import Optional, Callable, Dict, List
import re
import signal
from glob import glob
from time import sleep
from itertools import islice

from utils import filled_in, TreeDecomposition, pairs, stream_bn, get_bn_stats, \
    filter_read_bn, compute_complexity_width, read_jkl, write_jkl, get_domain_sizes, \
    CWDecomposition, Constraints, count_satisfied_constraints, total_satisfied_constraints

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import pygraphviz_layout

SOLVER_DIR = "../solvers"
SCORE_PATN = re.compile(r"New improvement! (?P<score>[\d.-]+) \(after (?P<time>[\d.-]+) s.\)")
CHECKPOINT_MINUTES = 30  # in minutes
CHECKPOINT_INTERVAL = int(CHECKPOINT_MINUTES*60)  # in seconds
CHECKPOINT_RETRY_INTERVAL = 5  # in seconds

# not that crucial, if you consider the stdout updates of the heuristic as mere
# triggers for reading new (not necessarily next) bn from .res file
PARSE_RETRY_INTERVAL = 0.01  # in seconds


class BayesianNetwork(object):
    def __init__(self, input_file, data=None, parents=None):
        self.parents = dict() if parents is None else parents
        self.input_file = input_file
        self.data = data
        self.sum_scores, self.best_score, self.offsets = get_bn_stats(self.input_file)
        self.best_norm_score = self.best_score - sum(self.offsets.values())
        self._score = None
        self._dag = None

    def _clear_cached(self):
        self._score = self._dag = None

    def compute_all_scores(self, subset: set = None) -> Dict[int, float]:
        if self.data is None:
            if subset is None:
                data = stream_bn(self.input_file, normalize=True)
            else:
                data = filter_read_bn(self.input_file, subset, normalize=True).items()
        else:
            data = self.data.items()  # data is assumed to be normalized
        scores = dict()
        for node, psets in data:
            if node in self.parents:  # could be a bn with a subset of nodes
                if subset is None or node in subset:
                    scores[node] = psets[self.parents[node]] + self.offsets[node]
        return scores

    def compute_score(self, subset: set=None) -> float:
        """recompute score based on parents sets"""
        scores = self.compute_all_scores(subset)
        return sum(scores.values())

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

    def draw(self, subset=None):
        if subset is None:
            dag = self.dag
        else:
            dag = self.dag.subgraph(subset)
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
        self.elim_order: List[int] = elim_order
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

    def verify(self, verify_treewidth=True):
        super().verify()
        if verify_treewidth:
            self.td.verify(graph=self.get_moralized())

    def get_triangulated(self, elim_order=None):
        if elim_order is None: elim_order = self.elim_order
        assert elim_order is not None, "elim order not specified"
        moral = self.get_moralized().subgraph(elim_order)
        triangulated, max_degree = filled_in(moral, elim_order)
        assert max_degree <= self.tw, f"tw {self.tw} < {max_degree} for elim order {elim_order}"
        return triangulated


class CWBayesianNetwork(TWBayesianNetwork):
    def __init__(self, input_file, cwidth, datfile, elim_order=None, td=None, *args, **kwargs):
        super().__init__(input_file, *args, **kwargs)
        self.tw = cwidth
        self.elim_order: List[int] = elim_order
        self._td: Optional[TreeDecomposition] = td
        self.domain_sizes = get_domain_sizes(datfile)

    def done(self):
        if self.elim_order is None:
            self.elim_order = list(nx.topological_sort(self.dag))[::-1]
        self._td = CWDecomposition(self.get_moralized(), self.elim_order, self.tw, self.domain_sizes)


class ConstrainedBayesianNetwork(TWBayesianNetwork):
    def __init__(self, constraints: Constraints, input_file, tw=0,
                 elim_order=None, td=None, *args, **kwargs):
        super().__init__(input_file, tw, elim_order, td, *args, **kwargs)
        self.constraints = constraints
        self.total_constraints = sum(map(len, constraints.values()))

    @staticmethod
    def fromTwBayesianNetwork(twbn: TWBayesianNetwork, constraints: Constraints):
        return ConstrainedBayesianNetwork(constraints, twbn.input_file, twbn.tw,
                                          twbn.elim_order, twbn.td,
                                          parents=twbn.parents)

    def count_satisfied_constraints(self):
        return count_satisfied_constraints(self, self.constraints)

    def total_satisfied_constraints(self):
        return total_satisfied_constraints(self, self.constraints)

    def find_overlapping_relatives(self, node, seen: set, ancestors=True):
        """find ancestors/descendants of node overlapping set seen"""
        overlap = set()
        queue = [node]
        visited = set()
        relative = self.dag.predecessors if ancestors else self.dag.successors
        while queue:
            cur = queue.pop(0)
            visited.add(cur)
            for nbr in relative(cur):
                if nbr in visited: continue
                if nbr in seen:
                    if ancestors:  # if ancestor add internal node
                        overlap.add(nbr)
                    else:          # else add parent of internal node
                        overlap.add(cur)
                else:
                    queue.append(nbr)
        return overlap

    def find_random_overlapping_descendant(self, node, seen: set):
        overlap = self.find_overlapping_relatives(node, seen, ancestors=False)
        if not overlap: return None
        overlap = list(overlap)
        random.shuffle(overlap)
        chosen = overlap.pop()
        nbrs = list(self.dag.successors(chosen))
        random.shuffle(nbrs)
        rand_nbr = nbrs.pop()
        return chosen, rand_nbr

    def find_paths(self, src, dest, k=1):
        """return upto k paths between src and dest"""
        paths = nx.shortest_simple_paths(self.dag, src, dest)
        return islice(paths, k)


def inject_tuples(basejkl, extra_tuples, destname, strong_injection=False):
    src = read_jkl(basejkl, normalize=False)
    inject_count = 0
    for node, pset in extra_tuples.items():
        parents, score = pset
        if strong_injection or parents not in src[node]:
            src[node][parents] = score
            inject_count += 1
    write_jkl(src, destname)


def parse_res(filename: str, treewidth: int, outfile: str, cwidth=-1,
              add_extra_tuples=False, augfile: str = "augmented.jkl",
              datfile=None, retry=True, debug=False) -> TWBayesianNetwork:
    """
    Parse a .res file containing a solution BN. Optionally merge the parent set
    tuples from the jkl file `filename` and the the res file `outfile` and
    save as a temporary file `augfile` (only when `add_extra_tuples` is True)

    :param filename: input jkl file
    :param treewidth: treewidth bound
    :param outfile: res file containing BN
    :param cwidth: cwidth bound, use -1 to ignore and simply parse a TWBN
    :param add_extra_tuples: whether to inject tuples from outfile into filename
    :param augfile: name of temporary file containing merged set of tuples,
                    (only applicable if add_extra_tuples is True)
    :param datfile: path to data file (with header rows)
    :param retry: keep retrying until file is non-empty
    :param debug: debugging
    :return: parsed BN as a TWBayesianNetwork
    """
    elim_order = None
    tuples = []
    extra_tuples = dict()
    score = None
    # keep retrying until file is non-empty
    while retry and os.path.getsize(outfile) == 0:
        sleep(PARSE_RETRY_INTERVAL)
    with open(outfile) as out:
        for line in out:
            if not line.strip():
                continue
            elif line.startswith("Score:"):
                score = float(line.split()[1].strip())
                break
            elif line.startswith("elim-order:"):
                elim_order = line.split()[1].strip("()").split(",")
                elim_order = list(map(int, elim_order))
                if debug: print("elim-order:", elim_order)
            else:
                vertex, rest = line.split(":")
                vertex = int(vertex)
                rest = rest.strip().split()
                if len(rest) == 1:
                    local_score = float(rest[0])
                    parents = frozenset()
                else:
                    local_score, parents = rest
                    local_score = float(local_score)
                    parents = frozenset(map(int, parents.strip("()").split(",")))
                # bn.add(int(vertex), map(int, parents))
                tuples.append((vertex, parents))
                if add_extra_tuples:
                    extra_tuples[vertex] = (parents, local_score)
                if debug: print(f"v:{vertex}, sc:{local_score}, par:{parents})")
    if add_extra_tuples:
        inject_tuples(filename, extra_tuples, augfile, True)
        if debug: print("temporary merged file saved as", augfile)
        input_file = augfile
    else:
        input_file = filename
    if cwidth > 0:
        assert datfile, "datfile needed for cwidth"
        bn = CWBayesianNetwork(cwidth=cwidth, input_file=input_file, datfile=datfile)
    else:
        bn = TWBayesianNetwork(tw=treewidth, input_file=input_file)
    if elim_order is not None: bn.elim_order = elim_order
    for node, parents in tuples: bn.add(node, parents)
    bn.done()
    if score is not None:
        bn._score = score
    else:
        print(f"warning: score not found in {outfile}")
    return bn


def parse_res_score(outfile: str):
    with open(outfile) as out:
        for line in out:
            if not line.strip():
                continue
            elif line.startswith("Score:"):
                score = float(line.split()[1].strip())
                return score
    return None


def write_res(bn: BayesianNetwork, outfile, write_elim_order=False, debug=False):
    nodes = sorted(bn.dag.nodes())
    scores = bn.compute_all_scores()
    with open(outfile, "w") as out:
        if write_elim_order and isinstance(bn, TWBayesianNetwork):
            elim_order = bn.td.recompute_elim_order()
            out.write(f"elim-order: ({','.join(map(str, elim_order))})\n")
        for node in nodes:
            parents = bn.parents[node]
            pstr = ""
            if parents:
                pstr = " (" + ",".join(map(str, parents)) + ")"
            out.write(f"{node}: {scores[node]:.2f} {pstr}\n")
        out.write(f"\nScore: {bn.compute_score():.3f}\n")


def write_net(bn: BayesianNetwork, datfile, tempprefix="lltemp",
              use_res="", debug=False):
    netfile = f"{tempprefix}.net"
    if use_res:
        resfile = use_res
    else:
        resfile = f"{tempprefix}.res"
        write_res(bn, resfile)
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"), "parle"]
    args = ["-d", datfile, "-r", resfile, "-n", netfile]
    cmd = basecmd + args
    if debug: print("running parle, cmd:", cmd)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    if proc.returncode == 0:  # success
        if debug: print("file generated:", netfile)
    else:  # error
        print("error encountered during 'parle', returncode:", proc.returncode)
        return None


def run_blip(filename, treewidth, outfile="temp.res", timeout=10, seed=0,
             solver="kg", logfile="temp.log", debug=False) -> TWBayesianNetwork:
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"),
               f"solver.{solver}", "-v", "1"]
    args = ["-j", filename, "-w", str(treewidth), "-r", outfile,
            "-t", str(timeout), "-seed", str(seed), "-l", logfile]
    cmd = basecmd + args
    if debug: print("running blip, cmd:", cmd)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    if proc.returncode == 0:  # success
        return parse_res(filename, treewidth, outfile, debug)
    else:  # error
        print("error encountered, returncode:", proc.returncode)
        return None


def activate_checkpoints(bnprovider, save_as):
    def alarm_handler(signalnum, frame):
        print("alarm triggered")
        try:
            bn = bnprovider()
        except IndexError:
            print("bn res file invalid (probably got overwritten)")
            signal.alarm(CHECKPOINT_RETRY_INTERVAL)  # snooze alarm
        else:
            patn = save_as.replace(".res", "*.res")
            prefix, ext = os.path.splitext(save_as)
            prev_checkpoint = 0
            for check_file in glob(patn):
                try:
                    saved_at = int(check_file.replace(prefix, "").replace(ext, ""))
                except ValueError:
                    continue
                else:
                    prev_checkpoint = max(prev_checkpoint, saved_at)
            new_checkpoint = prev_checkpoint + CHECKPOINT_MINUTES
            fname = save_as.replace(".res", f"{new_checkpoint}.res")
            print("saving checkpoint to", fname)
            write_res(bn, fname, write_elim_order=True)
            signal.alarm(CHECKPOINT_INTERVAL)  # reset alarm
    signal.signal(signal.SIGALRM, alarm_handler)  # register handler
    signal.alarm(CHECKPOINT_INTERVAL)  # set first alarm
    print(f"checkpointing for {CHECKPOINT_MINUTES}m activated")


def monitor_blip(filename, treewidth, logger: Callable, outfile="temp.res",
                 timeout=10, seed=0, solver="kg", datfile=None,
                 cwidth=0, onlyfilter=False, save_as="", debug=False):
    """
    Run BLIP in monitoring mode, where each new score update is logged

    :param filename: path to jkl file
    :param treewidth: treewidth bound (ignored if in CWIDTH_MODE)
    :param logger: logging function to be used
    :param outfile: path to .res file containing learned network (volatile)
    :param timeout: total time limit on blip computation
    :param seed: random seed passed on to blip
    :param solver: blip sub-algo to use (for tw: [kg, ka, kmax], cwidth: [old, greedy, max])
    :param datfile: path to data file (with header rows, for domain sizes)
    :param cwidth: cwidth bound (if positive, activates CWIDTH_MODE)
    :param onlyfilter: only use pset filtering algo (ignored if not in CWIDTH_MODE)
    :param save_as: filepath prefix to use for saving checkpoint solutions
    :param debug: enable debug mode
    """
    CWIDTH_MODE = cwidth > 0
    if CWIDTH_MODE:
        assert solver in ("old", "greedy", "max"), \
            f"invalid solver({solver}) for monitor_blip in CWIDTH_MODE"
        basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip-cw.jar"),
                   f"solver.kg.adv", "-v", "1", "-src", f"cwidth-{solver}"]
        args = ["-j", filename, "-d", datfile, "-w", "0", "-cw", str(cwidth),
                "-r", outfile, "-t", str(timeout), "-seed", str(seed)]
        if onlyfilter: args.append("-filter")
    else:
        basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"),
                   f"solver.{solver}", "-v", "1"]
        args = ["-j", filename, "-w", str(treewidth), "-r", outfile,
                "-t", str(timeout), "-seed", str(seed)]
    cmd = basecmd + args
    if debug: print("monitoring blip, cmd:", " ".join(cmd))
    if save_as:
        if CWIDTH_MODE:
            bnprovider = lambda: parse_res(filename, 0, outfile, cwidth=cwidth,
                                           datfile=datfile)
        else:
            bnprovider = lambda: parse_res(filename, treewidth, outfile)
        activate_checkpoints(bnprovider, save_as)
    domain_sizes = None if datfile is None else get_domain_sizes(datfile)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1,
                          universal_newlines=True) as proc:
        for line in proc.stdout:
            if debug: print("got line:", line, end='')
            match = SCORE_PATN.match(line)
            if match:
                score = float(match['score'])
                logdata = {"score": score}
                if domain_sizes is not None:
                    try:
                        bn = bnprovider()
                    except IndexError:  # todo: not reached anymore, retries, rethink
                        print("bn res file invalid (probably got overwritten)")
                        cw = acw = -1
                    else:
                        tw = bn.td.compute_width()
                        cw = compute_complexity_width(bn.td, domain_sizes)
                        logdata["tw"] = tw
                        logdata["cw"] = cw
                logger(logdata)
    print(f"done returncode: {proc.returncode}")


def start_blip_proc(filename, treewidth, outfile="temp.res",
                    timeout=10, seed=0, solver="kg", debug=False) -> subprocess.Popen:
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"),
               f"solver.{solver}", "-v", "1"]
    args = ["-j", filename, "-w", str(treewidth), "-r", outfile,
            "-t", str(timeout), "-seed", str(seed)]
    cmd = basecmd + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1,
                            universal_newlines=True)
    os.set_blocking(proc.stdout.fileno(), False)  # set stdout to be non-blocking
    if debug: print(f"starting blip proc, pid: {proc.pid}, \ncmd: {cmd}")
    return proc


def start_blip_proc_cw(filename, datfile, cwidth, outfile="temp.res", timeout=10,
                       seed=0, searcher="greedy", onlyfilter=False, debug=False) -> subprocess.Popen:
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip-cw.jar"),
               f"solver.kg.adv", "-v", "1", "-src", f"cwidth-{searcher}"]
    args = ["-j", filename, "-d", datfile, "-w", "0", "-cw", str(cwidth),
            "-r", outfile, "-t", str(timeout), "-seed", str(seed)]
    if onlyfilter: args.append(["-filter"])
    cmd = basecmd + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1,
                            universal_newlines=True)
    os.set_blocking(proc.stdout.fileno(), False)  # set stdout to be non-blocking
    if debug: print(f"starting blip-cw proc, pid: {proc.pid}, \ncmd: {cmd}")
    return proc


def check_blip_proc(proc: subprocess.Popen, debug=False) -> float:
    score = float("-infinity")
    rc = proc.poll()
    if rc is not None:
        if debug: print(f"blip proc already terminated with rc:", rc)
        return score
    while True:
        line = proc.stdout.readline()
        if line:
            if debug: print("got line:", line, end='')
            match = SCORE_PATN.match(line)
            if match:
                score = float(match['score'])
        else:
            rc = proc.poll()
            if rc is not None:
                if debug: print(f"no output and proc is completed (rc: {rc})")
            else:
                if debug: print(f"no output and proc is still running")
            break
    return score


def stop_blip_proc(proc: subprocess.Popen):
    proc.terminate()
    proc.stdout.close()
    if proc.stderr is not None: proc.stderr.close()
    if proc.stdin is not None: proc.stdin.close()
    proc.wait()


def print_bn(bn: BayesianNetwork):
    for node, pset in bn.parents.items():
        score = bn.compute_all_scores({node})[node]
        print(f"{node}: {score:.2f} ({','.join(map(str,pset))})")


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

