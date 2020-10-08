#!/bin/python3.6

import subprocess
import os
import networkx as nx
import random
from typing import Optional, Callable, Dict, List
import re
import signal
from glob import glob

from utils import filled_in, TreeDecomposition, pairs, stream_bn, get_bn_stats, filter_read_bn

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import pygraphviz_layout

SOLVER_DIR = "../solvers"
SCORE_PATN = re.compile(r"New improvement! (?P<score>[\d.-]+) \(after (?P<time>[\d.-]+) s.\)")
CHECKPOINT_MINUTES = 30  # in minutes
CHECKPOINT_INTERVAL = int(CHECKPOINT_MINUTES*60)  # in seconds
CHECKPOINT_RETRY_INTERVAL = 5  # in seconds


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

    def verify(self):
        super().verify()
        self.td.verify(graph=self.get_moralized())

    def get_triangulated(self, elim_order=None):
        if elim_order is None: elim_order = self.elim_order
        assert elim_order is not None, "elim order not specified"
        moral = self.get_moralized().subgraph(elim_order)
        triangulated, max_degree = filled_in(moral, elim_order)
        assert max_degree <= self.tw, f"tw {self.tw} < {max_degree} for elim order {elim_order}"
        return triangulated


def parse_res(filename, treewidth, outfile, debug=False) -> TWBayesianNetwork:
    bn = TWBayesianNetwork(tw=treewidth, input_file=filename)
    with open(outfile) as out:
        for line in out:
            if not line.strip():
                continue
            elif line.startswith("Score:"):
                score = float(line.split()[1].strip())
                break
            elif line.startswith("elim-order:"):
                elim_order = line.split()[1].strip("()").split(",")
                bn.elim_order = list(map(int, elim_order))
                if debug: print("elim-order:", bn.elim_order)
            else:
                vertex, rest = line.split(":")
                rest = rest.strip().split()
                if len(rest) == 1:
                    local_score = float(rest[0])
                    parents = []
                else:
                    local_score, parents = rest
                    parents = parents.strip("()").split(",")
                bn.add(int(vertex), map(int, parents))
                if debug: print(f"v:{vertex}, sc:{local_score}, par:{parents})")
    bn.done()
    bn._score = score
    return bn


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


def write_net(bn: BayesianNetwork, datfile, tempprefix="lltemp", debug=False):
    resfile = f"{tempprefix}.res"
    netfile = f"{tempprefix}.net"
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
                 timeout=10, seed=0, solver="kg", save_as="", debug=False):
    basecmd = ["java", "-jar", os.path.join(SOLVER_DIR, "blip.jar"),
               f"solver.{solver}", "-v", "1"]
    args = ["-j", filename, "-w", str(treewidth), "-r", outfile,
            "-t", str(timeout), "-seed", str(seed)]
    cmd = basecmd + args
    if debug: print("monitoring blip, cmd:", " ".join(cmd))
    if save_as: activate_checkpoints(lambda: parse_res(filename, treewidth, outfile), save_as)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1,
                          universal_newlines=True) as proc:
        for line in proc.stdout:
            if debug: print("got line:", line, end='')
            match = SCORE_PATN.match(line)
            if match:
                score = float(match['score'])
                logger({"score": score})
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

