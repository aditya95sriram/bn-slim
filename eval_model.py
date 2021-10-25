#!/bin/python3.6

import os
import sys
import subprocess
import json
import random
from math import exp
from time import time as now
from datetime import datetime
from shutil import copyfile

from bnio import to_uai
from blip import BayesianNetwork, write_net, parse_res
from utils import get_vardata

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class Evaluator:
    SOLVER_DIR = "../solvers"

    def __init__(self, bn: BayesianNetwork = None, datfile=None, tempprefix="lltemp", 
                 uaifile=None, debug=False):
        self.tempprefix = tempprefix
        if uaifile is not None:
            self.uaifile = uaifile
            if debug: print("directly uai file provided, skipping conversion")
            return
        self.bn = bn
        self.datfile = datfile
        #self.datfile = f"{tempprefix}.dat"
        #copyfile(datfile, self.datfile)
        self.netfile = f"{tempprefix}.net"
        self.uaifile = f"{tempprefix}.uai"
        if debug: print("writing net file")
        write_net(bn, self.datfile, tempprefix)
        self.elim_order = None
        self.treewidth = -1
        if hasattr(bn, 'elim_order'):
            varnames = list(get_vardata(self.datfile).keys())
            self.elim_order = [varnames[i] for i in getattr(bn, 'elim_order')]
        if hasattr(bn, 'td'):
            self.treewidth = getattr(bn, "td").compute_width()
        if debug: print("writing uai file")
        to_uai(self.netfile, self.uaifile, self.elim_order)

    @staticmethod
    def from_uai(uaifile, td):
        eva = Evaluator(uaifile=uaifile)
        eva.treewidth = td.width
        eva.elim_order = td.elim_order
        return eva

    def ll(self, debug=False):
        basecmd = ["java", "-jar", os.path.join(self.SOLVER_DIR, "blip.jar"), "lleval"]
        args = ["-n", self.netfile, "-d", self.datfile]
        cmd = basecmd + args
        if debug: print("running blip, cmd:", cmd)
        try:
            output = subprocess.check_output(cmd, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print("error encountered, returncode:", e.returncode)
            return None
        else:
            return float(output)

    def prob_evid(self, evid, exact=True, debug=False):
        nevid = len(evid)
        evid_fname = f"{self.tempprefix}.evid"
        with open(evid_fname, 'w') as evid_file:
            evid_file.write(f"{nevid}")
            for var, val in evid.items():
                var_id = self.elim_order.index(var)
                evid_file.write(f" {var_id} {val}")
            evid_file.write("\n")
        basecmd = [os.path.join(self.SOLVER_DIR, "merlin"), "-t", "PR", "-a", "wmb",
                   "-M", "lex", "-o", self.tempprefix, "-O", "json"]
        if exact:
            basecmd += ["-i", str(self.treewidth)]
        args = ["-f", self.uaifile, "-e", evid_fname]
        cmd = basecmd + args
        out_fname = f"{self.tempprefix}.PR.json"
        if debug: print("running merlin|pr task, cmd:", cmd)
        start = now()
        try:
            proc = subprocess.run(cmd, check=True, stderr=subprocess.PIPE,
                                  stdout=subprocess.PIPE, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == -9:
                print(f"possibly, out of memory error, time: {datetime.now()} ")
                raise MemoryError
            print("error encountered, retcode:", e.returncode)
            print("error message:", e.stderr)
            print("cmd:", " ".join(cmd))
            return None
        else:
            elapsed = now() - start
            with open(out_fname) as outfile:
                result = json.load(outfile)
                prob = exp(result["value"])
                return prob, elapsed


class Comparer:
    def __init__(self, bn: BayesianNetwork, datfile,
                 other: BayesianNetwork = None, tempprefix="lltemp", debug=False):
        self.eval_bn = Evaluator(bn, datfile, f"{tempprefix}_bn", debug=debug)
        if other is not None:
            self.singular = False
            self.eval_base = Evaluator(other, datfile, f"{tempprefix}_base", debug)
        else:
            self.singular = True
        self.vardata = get_vardata(datfile)

    def compare(self, method, *args):
        val_bn = getattr(self.eval_bn, method)(*args)
        if self.singular:
            return val_bn
        else:
            val_base = getattr(self.eval_base, method)(*args)
            return val_bn - val_base

    def delta_ll(self):
        return self.compare('ll')

    def compare_evid(self, nevid=5, exact=True):
        varnames, domains = zip(*self.vardata.items())
        num_nodes = len(varnames)
        evid_vars = random.sample(range(num_nodes), nevid)
        evid = {varnames[idx]: random.randint(0, domains[idx] - 1) for idx in evid_vars}
        prob_bn, elapsed_bn = self.eval_bn.prob_evid(evid, exact)
        if self.singular: return prob_bn, elapsed_bn
        prob_base, elapsed_base = self.eval_base.prob_evid(evid, exact)
        prob = prob_bn - prob_base
        elapsed = elapsed_bn - elapsed_base
        return prob, elapsed

    def compute_mae(self, nevid=5, ntests=100, seed=None, exact=True, debug=False):
        totalprob = 0
        totalelapsed = 0
        negative_elapsed_count = 0
        if seed is not None: random.seed(seed)
        interval = tqdm(range(ntests))
        for _ in interval:
            prob, elapsed = self.compare_evid(nevid, exact)
            totalprob += abs(prob)
            totalelapsed += elapsed
            negative_elapsed_count += elapsed < 0
            if isinstance(interval, tqdm):
                interval.set_description(f"avg={totalprob/ntests:g}")
        if debug:
            print("total elapsed:", totalelapsed)
            print("negative elapsed:", negative_elapsed_count)
        return totalprob / ntests, totalelapsed / ntests


def eval_ll(bn: BayesianNetwork, datfile, tempprefix="lltemp", debug=False):
    return Evaluator(bn, datfile, tempprefix, debug).ll()


def eval_all(filename, tw, datfile, resfile, seed):
    bn = parse_res(filename, tw, resfile)
    cmp = Comparer(bn, datfile, None)
    ll = cmp.delta_ll()
    maescore, maetime = cmp.compute_mae(seed=seed, debug=True)
    return ll, maescore, maetime


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("usage: python eval_model.py <jklfile> <tw> <datfile> <resfile1> [resfile2] [seed]")
        print("(use '-' to skip the optional parameters)")
        sys.exit(1)
    filename, tw, datfile, res1, res2, seed = sys.argv[1:7]
    tw = int(tw)
    bn1 = parse_res(filename, tw, res1)
    bn2 = parse_res(filename, tw, res2) if res2 != "-" else None
    seed = int(seed) if seed != "-" else None
    print(bn1.input_file)
    cmp = Comparer(bn1, datfile, bn2)
    print(f"delta ll: {cmp.delta_ll()}")
    maescore, maetime = cmp.compute_mae(seed=seed, debug=True)
    print(f"mae: {maescore} ({maetime} s avg.)")
