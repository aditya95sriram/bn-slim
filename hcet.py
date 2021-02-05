#!/bin/python2

import os
import sys
from time import time as now
import pandas
import argparse

from learn_structure_cache import hill_climbing_cache
from scoring_functions import score_function
import data_type

WANDB_FOUND = True
try:
    import wandb
except ImportError:
    WANDB_FOUND = False


# i/o utility functions

class FileReader(object):  # todo[safety]: add support for `with` usage
    def __init__(self, filename, ignore="#"):
        self.file = open(filename)
        self.ignore = ignore

    def readline(self):
        line = self.file.readline()
        while line.startswith(self.ignore):
            line = self.file.readline()
        return line

    def readints(self):
        return map(int, self.readline().split())

    def readint(self):
        return int(self.readline().strip())

    def close(self):
        self.file.close()


def read_jkl(filename):
    data = dict()
    reader = FileReader(filename, ignore="#")
    n = reader.readint()
    for i in range(n):
        psets = dict()
        node, numsets = reader.readints()
        for j in range(numsets):
            score, parents = reader.readline().split(" ", 1)
            score = float(score)
            parents = frozenset(map(int, parents.split()[1:]))
            psets[parents] = score
        data[node] = psets
    reader.close()
    return data


def write_jkl(data, outfname):
    with open(outfname, 'w') as outfile:
        outfile.write("%d\n" % len(data))
        for node, psets in data.items():
            outfile.write("%d %d\n" % (node, len(psets)))
            for parents, score in sorted(psets.items(), key=lambda a: a[1], reverse=True):
                parentstr = " ".join(map(str, sorted(parents)))
                outfile.write("%.4f %d %s\n" % (score, len(parents), parentstr))


# et-learn

def extract_eo(et):
    num_vars = et.nodes.num_nds
    parents = {i: et.nodes[i].parent_et for i in range(num_vars)}
    elim_order = []
    while len(elim_order) < num_vars:
        not_parent = set(parents.keys())
        for var in parents.keys():
            par = parents[var]
            not_parent.discard(par)
        for var in not_parent:
            elim_order.append(var)
            del parents[var]
    return elim_order


def export_et(et, dataframe, basename):
    """
    writes out given elimination tree as a .res file containing the BN and
    a .jkl file containing the parent set scores
    """
    data = data_type.data(dataframe)
    total_score = 0.0
    all_scores = {}
    with open(basename + '.res', 'w') as outfile:
        eo = extract_eo(et)
        outfile.write("elim-order: ({})\n".format(",".join(map(str, eo))))
        for i in range(et.nodes.num_nds):
            parents = et.nodes[i].parents.display()
            parentstr = "({})".format(",".join(map(str, parents))) if parents else ""
            local_score = score_function(data, i, parents, metric='bic')
            all_scores[i] = {frozenset(parents): local_score}
            total_score += local_score
            outfile.write("{}: {:.2f}  {}\n".format(i, local_score, parentstr))
        outfile.write("Score: {:.2f}\n".format(total_score))
    write_jkl(all_scores, basename + ".jkl")
    return total_score


# noinspection PyTypeChecker
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("datfile", help="path to input dat file")
parser.add_argument("treewidth", help="bound for treewidth", type=int)
parser.add_argument("-p", "--poly", help="use hc-et-poly", action='store_true')
parser.add_argument("-o", "--out-prefix", default="hcet",
                    help="prefix of output files (.res, .jkl, .log)")
parser.add_argument("-j", "--jkl-file", default="", help="path to jkl file, "
                                                         "for forcing hc to use precomputed scores")
parser.add_argument("-l", "--logging", help="wandb logging", action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    datfile = os.path.abspath(args.datfile)
    data = pandas.read_csv(datfile)
    nrows, ncols = data.shape
    if ncols == 1:
        data = pandas.read_csv(datfile, sep=" ")

    if args.logging:
        if not WANDB_FOUND: print("wandb not installed, cannot log")
        # wandb init and config
        wandb.init(project="etlearn-exp1")
        wandb.config.datfile = os.path.basename(args.datfile)
        wandb.config.treewidth = args.treewidth
        wandb.config.poly = args.poly
        # process config
        wandb.config.jobid = os.environ.get("MY_JOB_ID", -1)
        wandb.config.taskid = os.environ.get("MY_TASK_ID", -1)

    basename = args.out_prefix

    # redirect output to logfile
    oldstdout, oldstderr = sys.stdout, sys.stderr
    logfile = open(basename + ".log", 'w')
    # sys.stdout = sys.stderr = logfile
    sys.stdout = sys.stderr
    start = now()
    if args.jkl_file:
        our_score_data = read_jkl(os.path.abspath(args.jkl_file))
    else:
        our_score_data = None
    et = hill_climbing_cache(data, tw_bound=args.treewidth, metric='bic', add_only=args.poly)
    elapsed = now() - start
    # sys.stdout, sys.stderr = oldstdout, oldstderr
    sys.stdout = oldstdout

    score = export_et(et, data, basename)
    if args.logging and WANDB_FOUND: wandb.log({'time': elapsed, 'score': score})
    print "score:", score
    print "time: %.6fs" % elapsed
