#!/bin/python3.6

# external
import networkx as nx
from typing import Dict, List, Tuple, FrozenSet, Set
import sys, os
import subprocess
from time import time as now
import shutil

# internal
from samer_veith import SvEncoding, SelfNamingDict
from utils import read_bn, read_model, BNData, TreeDecomposition, NoSolutionException, weights_from_domain_sizes
from utils import pairs, posdict_to_ordering, check_subgraph, get_domain_sizes
from blip import TWBayesianNetwork
from complexity_encoding import SvEncodingWithComplexity

TOP = int(1e15)  # todo: make dynamic
SOLVER_DIR = "../solvers"
TIMEOUT = 10
ROUNDING = 1

# todo[design]: decide data structure
#  maybe Dict[node, Dict[green parent g1, List[red verts upstream from g1]]]
PSET_ACYC = Dict[Tuple[int, FrozenSet[int]], Set[int]]


class TwbnEncoding(SvEncoding):
    def __init__(self, data: BNData, stream, forced_arcs=None,
                 forced_cliques=None, pset_acyc=None, debug=False):
        dummy_graph = nx.Graph()
        dummy_graph.add_nodes_from(data.keys())
        self.debug = debug
        super().__init__(stream, dummy_graph)
        self.data = data
        self.non_improved = True  # only use improved for SMT encodings

        # sat variables
        self.acyc = None
        self.par = None
        self.header_template = f"p wcnf {{}} {{}} {TOP}"

        # slim related sat variables
        forced_arcs = [] if forced_arcs is None else forced_arcs
        self.forced_arcs: List[Tuple[int, int]] = forced_arcs
        forced_cliques = dict() if forced_cliques is None else forced_cliques
        self.forced_cliques: Dict[int, FrozenSet[int]] = forced_cliques
        pset_acyc = dict() if pset_acyc is None else pset_acyc
        self.pset_acyc: PSET_ACYC = pset_acyc
        self.rounding: int = ROUNDING

    def _add_clause(self, *args, weight=TOP):
        # use TOP if no weight specified (hard clauses)
        self.stream.write(f"{weight} ")
        super()._add_clause(*args)

    def _add_comment(self, comment):
        if self.debug: self.stream.write(f"c {comment}\n")

    def _acyc(self, i, j):
        if i < j:
            return self.acyc[i][j]
        else:
            return -1 * self.acyc[j][i]

    def encode_exactly_one(self, variables):
        # at least one
        self._add_clause(*variables)

        # at most one
        for v1, v2 in pairs(variables):
            self._add_clause(-v1, -v2)

    def encode_bn(self):
        # clauses: transitivity of acyc
        self._add_comment("begin transitivity for acyc")
        self.encode_transitivity(self._acyc)
        self._add_comment("end transitivity for acyc")

        data = self.data
        for _v in data:
            v = self.node_lookup[_v]
            for p, score in data[_v].items():
                # clause: par(v,p) weighted by f(v,p)
                self._add_comment(f"soft clause par({_v}, {set(p)}), wt: {score:.2f}")
                weight = self.rounding * int(score / self.rounding)
                if len(p) != 0 and weight == 0:  # causes floating point exception in solver
                    self._add_comment(f"skipping, because non-trivial parent with weight 0")
                    continue
                self._add_clause(self.par[v][p], weight=weight)

                for _u in p:
                    u = self.node_lookup.get(_u)
                    if u is None: continue  # external vertex, ignore
                    # clause: par(v,p) => acyc*(u,v) for each u in p, p in Pf(v)
                    self._add_comment(f"par({_v}, {set(p)}) => acyc*({_u},{_v})")
                    self._add_clause(-self.par[v][p], self._acyc(u,v))

            # clauses: exactly one of par(v,p) for each v
            self._add_comment(f"begin exactly one par for {_v}")
            self.encode_exactly_one(self.par[v].values())
            self._add_comment(f"end exactly one par for {_v}")

        # slim only
        for _v, _u in self.forced_arcs:
            v, u = self.node_lookup[_v], self.node_lookup[_u]
            # slim-clause: acyc*(u,v) for each forced directed edge u->v
            self._add_comment(f"[slim] forced acyc {_v}->{_u}")
            self._add_clause(self._acyc(v, u))

        # external parent set compelled acyc ordering
        for (_v, p), acyc_predecessors in self.pset_acyc.items():
            v = self.node_lookup[_v]
            for _u in acyc_predecessors:
                u = self.node_lookup[_u]
                self._add_comment(f"[slim] forced acyc par({_v}, {set(p)}) => acyc*({_u},{_v})")
                self._add_clause(-self.par[v][p], self._acyc(u, v))

    def encode_tw(self):
        data = self.data
        for _v in data:
            v = self.node_lookup[_v]
            for p, score in data[_v].items():
                for _u in p:
                    u = self.node_lookup.get(_u)
                    if u is None: continue  # external vertex, ignore
                    # clause: if par and ord then arc
                    self._add_comment(f"par({_v}, {set(p)}) and ord*({_u},{_v}) => arc({_u},{_v})")
                    self._add_clause(-self.par[v][p], -self._ord(u, v), self.arc[u][v])
                    self._add_comment(f"par({_v}, {set(p)}) and ord*({_v},{_u}) => arc({_v},{_u})")
                    self._add_clause(-self.par[v][p], -self._ord(v, u), self.arc[v][u])

                self._add_comment(f"begin moralization of parent {set(p)} of {v}")
                for _u, _w in pairs(p):
                    u, w = self.node_lookup.get(_u), self.node_lookup.get(_w)
                    if u is None or w is None: continue  # external vertices, ignore
                    # clause: moralization (arc between common parents)
                    self._add_comment(f"par({_v}, {set(p)} and ord*({_u},{_w}) => arc({_u},{_w})")
                    self._add_clause(-self.par[v][p], -self._ord(u, w), self.arc[u][w])
                    self._add_comment(f"par({_v}, {set(p)} and ord*({_w},{_u}) => arc({_w},{_u})")
                    self._add_clause(-self.par[v][p], -self._ord(w, u), self.arc[w][u])
                self._add_comment(f"end moralization of parent {set(p)} of {v}")

        # slim only
        for _, bag in self.forced_cliques.items():
            if len(bag) <= 1: continue  # nothing to encode
            self._add_comment(f"begin [slim] forced clique {set(bag)}")
            for _u, _v in pairs(bag):
                u, v = self.node_lookup[_u], self.node_lookup[_v]
                # slim-clause: ord* => arc for nodes in same boundary bag
                self._add_comment(f"\t {_u} before {_v} implies {_u}->{_v}")
                self._add_clause(-self._ord(u, v), self.arc[u][v])
                self._add_comment(f"\t {_v} before {_u} implies {_v}->{_u}")
                self._add_clause(-self._ord(v, u), self.arc[v][u])
            self._add_comment(f"end [slim] forced clique {set(bag)}")

    def encode(self):
        # allow at most one arc (not strictly necessary)
        for i,j in pairs(range(self.num_nodes)):
            _i, _j = self.node_reverse_lookup[i], self.node_reverse_lookup[j]
            self._add_comment(f"at most one arc {_i}->{_j} or {_j}->{_i}")
            self._add_clause(-self.arc[i][j], -self.arc[j][i])

        self.encode_bn()

        # setup graph variables for treewidth computation
        self.encode_tw()

        super().encode()

    def encode_edges(self):
        # no edges to encode in case of BN
        pass

    def sat_init_vars(self):
        super().sat_init_vars()
        self.acyc = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, self.num_nodes)}
        self.par  = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, self.num_nodes)}

        for i in range(0, self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self._acyc(i, j)


class TwbnDecoder(object):
    def __init__(self, encoder: TwbnEncoding, tw: int, model: set, infile: str):
        self.encoder = encoder
        self.infile  = infile
        self.model = model
        self.tw = tw

    def _get_order(self, vars):
        num_nodes = self.encoder.num_nodes
        position = {node: 0 for node in range(num_nodes)}
        for u, v in pairs(range(num_nodes)):
            var = vars[u][v]
            if var in self.model:
                position[v] += 1
            else:
                position[u] += 1
        ordering = posdict_to_ordering(position)
        return [self.encoder.node_reverse_lookup[i] for i in ordering]

    def get_elim_order(self):
        return self._get_order(self.encoder.ord)

    def get_dag_order(self):
        return self._get_order(self.encoder.acyc)

    def get_triangulated(self):
        graph = nx.DiGraph()
        elim_order = self.get_elim_order()
        # add nodes
        graph.add_nodes_from(elim_order)
        # add edges
        for _u, _v in pairs(elim_order):
            u = self.encoder.node_lookup[_u]
            v = self.encoder.node_lookup[_v]
            # only consider arc if it obeys elim order
            if self.encoder.arc[u][v] in self.model:
                graph.add_edge(_u, _v)
        return graph

    def get_parents(self):
        parents = dict()
        for v in range(self.encoder.num_nodes):
            _v = self.encoder.node_reverse_lookup[v]
            for p in self.encoder.par[v]:
                var = self.encoder.par[v][p]
                if var in self.model:
                    assert _v not in parents, "invalid model, two parent sets assigned"
                    parents[_v] = p
            assert _v in parents, "invalid model, no parent set assigned"
        return parents

    def get_bn(self):
        parents = self.get_parents()
        tri = self.get_triangulated().to_undirected()
        elim_order = self.get_elim_order()
        td = TreeDecomposition(tri, elim_order, width=self.tw)
        bn = TWBayesianNetwork(self.infile, self.tw, elim_order, td=td, parents=parents)
        assert check_subgraph(tri, bn.get_triangulated().subgraph(elim_order)), \
            "parent-based ordered triangulated graph is subgraph assumption failed"
        return bn


def solve_bn(data: BNData, treewidth: int, input_file: str, forced_arcs=None,
             forced_cliques=None, pset_acyc=None, timeout: int = TIMEOUT,
             domain_sizes=None, debug=False):
    cnfpath = "temp.cnf"
    with open(cnfpath, 'w') as cnffile:
        if domain_sizes is None:
            enc = TwbnEncoding(data, cnffile, forced_arcs, forced_cliques,
                               pset_acyc, debug)
        else:
            enc = CwbnEncoding(data, cnffile, forced_arcs, forced_cliques,
                               pset_acyc, debug)
            enc.use_dd = True
            enc.set_weights(weights_from_domain_sizes(domain_sizes))
        enc.encode_sat(treewidth)
    if debug: print("encoding done")
    base_cmd = [os.path.join(SOLVER_DIR, "uwrmaxsat"), "-m", "-v0"]
    cmd = base_cmd + [cnfpath, f"-cpu-lim={timeout}"]
    start = now()
    try:
        output = subprocess.check_output(cmd, universal_newlines=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        # copy over problematic cnf file
        errcnf = "error.cnf"
        shutil.copy(cnfpath, errcnf)
        if err.stdout is not None:
            errfilename = "uwrmaxsat-err.log"
            with open(errfilename, 'w') as errfile:
                errfile.write(err.stdout)
                #raise RuntimeError(f"error while running uwrmaxsat on {errcnf}"
                #                   f"\nrc: {err.returncode}, check {errfilename}")
                print(f"error while running uwrmaxsat on {errcnf}"
                      f"\nrc: {err.returncode}, check {errfilename}")
                raise NoSolutionException("nonzero returncode")
        else:
            #raise RuntimeError(f"error while running uwrmaxsat on {errcnf}"
            #                   f"\nrc: {err.returncode}, no stdout captured")
            print(f"error while running uwrmaxsat on {errcnf}"
                  f"\nrc: {err.returncode}, no stdout captured")
            raise NoSolutionException("nonzero returncode")
    else:  # if no error while maxsat solving
        runtime = now() - start
        model = read_model(output)
        dec = TwbnDecoder(enc, treewidth, model, input_file)
        return dec.get_bn()


class CwbnEncoding(TwbnEncoding, SvEncodingWithComplexity):
    pass


if __name__ == '__main__':
    input_file = "child-norm.jkl"
    if input("complexity-width? y/[n]:"):
        print("complexity-width")
        datfile = "../input/dat/child-5000.dat"
        domain_sizes = get_domain_sizes(datfile)
        print("weights:", weights_from_domain_sizes(domain_sizes))
    else:
        print("treewidth")
        domain_sizes = None
    bn = solve_bn(read_bn(input_file), 5, input_file, timeout=30,
                  domain_sizes=domain_sizes, debug=False)
    print(bn.score)
