#!/bin/python3.6

# external
import networkx as nx
from typing import Dict, List, Tuple
import sys, os
import subprocess
from time import time as now

# internal
from samer_veith import SvEncoding, SelfNamingDict
from utils import num_nodes_bn, read_bn, read_model, BNData
from utils import pairs, ord_triples, posdict_to_ordering
from blip import TWBayesianNetwork

TOP = int(1e15)  # todo: make dynamic
SOLVER_DIR = "../solvers"
TIMEOUT = 10


class TwbnEncoding(SvEncoding):
    def __init__(self, data: BNData, stream, forced_arcs=None,
                 forced_cliques=None, debug=False):
        dummy_graph = nx.Graph()
        dummy_graph.add_nodes_from(data.keys())
        super().__init__(stream, dummy_graph)
        self.data = data
        self.debug = debug
        self.non_improved = True

        # sat variables
        self.acyc = None
        self.par = None
        self.header_template = f"p wcnf {{}} {{}} {TOP}"

        # slim related sat variables
        forced_arcs = [] if forced_arcs is None else forced_arcs
        self.forced_arcs: List[Tuple[int, int]] = forced_arcs
        forced_cliques = [] if forced_cliques is None else forced_cliques
        self.forced_cliques: List[List[int]] = forced_cliques

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
                self._add_clause(self.par[v][p], weight=int(score))

                for _u in p:
                    u = self.node_lookup[_u]
                    # clause: par(v,p) => acyc*(u,v) for each u in p, p in Pf(v)
                    self._add_comment(f"par({_v}, {set(p)}) => acyc*({_u},{_v})")
                    self._add_clause(-self.par[v][p], self._acyc(u,v))

            # clauses: exactly one of par(v,p) for each v
            self._add_comment(f"begin exactly one par for {_v}")
            self.encode_exactly_one(self.par[v].values())
            self._add_comment(f"end exactly one par for {_v}")

        # slim only
        for _v in self.forced_arcs:
            for _u in self.forced_arcs[_v]:
                v, u = self.node_lookup[_v], self.node_lookup[_u]
                # slim-clause: acyc*(u,v) for each forced directed edge u->v
                self._add_comment(f"[slim] forced edge {_u}->{_v}")
                self._add_clause(self._acyc(u, v))

    def encode_tw(self):
        data = self.data
        for _v in data:
            v = self.node_lookup[_v]
            for p, score in data[_v].items():
                for _u in p:
                    u = self.node_lookup[_u]
                    # clause: if par and ord then arc
                    self._add_comment(f"par({_v}, {set(p)}) and ord*({_u},{_v}) => arc({_u},{_v})")
                    self._add_clause(-self.par[v][p], -self._ord(u, v), self.arc[u][v])
                    self._add_comment(f"par({_v}, {set(p)}) and ord*({_v},{_u}) => arc({_v},{_u})")
                    self._add_clause(-self.par[v][p], -self._ord(v, u), self.arc[v][u])

                self._add_comment(f"begin moralization of parent {set(p)} of {v}")
                for _u, _w in pairs(p):
                    u, w = self.node_lookup[_u], self.node_lookup[_w]
                    # clause: moralization (arc between common parents)
                    self._add_comment(f"par({_v}, {set(p)} and ord*({_u},{_w}) => arc({_u},{_w})")
                    self._add_clause(-self.par[v][p], -self._ord(u, w), self.arc[u][w])
                    self._add_comment(f"par({_v}, {set(p)} and ord*({_w},{_u}) => arc({_w},{_u})")
                    self._add_clause(-self.par[v][p], -self._ord(w, u), self.arc[w][u])
                self._add_comment(f"end moralization of parent {set(p)} of {v}")

        # slim only
        for bag in self.forced_cliques:
            self._add_comment(f"begin [slim] forced clique {bag}")
            for _u, _v in pairs(bag):
                u, v = self.node_lookup[_u], self.node_lookup[_v]
                # slim-clause: ord* => arc for nodes in same boundary bag
                self._add_clause(-self._ord(u, v), self.arc[u][v])
                self._add_clause(-self._ord(v, u), self.arc[v][u])
            self._add_comment(f"end [slim] forced clique {bag}")

    def encode(self):
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
        elim_order = self.get_elim_order()
        # todo: compute score and store in bn.score
        bn = TWBayesianNetwork(self.infile, self.tw, elim_order, parents=parents)
        bn.done()
        return bn


def solve_bn(data: BNData, treewidth: int, input_file: str, forced_arcs=None,
             forced_cliques=None, timeout: int = TIMEOUT, debug=False):
    cnfpath = "temp.cnf"
    with open(cnfpath, 'w') as cnffile:
        enc = TwbnEncoding(data, cnffile, forced_arcs, forced_cliques, debug)
        enc.encode_sat(treewidth)
    if debug: print("encoding done")
    base_cmd = [os.path.join(SOLVER_DIR, "uwrmaxsat"), "-m", "-v0"]
    cmd = base_cmd + [cnfpath, f"-cpu-lim={timeout}"]
    start = now()
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"error while running uwrmaxsat, rc: {err.returncode}")
    else:
        runtime = now() - start
        model = read_model(output)
        dec = TwbnDecoder(enc, treewidth, model, input_file)
        return dec.get_bn()


if __name__ == '__main__':
    input_file = "../past-work/blip-publish/data/child-5000.jkl"
    bn = solve_bn(read_bn(input_file), 3, input_file)
