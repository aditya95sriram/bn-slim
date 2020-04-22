#!/bin/python3.6

# external
import networkx as nx
from typing import Dict, List

# internal
from samer_veith import SvEncoding, SelfNamingDict
from utils import num_nodes_bn, read_bn, read_model
from utils import pairs, ord_triples, posdict_to_ordering
from blip import TWBayesianNetwork

TOP = int(1e15)  # todo: make dynamic


class TwbnEncoding(SvEncoding):
    def __init__(self, filename: str, stream, debug=False):
        super().__init__(stream, nx.Graph())
        self.filename = filename
        self.num_nodes = num_nodes_bn(filename)
        self.debug = debug

        # sat variables
        self.acyc = None
        self.par = None
        self.header_template = f"p wcnf {{}} {{}} {TOP}"

        # slim related sat variables
        self.forced_parents: Dict[int, List[int]] = dict()
        self.forced_cliques: List[List[int]] = []

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

        data = read_bn(self.filename)
        for v in data:

            for score, p in data[v]:
                # clause: par(v,p) weighted by f(v,p)
                self._add_comment(f"soft clause par({v}, {set(p)}), wt: {score:.2f}")
                self._add_clause(self.par[v][p], weight=int(score))

                for u in p:
                    # clause: par(v,p) => acyc*(u,v) for each u in p, p in Pf(v)
                    self._add_comment(f"par({v}, {set(p)}) => acyc*({u},{v})")
                    self._add_clause(-self.par[v][p], self._acyc(u,v))

            # clauses: exactly one of par(v,p) for each v
            self._add_comment(f"begin exactly one par for {v}")
            self.encode_exactly_one(self.par[v].values())
            self._add_comment(f"end exactly one par for {v}")

        # slim only
        for v in self.forced_parents:
            for u in self.forced_parents[v]:
                # slim-clause: acyc*(u,v) for each forced directed edge u->v
                self._add_comment(f"[slim] forced edges {u}->{v}")
                self._add_clause(self._acyc(u, v))

    def encode_tw(self):
        data = read_bn(self.filename)
        for v in data:
            for score, p in data[v]:
                for u in p:
                    # clause: if par and ord then arc
                    self._add_comment(f"par({v}, {set(p)}) and ord*({u},{v}) => arc({u},{v})")
                    self._add_clause(-self.par[v][p], -self._ord(u, v), self.arc[u][v])
                    self._add_comment(f"par({v}, {set(p)}) and ord*({v},{u}) => arc({v},{u})")
                    self._add_clause(-self.par[v][p], -self._ord(v, u), self.arc[v][u])

                self._add_comment(f"begin moralization of parent {set(p)} of {v}")
                for u, w in pairs(p):
                    # clause: moralization (arc between common parents)
                    self._add_comment(f"par({v}, {set(p)} and ord*({u},{w}) => arc({u},{w})")
                    self._add_clause(-self.par[v][p], -self._ord(u, w), self.arc[u][w])
                    self._add_comment(f"par({v}, {set(p)} and ord*({w},{u}) => arc({w},{u})")
                    self._add_clause(-self.par[v][p], -self._ord(w, u), self.arc[w][u])
                self._add_comment(f"end moralization of parent {set(p)} of {v}")

        # slim only
        for bag in self.forced_cliques:
            for u, v in pairs(bag):
                # slim-clause: ord* => arc for nodes in same boundary bag
                self._add_clause(-self._ord(u, v), self.arc[u][v])
                self._add_clause(-self._ord(v, u), self.arc[v][u])

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
    def __init__(self, encoder: TwbnEncoding, solfile: str):
        self.encoder = encoder
        self.outfile = solfile
        self.model = read_model(solfile)

    def _get_order(self, vars):
        num_nodes = self.encoder.num_nodes
        position = {node: 0 for node in range(num_nodes)}
        for u, v in pairs(range(num_nodes)):
            var = vars[u][v]
            if var in self.model:
                position[v] += 1
            else:
                position[u] += 1
        return posdict_to_ordering(position)

    def get_elim_order(self):
        return self._get_order(self.encoder.ord)

    def get_dag_order(self):
        return self._get_order(self.encoder.acyc)

    def get_parents(self):
        parents = dict()
        for v in range(self.encoder.num_nodes):
            for p in self.encoder.par[v]:
                var = self.encoder.par[v][p]
                if var in self.model:
                    assert v not in parents, "invalid model, two parent sets assigned"
                    parents[v] = p
            assert v in parents, "invalid model, no parent set assigned"
        return parents

    def get_bn(self):
        parents = self.get_parents()
        elim_order = self.get_elim_order()
        # todo: compute score and store in bn.score
        bn = TWBayesianNetwork(parents=parents, elim_order=elim_order)
        bn.done()
        return bn


if __name__ == '__main__':
    with open("temp.cnf", "w") as f:
        # enc = TwbnEncoding("../past-work/blip-publish/data/child-5000.jkl", f, debug=True)
        enc = TwbnEncoding("testfinal.jkl", f, debug=True)
        enc.encode_sat(3, cardinality=True)
    print("encoding done")
    solfile = input("enter sol file:")
    dec = TwbnDecoder(enc, solfile)
    print("elim-order:", dec.get_elim_order())
    print("dag-order:", dec.get_dag_order())
    print("parents:", dec.get_parents())
    bn = dec.get_bn()
