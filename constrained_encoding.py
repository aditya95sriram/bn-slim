#!/bin/python3.6

# external (built-ins)
import itertools
import os
import sys
import subprocess
import shutil  # for copypath
from time import perf_counter as now
from typing import Dict, Tuple, Iterable
from contextlib import contextmanager

# external
import networkx as nx

# internal
from utils import BNData, Constraints, IntPairs, read_bn,\
    read_constraints, read_model, remove_zero_weight_parents
from samer_veith import SelfNamingDict
from berg_encoding import TwbnEncoding, TwbnDecoder, \
    TIMEOUT, SOLVER_DIR, NoSolutionException
from blip import print_bn, ConstrainedBayesianNetwork

BinaryBooleanVars = Dict[int, Dict[int, int]]

# todo: figure out if bidirectional has a fixed value
BIDIRECTIONAL = True
# todo: figure out if imply_arc speeds up solving
IMPLY_ARC = True

IncomingArcs = Tuple[IntPairs, IntPairs]  # posarc, negarc
# list of pairs (Ai, Bi) such that any(ai ~~~> bi) where ai in Ai, bi in Bi
PathPairs = Iterable[Tuple[Iterable[int], Iterable[int]]]


class ConstrainedBnEncoding(TwbnEncoding):
    def __init__(self, constraints: Constraints, incoming_arcs: IncomingArcs,
                 path_pairs: PathPairs, external_pairs: PathPairs, data: BNData,
                 stream, forced_arcs=None, forced_cliques=None, pset_acyc=None,
                 debug=False):
        """
        Construct an encoding for the Constrained BNSL problem.
        Use ``ConbnDecoder`` along with an object of this class to decode the
        model output by the MaxSAT solver.

        :param constraints:   expert constraints (only those whose endpoints
                              are in the local instance)
        :param incoming_arcs  pair of tuples ((u*, v), (a*, b) requiring
                              u* ---> v and a* -/-> b where u*, a* are not
                              a part of the local instance and v, b are
        :param path_pairs     tuples (A, B) of sets such that a disjunction of
                              paths is required over the cartesian product AxB
        :param external_pairs forbidden path pairs where left end lies outside.
                              (A, B) such that no path from a in A to b in B
        :param data:          pset cache data
        :param stream:        stream to write encoding to
        :param forced_arcs:
        :param forced_cliques:
        :param pset_acyc:
        :param debug:         debug mode on/off
        """
        super().__init__(data, stream, forced_arcs, forced_cliques, pset_acyc, debug)
        self.constraints = constraints
        self.incoming_posarcs, self.incoming_negarcs = incoming_arcs
        self.path_pairs = path_pairs
        self.external_pairs = external_pairs

        # needed for encode_dagarcs
        self.bidirectional = BIDIRECTIONAL
        self.imply_arc = IMPLY_ARC

        # sat variables

        # only maintain actual dag arcs,
        # unlike arcs which also has moralization and fill-in
        # (structure similar to arc, n(n-1) variables)
        self.dagarc: BinaryBooleanVars = None
        # transarc is transitivity applied to dagarc
        self.transarc: BinaryBooleanVars = None

        # for ancestry constraints
        # binary variable pathp[u][v] denoting if path u ~> v
        self.pathp: BinaryBooleanVars = None
        # ternary helper variable pathq[u,v][z] denoting if path u ~> z -> v
        self.pathq: Dict[Tuple[int, int], Dict[int, int]] = None
        # helper dict storing all possible parents of a variable
        self.allpars: Dict[int, frozenset] = dict()
        for _v in self.data:
            v = self.lookup(_v)
            allpars = set()
            for p, _ in self.data[_v].items():
                # if len(p) == 1: allpars.add(self.lookup(p))
                # else: allpars.update(self.lookup(*p))
                for _u in p:
                    u = self.lookup(_u)
                    if u is not None: allpars.add(u)
            self.allpars[v] = frozenset(allpars)

    @contextmanager
    def clause_block(self, title, section=""):
        """purely for aesthetic purposes of visually distinguishing different
        clause blocks in the cnf file"""
        prefix = f"[{section}] " if section else ""
        self._add_comment("")
        self._add_comment(f"{prefix}begin {title}")
        try:
            yield None
        except Exception as e:
            print(f"{prefix}error while in {title} block", file=sys.stderr)
            # print(e, file=sys.stderr)
            raise e  # must reraise exception as per docs
        finally:
            self._add_comment(f"{prefix}end {title}")

    def lookup(self, *args):
        """convenience vectorized form of self.node_lookup"""
        if len(args) == 1:
            return self.node_lookup.get(args[0])
        else:
            return (self.node_lookup.get(arg) for arg in args)

    def uplook(self, *args):
        """convenience vectorized form of self.node_reverse_lookup"""
        if len(args) == 1:
            return self.node_reverse_lookup.get(args[0])
        else:
            return (self.node_reverse_lookup.get(arg) for arg in args)

    # similar to arc
    def _dagarc(self, i, j):
        return self.dagarc[i][j]  # if i < j else -self.dagarc[j][i]

    # similar to arc
    def _transarc(self, i, j):
        return self.transarc[i][j] #if i < j else -self.transarc[j][i]

    # similar to arc
    def _pathp(self, i, j):
        return self.pathp[i][j] #if i < j else -self.pathp[j][i]

    def _pathq(self, u, v, z):
        return self.pathq[(u, v)][z]

    def encode_dagarcs(self):
        """
        encode dagarcs via ``par(v, p)``. if ``bidirectional``,
        *define* dagarcs by encoding ``dagarc(u,v) <=> par(v,p)``.
        if ``imply_arc``, ``dagarc(u,v) => arc(u,v)``.
        """
        for u, v in self.nodepairs():
            _u, _v = self.uplook(u, v)
            self._add_comment(f"[con] at most one dagarc({_u}, {_v}), dagarc({_v}, {_u})")
            self._add_clause(-self._dagarc(u, v), -self._dagarc(v, u))

        for _v in self.data:
            v = self.lookup(_v)
            parents_v = {u: [] for u in range(self.num_nodes) if u != v}
            with self.clause_block(f"par({_v}, _) => dagarc(_, {_v})", "con"):
                for p, _ in self.data[_v].items():
                    for _u in p:
                        u = self.lookup(_u)
                        if u is None: continue  # external vertex, ignore
                        if self.bidirectional:  # remember all parents of v containing u
                            parents_v[u].append(p)
                        # clause: par(v,p) => dagarc(u,v) for each u in p, p in Pf(v)
                        self._add_comment(f"[con] par({_v}, {set(p)}) => dagarc({_u}, {_v})")
                        self._add_clause(-self.par[v][p], self._dagarc(u, v))

            if self.bidirectional:
                with self.clause_block(f"encoding bidirectional dagarc(_, {_v}) => OR[ par({_v}, _) ]", "con"):
                    for u, psets in parents_v.items():
                        if len(psets) == 0: self._add_comment("#### empty pset")
                        _u = self.uplook(u)
                        par_vars = (self.par[v][p] for p in psets)
                        self._add_comment(f"[con] dagarc({_u}, {_v}) => one of par({_v}, {list(map(set, psets))})")
                        self._add_clause(-self._dagarc(u, v), *par_vars)

        if self.imply_arc:
            with self.clause_block("encoding implyarc dagarc => arc", "con"):
                for u, v in self.nodepairs():
                    _u, _v = self.uplook(u, v)
                    self._add_comment(f"[con] dagarc({_u}, {_v}) => arc({_u}, {_v})")
                    self._add_clause(-self._dagarc(u, v), self.arc[u][v])
                    self._add_comment(f"[con] dagarc({_v}, {_u}) => arc({_v}, {_u})")
                    self._add_clause(-self._dagarc(v, u), self.arc[v][u])

    def encode_transarcs(self):
        """encode transitivity of dagarcs into new variable"""
        # first encode dagarc => transarc
        for u, v in self.nodeopairs():
            _u, _v = self.uplook(u, v)
            self._add_comment(f"[con] dagarc({_u}, {_v}) => transarc( {_u}, {_v})")
            self._add_clause(-self._dagarc(u, v), self._transarc(u, v))

        with self.clause_block("transitivity for transarc", "con"):
            self.encode_transitivity(self._transarc)

    def encode_pathvars(self):
        for u, v in self.nodeopairs():
            _u, _v = self.uplook(u, v)
            allpars = self.allpars[v] - {u}
            if len(allpars) == 1:
                _zs = [self.uplook(*allpars)]
            else:
                _zs = self.uplook(*allpars)

            self._add_comment(f"[con] pathp({_u}, {_v}) => dagarc({_u}, {_v}) or OR[ pathq({_u}, {_v}, {list(_zs)} ]")
            pathqvars = (self._pathq(u, v, z) for z in allpars)
            self._add_clause(-self._pathp(u, v), self._dagarc(u, v), *pathqvars)

            for z in allpars:
                _z = self.uplook(z)
                self._add_comment(f"[con] pathq({_u}, {_v}, {_z}) => pathp({_u}, {_z}) and dagarc({_z}, {_v})")
                self._add_clause(-self._pathq(u, v, z), self._pathp(u, z))
                self._add_clause(-self._pathq(u, v, z), self._dagarc(z, v))

    def encode_constraint_aux_vars(self):
        if self.dagarc is None:
            self.init_constraint_aux_vars()

        with self.clause_block("encoding dagarcs", "con"):
            self.encode_dagarcs()

        with self.clause_block("encoding transarcs", "con"):
            self.encode_transarcs()

        with self.clause_block("encoding pathvars", "con"):
            self.encode_pathvars()

    def encode_internal_constraints(self):
        """encode constraints between variables contained in local instance"""
        with self.clause_block("internal constraints", "con"):
            with self.clause_block("posarc constraints", "con"):
                for _u, _v in self.constraints["posarc"]:
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  ---> {_v}")
                    self._add_clause(self._dagarc(u, v))

            with self.clause_block("negarc constraints", "con"):
                for _u, _v in self.constraints["negarc"]:
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  -/-> {_v}")
                    self._add_clause(-self._dagarc(u, v))

            with self.clause_block("undarc constraints", "con"):
                for _u, _v in self.constraints["undarc"]:
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  ---  {_v}")
                    self._add_clause(self._dagarc(u, v), self._dagarc(v, u))

            with self.clause_block("posanc constraints", "con"):
                for _u, _v in self.constraints["posanc"]:
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  ~~~> {_v}")
                    self._add_clause(self._pathp(u, v))

            with self.clause_block("neganc constraints", "con"):
                for _u, _v in self.constraints["neganc"]:
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  ~/~> {_v}")
                    self._add_clause(-self._transarc(u, v))

            with self.clause_block("undanc constraints", "con"):
                for _u, _v in self.constraints["undanc"]:
                    # raise NotImplementedError("undirected ancestry constraints not supported")
                    u, v = self.lookup(_u, _v)
                    self._add_comment(f"[con] constraint {_u}  ~~~  {_v}")
                    self._add_clause(self._pathp(u, v), self._pathp(v, u))

    def encode_incoming_arcs(self):
        with self.clause_block("incoming arcs", "con+slim"):
            blocked_vars = set()
            with self.clause_block("posarcs", "con+slim"):
                for _u, _v in self.incoming_posarcs:
                    u, v = self.lookup(_u, _v)
                    for pset, var in self.par[v].items():
                        if u not in pset:  # discard psets not containing u
                            self._add_comment(f"block pset {set(pset)} ({var}) "
                                              f"for {_v} as it doesnt contain {_u}")
                            blocked_vars.add(var)

            with self.clause_block("negarcs", "con+slim"):
                for _u, _v in self.incoming_negarcs:
                    u, v = self.lookup(_u, _v)
                    for pset, var in self.par[v].items():
                        if u in pset:  # discard psets containing u
                            self._add_comment(f"block pset {set(pset)} ({var}) "
                                              f"for {_v} as it contains {_u}")
                            blocked_vars.add(var)

            for var in blocked_vars: self._add_clause(-var)

            # raise ValueError("incoming undirected arc must be converted "
            #                  "to incoming directed arc")
            #
            # raise ValueError("incoming path must be converted to either"
            #                  " incoming arc or internal path")
            #
            # raise ValueError("incoming forbidden path must be converted "
            #                  "to internal forbidden paths from descendants")
            #
            # raise NotImplementedError("undirected ancestry constraints not supported")

    def encode_path_pairs(self):
        with self.clause_block("path pairs", "con+slim"):
            for U, V in self.path_pairs:
                self._add_comment(f"some path from {U} to {V}")
                path_vars = []
                for _u, _v in itertools.product(U, V):
                    u, v = self.lookup(_u, _v)
                    path_vars.append(self._pathp(u, v))
                self._add_clause(*path_vars)

    def encode_forbidden_external_pairs(self):
        with self.clause_block("forbidden external paths", "con+slim"):
            for U, V in self.external_pairs:
                with self.clause_block(f"encoding {U} ~/~> {V}", "con+slim"):
                    for _u in self.data:
                        u = self.lookup(_u)
                        for p, _ in self.data[_u].items():
                            if p.intersection(U):  # if parent set intersects
                                self._add_comment(f"par({_u}, {set(p)}) => {_u} ~/~> {V}")
                                for v in V:
                                    self._add_clause(-self.par[u][p], -self._transarc(u, v))

    def encode_constraints(self):
        self.encode_internal_constraints()
        self.encode_incoming_arcs()
        self.encode_path_pairs()
        self.encode_external_pairs()

    def encode(self):
        super().encode()  # encode bn, tw, arcs, transitivity, fill-in (in order)

        # initialize variables needed for constraint encoding
        with self.clause_block("encoding conbnsl", "con"):
            self.init_constraint_aux_vars()
            self.encode_constraint_aux_vars()
            self.encode_constraints()

    def init_constraint_aux_vars(self):
        self.dagarc = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, self.num_nodes)}
        self.transarc = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, self.num_nodes)}

        self.pathp = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, self.num_nodes)}
        self.pathq = {(u, v): SelfNamingDict(lambda: self._add_var()) for u, v in self.nodeopairs()}


class ConbnDecoder(TwbnDecoder):
    def __init__(self, encoder: ConstrainedBnEncoding, tw: int, model: set,
                 infile: str, constraints: Constraints):
        super().__init__(encoder, tw, model, infile)
        self.encoder: ConstrainedBnEncoding = encoder  # for type analysis
        self.constraints = constraints

    def check_dagarc(self, ref_dag):
        dag = nx.DiGraph()
        for u, vs in self.encoder.dagarc.items():
            for v, var in vs.items():
                if var in self.model:
                    _u, _v = self.encoder.uplook(u, v)
                    dag.add_edge(_u, _v)

        our_edges = set(dag.edges)
        ref_edges = set(ref_dag.edges)

        # we only encode internal edges, while ref has some extra incoming arcs
        assert our_edges.issubset(ref_edges), "dagarc encoded additional edges:" \
                                              f"{our_edges - ref_edges}"

        # for all the edges that are extra in ref, left endpoint must be outside
        for _u, _v in ref_edges.difference(our_edges):
            assert self.encoder.lookup(_u) is None, f"dagarc missed edge: {_u}, {_v}"

    def get_bn(self):
        bn = super().get_bn()
        if __debug__: self.check_dagarc(bn.dag)
        return ConstrainedBayesianNetwork.fromTwBayesianNetwork(bn, self.constraints)


def solve_conbn(data: BNData, treewidth: int, input_file: str,
                internal_constraints: Constraints, incoming_arcs: IncomingArcs,
                path_pairs: PathPairs, external_pairs: PathPairs,
                forced_arcs=None, forced_cliques=None, pset_acyc=None,
                timeout: int = TIMEOUT, debug=False):
    cnfpath = "temp.cnf"
    # safer to eliminate zero weight non-trivial parents at this stage
    data = remove_zero_weight_parents(data, debug=debug)
    with open(cnfpath, 'w') as cnffile:
        enc = ConstrainedBnEncoding(internal_constraints, incoming_arcs,
                                    path_pairs, external_pairs, data, cnffile,
                                    forced_arcs, forced_cliques, pset_acyc,
                                    debug=True)
        enc.encode_sat(treewidth)
    # if debug: print("encoding done")
    if debug: print(f"maxsat stats: {len(enc.vars)} vars, {enc.num_clauses} clauses")
    # sys.exit()
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
                print(f"error while running uwrmaxsat on {errcnf}"
                      f"\nrc: {err.returncode}, check {errfilename}")
                raise NoSolutionException("nonzero returncode")
        else:
            print(f"error while running uwrmaxsat on {errcnf}"
                  f"\nrc: {err.returncode}, no stdout captured")
            raise NoSolutionException("nonzero returncode")
    else:  # if no error while maxsat solving
        runtime = now() - start
        model = read_model(output)
        dec = ConbnDecoder(enc, treewidth, model, input_file, internal_constraints)
        bn = dec.get_bn()
        satisfied, total = bn.total_satisfied_constraints(), bn.total_constraints
        print(f"satisfied: {satisfied}/{total} constraints")
        assert satisfied == total, "maxsat solution does not satisfy all" \
                                   f" constraints (only {satisfied}/{total})"
        return bn


def solve_constrained(instance, treewidth, percent, seed, timeout):
    JKLDIR = "../input"
    CONDIR = "../input/constraint-files"

    jklfile = f"{instance}-5000.jkl"
    jklpath = os.path.join(JKLDIR, jklfile)
    bndata = read_bn(jklpath)
    # resfile = "../past-work/blip-publish/tmp.res"

    confile = f"{instance}-{percent}-{seed}.con"
    conpath = os.path.join(CONDIR, confile)
    constraints = read_constraints(conpath, int)

    constraints["undanc"].clear()
    # todo: verify unsat instance for sachs 10-3
    # constraints["posanc"].append((5, 0))

    incoming_arcs: IncomingArcs = ([], [])
    path_pairs: PathPairs = [({5, 6, 4}, {3, 0})]

    # constraints["posanc"] = [(3,4)]
    # constraints["neganc"].clear()

    print(f"working on {jklfile} + {confile}, timeout: {timeout}")
    bn = solve_conbn(bndata, treewidth, jklpath, constraints, incoming_arcs,
                     path_pairs, timeout=timeout, debug=True)
    print("bn score:", bn.score)
    # print("bn psets:")
    # print_bn(bn)
    print()
    return bn


if __name__ == '__main__':
    # instance, treewidth = "cancer", 3
    # for (percent, seed) in [(5,5), (10,3), (10,4), (10,5), (15,3), (15,4), (15,5), (20,2), (20,3), (20,5)]:
    instance, treewidth = "sachs", 4
    percent, seed = 10, 3
    bn = solve_constrained(instance, treewidth, percent, seed, timeout=10)
    from pprint import pprint
    pprint(list(bn.dag.edges))
    # instance, treewidth = "alarm", 5
    # percent, seed = 5, 1
    # for percent in (5, 10, 15, 20):
    #     for seed in (1, 2, 3, 4, 5):
    #         try:
    #             solve_constrained(instance, treewidth, percent, seed, timeout=7+percent//5)
            # except NoSolutionException:
            #     print("no solution, continuing\n")
            # print()


