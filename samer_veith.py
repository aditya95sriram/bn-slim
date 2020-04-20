#!/bin/python3.6
# author: André Schidler

from collections import defaultdict


class SelfNamingDict(defaultdict):
    def __init__(self, cb, name=None):
        super().__init__()
        self.name = name
        self.callback = cb

    def __missing__(self, key):
        if self.name is None:
            val = self.callback()
        else:
            val = self.callback(self.name.format(key))

        self[key] = val
        return val


class SvEncoding:
    def __init__(self, stream, g):
        self.g = g
        self.ord = None
        self.arc = None
        self.vars = [None]  # None to avoid giving any var 0
        self.var_literal = None
        self.clause_literal = "{} 0"
        self.clause_literal_fun = lambda x: self.vars[x] if x > 0 else f"-{self.vars[-x]}"
        self.stream = stream
        self.nodes = list(g.nodes)
        self.node_lookup = {self.nodes[n]: n for n in range(0, len(self.nodes))}
        self.node_reverse_lookup = {n: self.nodes[n] for n in range(0, len(self.nodes))}
        self.num_clauses = 0

    def _add_var(self, nm=None):
        self.vars.append(nm if nm is not None else str(len(self.vars)))
        if self.var_literal is not None:
            self.stream.write(self.var_literal.format(self.vars[len(self.vars) - 1]))
            self.stream.write("\n")

        return len(self.vars) - 1

    def _add_clause(self, *args):
        self.stream.write(self.clause_literal.format(' '.join((self.clause_literal_fun(x) for x in args))))
        self.stream.write("\n")
        self.num_clauses += 1

    def _ord(self, i, j):
        if i < j:
            return self.ord[i][j]
        else:
            return -1 * self.ord[j][i]

    def encode(self):
        for i in range(0, len(self.g.nodes)):
            # No self loops
            self._add_clause(-self.arc[i][i])

            for j in range(0, len(self.g.nodes)):
                if i == j:
                    continue
                for l in range(0, len(self.g.nodes)):
                    if i == l or j == l:
                        continue

                    # Transitivity of ordering
                    self._add_clause(-self._ord(i, j), -self._ord(j, l), self._ord(i, l))

                    # Additional edges due to linear ordering
                    if j < l:
                        # ord instead of _ord is on purpose in both clauses
                        # TODO: This is necessary in the non-improved encoding...
                        # self._add_clause(-self.arc[i][j], -self.arc[i][l], -self.ord[j][l], self.arc[j][l])
                        # self._add_clause(-self.arc[i][j], -self.arc[i][l], self.ord[j][l], self.arc[l][j])

                        # Redundant, but speeds up solving
                        self._add_clause(-self.arc[i][j], -self.arc[i][l], self.arc[j][l], self.arc[l][j])

        # Encode edges
        for u, v in self.g.edges:
            u = self.node_lookup[u]
            v = self.node_lookup[v]

            if u > v:
                u, v = v, u
            self._add_clause(-self.ord[u][v], self.arc[u][v])
            self._add_clause(self.ord[u][v], self.arc[v][u])

    def encode_smt(self, g, stream, lb=0, ub=0):
        self.ord = {x: SelfNamingDict(lambda x: self._add_var(x), f"ord_{x}_{{}}") for x in range(0, len(g.nodes))}
        self.arc = {x: SelfNamingDict(lambda x: self._add_var(x), f"arc_{x}_{{}}") for x in range(0, len(g.nodes))}
        self.stream.write("(set-option :produce-models true)")
        self.var_literal = "(declare-const {} Bool)"
        self.clause_literal = "(assert (or {}))"
        self.clause_literal_fun = lambda x: self.vars[x] if x > 0 else f"(not {self.vars[-x]})"

        stream.write("(declare-const m Int)\n")
        stream.write("(assert (>= m 1))\n")
        if lb > 0:
            stream.write(f"(assert (>= m {lb}))\n")
        if ub > 0:
            stream.write(f"(assert (<= m {ub}))\n")

        self.encode()
        #self.encode_cardinality_smt(ub)
        self.encode_cardinality_sat(ub, self.arc)

        # stream.write("(minimize m)\n")
        stream.write("(check-sat)\n")
        stream.write("(get-model)\n")

    def encode_sat(self, target, cardinality=True):
        # We do not know the exact number of variables and clauses beforehand. Leave a placeholder to change afterwards
        # equals to ord + arc + ctr
        estimated_vars = 2 * len(self.g.nodes) * len(self.g.nodes) + len(self.g.nodes) * len(self.g.nodes) * target
        # This is way too much, but better too many than too few: m * n^4 * 100
        estimated_clauses = 100 * len(self.g.edges) * len(self.g.nodes) \
                            * len(self.g.nodes) * len(self.g.nodes) * len(self.g.nodes)

        placeholder = f"p cnf {estimated_vars} {estimated_clauses}"
        self.stream.write(placeholder)
        self.stream.write("\n")

        # Init variable references
        self.ord = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, len(self.g.nodes))}
        self.arc = {x: SelfNamingDict(lambda: self._add_var()) for x in range(0, len(self.g.nodes))}

        # Ensure that ord vars are first assigned, so we know the position
        for i in range(0, len(self.g.nodes)):
            for j in range(i+1, len(self.g.nodes)):
                self._ord(i, j)

        self.encode()
        if cardinality:
            self.encode_cardinality_sat(target, self.arc)

        header = f"p cnf {len(self.vars) - 1} {self.num_clauses}"
        self.stream.seek(0)
        self.stream.write(header)
        for _ in range(len(header), len(placeholder)):
            self.stream.write(" ")

    def encode_cardinality_smt(self, ub):
        for i in range(0, len(self.g.nodes)):
            vars = []
            for j in range(0, len(self.g.nodes)):
                if i == j:
                    continue

                arcvar = self.vars[self.arc[i][j]]
                # stream.write(f"(declare-const w_{i}_{j} Int)\n")
                # stream.write(f"(assert (=> {arcvar} (= w_{i}_{j} 1)))\n")
                # stream.write(f"(assert (=> (not {arcvar}) (= w_{i}_{j} 0)))\n")
                #vars.append(f"(ite {arcvar} 1 0)")
                vars.append(arcvar)
                self.stream.write(f"(assert-soft (not {arcvar}) :id goal{i})\n")
            self.stream.write(f"(assert (<= goal{i} m))\n")

            #self.stream.write("(assert ((_ at-most {ub}) {weights}))\n".format(ub=ub, weights=" ".join(vars)))

    def encode_cardinality_sat(self, bound, variables):
        """Enforces cardinality constraints. Cardinality of 2-D structure variables must not exceed bound"""
        # Counter works like this: ctr[i][j][0] states that an arc from i to j exists
        # These are then summed up incrementally edge by edge

        # Define counter variables ctr[i][j][l] with 1 <= i <= n, 1 <= j < n, 1 <= l <= min(j, bound)
        ctr = [[[self._add_var()
                 for _ in range(0, min(j, bound))]
                # j has range 0 to n-1. use 1 to n, otherwise the innermost number of elements is wrong
                for j in range(1, len(variables[0]))]
               for _ in range(0, len(variables))]

        for i in range(0, len(variables)):
            for j in range(1, len(variables[i]) - 1):
                # Ensure that the counter never decrements, i.e. ensure carry over
                for ln in range(0, min(len(ctr[i][j-1]), bound)):
                    self._add_clause(-ctr[i][j - 1][ln], ctr[i][j][ln])

                # Increment counter for each arc
                for ln in range(1, min(len(ctr[i][j]), bound)):
                    self._add_clause(-variables[i][j], -ctr[i][j-1][ln-1], ctr[i][j][ln])

        # Ensure that counter is initialized on the first arc
        for i in range(0, len(variables)):
            for j in range(0, len(variables[i]) - 1):
                self._add_clause(-variables[i][j], ctr[i][j][0])

        # Conflict if target is exceeded
        for i in range(0, len(variables)):
            for j in range(bound, len(variables[i])):
                # Since we start to count from 0, bound - 2
                self._add_clause(-variables[i][j], -ctr[i][j-1][bound - 1])


if __name__ == '__main__':
    import networkx as nx

    g1 = nx.cycle_graph(5)
    g1.add_edge(0,2)
    with open("temp.cnf", "w") as f:
        sve = SvEncoding(f, g1)
        sve.encode_sat(2, cardinality=True)