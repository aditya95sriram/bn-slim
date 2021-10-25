#!/bin/python3.6

import re
from typing import *
from collections import defaultdict
import numpy as np
import networkx as nx
from itertools import combinations


class AutoIncrementingDict(dict):
    def __getitem__(self, item):
        if super().__contains__(item):
            return super().__getitem__(item)
        else:
            val = len(self)
            super().__setitem__(item, val)
            return val


class BNNode:
    def __init__(self, parents: list, name: str, states: list):
        self.parents = parents
        self.name = name
        self.states = states


class BNTable:
    def __init__(self, child, parents: list, table: np.ndarray):
        self.child = child
        self.parents = parents
        self.table = table


Structure = Dict[int, BNNode]
Parameters = Dict[int, BNTable]


def parse_space_sep_strings(s):
    return s.replace('"', "").split()


def parse_nd_array(s):
    if "(" not in s:
        return list(map(float, s.split()))
    ar = []
    captured = ""
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
            if depth == 1: continue  # skip outermost level bracket
        elif c == ")":
            depth -= 1
        if depth == 0 and captured.strip():
            ar.append(parse_nd_array(captured))
            captured = ""
        else:
            captured += c
    return ar


def array2string(array):
    #return str(np.around(array, 6)).replace("[", " ").replace("]", " ")
    #return " ".join("%.6f"%el for el in np.flatten(array))
    width = array.shape[-1]
    print_array = np.reshape(array, (-1, width), order='C')
    return "\n".join(" ".join("%.6f"%el for el in row) for row in print_array)


class BNIO:
    def __init__(self, filename: str, elim_order: Optional[List[str]] = None):
        self.filename = filename
        with open(self.filename) as infile:
            self.contents = infile.read()
        if elim_order is None or len(elim_order) == 0:
            self.nodemap = AutoIncrementingDict()
        else:
            self.nodemap = {name: idx for idx, name in enumerate(elim_order)}
        self.noderevmap = dict()
        self.structure: Structure = self.read_structure()
        self.parameters: Parameters = self.read_parameters()
        self.copy_parent_data()

    def read_net(self):
        return self.read_structure(), self.read_parameters()

    def read_structure(self):
        nodepatn = re.compile(r"node\s*(?P<name>\w+)\s*{(?P<body>.*?)}", flags=re.DOTALL)
        statepatn = re.compile(r"states\s*=\s*\((?P<states>.*?)\)", re.DOTALL)
        parpatn = re.compile(r"parents\s*=\s*\((?P<parents>.*?)\)", re.DOTALL)
        structure: Structure = dict()
        for match in nodepatn.finditer(self.contents):
            name = match.group('name')
            nodeid = self.nodemap[name]
            nodedata = match.group('body')
            statestr = statepatn.search(nodedata).group('states')
            states = parse_space_sep_strings(statestr)
            parentstr = parpatn.search(nodedata)
            if parentstr is None:
                parents = None
            else:
                parentstr = parentstr.group('parents')
                parent_names = parse_space_sep_strings(parentstr)
                parents = [self.nodemap[p] for p in parent_names]
            structure[nodeid] = BNNode(parents, name, states)
        self.noderevmap = {nodeid: nodename for nodename, nodeid in self.nodemap.items()}
        return structure

    def read_parameters(self):
        probpatn = re.compile(r"potential\s*\(\s*(?P<child>\w+)\s*(?:\|(?P<parents>[^)]*))?\)\s*{(?P<body>.*?)}",
                              re.DOTALL)
        tablepatn = re.compile(r"data\s*=\s*\((?P<table>.*?)\)\s*;", re.DOTALL)
        parameters: Parameters = dict()
        for match in probpatn.finditer(self.contents):
            child = match.group('child')
            parentstr = match.group('parents')
            if parentstr is not None:
                parents = [self.nodemap[p] for p in parentstr.split()]
            else:
                parents = []
            probdata = match.group('body')
            tablestr = tablepatn.search(probdata).group('table')
            table = np.array(parse_nd_array(tablestr))
            parameters[self.nodemap[child]] = BNTable(child, parents, table)
        return parameters

    def copy_parent_data(self):
        for node, data in self.parameters.items():
            self.structure[node].parents = data.parents

    def write_uai(self, filename: str):
        numnodes = len(self.structure)
        with open(filename, 'w') as outfile:
            outfile.write("BAYES\n")
            outfile.write(f"{numnodes}\n")
            numstates = [len(self.structure[i].states) for i in range(numnodes)]
            outfile.write(" ".join(map(str, numstates)) + "\n")
            outfile.write(f"{numnodes}\n")
            for nodeid in range(numnodes):
                parents = self.parameters[nodeid].parents
                cliquestr = [len(parents)+1] + parents + [nodeid]
                outfile.write(" ".join(map(str, cliquestr)) + "\n")
            outfile.write("\n")

            for nodeid in range(numnodes):
                table = self.parameters[nodeid].table
                outfile.write(f"{table.size}\n")
                outfile.write(f"{array2string(table)}\n\n")

    def reorder(self, new_order: list):
        raise NotImplementedError


    def get_moral_graph(self):
        M = nx.Graph()
        for node, data in self.structure.items():
            for par1 in data.parents:
                M.add_edge(node, par1)
            for par1, par2 in combinations(data.parents, 2):
                M.add_edge(par1, par2)
        return M



def to_uai(infile: str, outfile: str, elim_order: Optional[List[str]] = None):
    bn = BNIO(infile, elim_order)
    bn.write_uai(outfile)


if __name__ == '__main__':
    to_uai("specimen2.net", "test-bnio.uai")
