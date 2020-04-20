#!/bin/python3.6

# external
import networkx as nx

# internal
from samer_veith import SvEncoding


class TwbnEncoding(SvEncoding):
    def __init__(self, filename: str, stream):
        self.filename = filename
        super().__init__(stream, nx.Graph())

    def encode(self):
        # encode bn
        # construct graph for treewidth computation
        self.encode_sat()
        pass


if __name__ == '__main__':
    with open("temp.cnf", "w") as f:
        enc = TwbnEncoding("../past-work/blip-publish/data/child-5000.jkl", f)
