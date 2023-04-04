import math
import re

import networkx as nx
from itertools import combinations_with_replacement, combinations, chain
import matplotlib.pyplot as plt
from typing import *
from time import time


def timing(f):
    def timing_f(*args, **kwargs):
        start_time = time()
        result = f(*args, **kwargs)
        print("--- %s seconds ---" % (time() - start_time))
        return result
    return timing_f


class SPG:
    def __init__(self, graph: nx.Graph, edge_labels: dict):
        self.graph: nx.Graph = graph
        self.edge_labels: dict = edge_labels
        self.colors = ['pink' if node.startswith('S') else 'lightblue' for node in self.graph]

    @staticmethod
    def by_max(maximum: int, highlights_cond=None):
        graph = nx.Graph()
        edge_labels = {}
        for i, j in combinations_with_replacement(range(2, maximum + 1), 2):
            edge = (f'S{i + j}', f'P{i * j}')
            edge_labels[edge] = f'{i}:{j}'
            graph.add_edge(*edge, _=(i, j))
        G = SPG(graph,edge_labels)
        if highlights_cond is not None:
            for index, node in enumerate(G.graph):
                if highlights_cond(node):
                    G.colors[index] = 'blue' if re.match(r'P(\d*)', node) is not None else 'red'
        return G

    def copy(self) -> 'SPG':
        return SPG(self.graph.copy(), self.edge_labels.copy())

    def plot(self, num: int = 1, figsize: Tuple[int, int] = (6, 6), options: Dict = None):
        if options is None:
            options = {}
        current_options = {
            'node_color': self.colors,
            'node_size': 600,
            'font_size': 10,
            'width': .8,
            'with_labels': True,
        }
        current_options.update(options)
        plt.figure(num, figsize)
        pos = nx.nx_agraph.graphviz_layout(self.graph)
        nx.draw(self.graph, pos, **current_options)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=self.edge_labels)

    def leaves(self) -> List:
        return [i for i in self.graph if self.graph.degree(i) <= 1]

    def rot(self) -> List:
        self.graph.remove_nodes_from(res := self.leaves())
        return res

    def succ(self) -> 'SPG':
        ret = self.graph.copy()
        ret.remove_nodes_from(self.leaves())
        return SPG(ret, self.edge_labels)

    def game_life(self) -> int:
        count = 0
        g = self.copy()
        while g.rot():
            count += 1
        return count

def SPG_stats(maximum: int, modes=()) -> List:
    G = SPG.by_max(maximum)
    initial_G = G.copy()
    last_leaves = []
    dropped_nodes = iter(())
    life_count = 0
    while True:
        G.graph.remove_nodes_from(last_leaves)
        if leaves := G.leaves():
            last_leaves = leaves
        else:
            res = []
            if 'game_life' in modes:
                res.append(life_count)
            if 'last_leaves' in modes:
                res.append(last_leaves)
            if 'chains' in modes:
                chains = initial_G.copy()
                chains.graph.remove_nodes_from(G.graph)
                res.append(chains)
            if 'initial_graph' in modes:
                res.append(initial_G)
            if 'terminal_graph' in modes:
                res.append(G)
            return res
        life_count += 1

def longest_chain(maximum: int) -> SPG:
    last_leaves, chains = SPG_stats(maximum, ('last_leaves', 'chains'))
    nodes = nx.node_connected_component(chains.graph, last_leaves[0])
    chains.graph.remove_nodes_from([n for n in chains.graph if n not in nodes])
    return chains

def n_step_chains(maximum: int, step_upper: int, step_lower: int = 1) -> SPG:
    G = SPG.by_max(maximum)
    subgraph_nodes = set()
    for s_node in G.graph:
        if (s_re := re.match(r'S(\d*)', s_node)) is not None and (s_node_val := int(s_re.group(1))):
            p_nodes = G.graph.neighbors(s_node)
            p_exist = False
            for p_node_0, p_node_1 in combinations(p_nodes, 2):
                diff = abs(int(re.match(r'P(\d*)', p_node_0).group(1)) - int(re.match(r'P(\d*)', p_node_1).group(1)))
                print(diff)
                if step_upper >= diff >= step_lower:
                    subgraph_nodes.add(p_node_0)
                    subgraph_nodes.add(p_node_1)
                    p_exist = True
            if p_exist:
                subgraph_nodes.add(s_node)

    return SPG(G.graph.subgraph(subgraph_nodes), {})


def strict_n_step_chains(maximum: int, step_upper: int, step_lower: int = 1) -> SPG:
    G = SPG.by_max(maximum)
    subgraph_nodes = set()
    for s_node in G.graph:
        if (s_re := re.match(r'S(\d*)', s_node)) is not None and (s_node_val := int(s_re.group(1))):
            p_nodes = G.graph.neighbors(s_node)

            p_node_vals = [int(re.match(r'P(\d*)', p).group(1)) for p in p_nodes]
            if (diff := abs(max(p_node_vals) - min(p_node_vals))) <= step_upper and diff >= step_lower:
                subgraph_nodes.add(s_node)
                subgraph_nodes.update(G.graph.neighbors(s_node))

    return SPG(G.graph.subgraph(subgraph_nodes), {})

def deg_n_nodes(maximum: int, deg: int):
    G = SPG.by_max(maximum)
    counter = 0
    subgraph_edges = set()
    for node in G.graph:
        if G.graph.degree(node) == deg and re.match(r'P(\d*)', node) is not None:
            counter += 1
            for edge in G.graph.edges(node):
                subgraph_edges.add(edge)
    print(counter)
    return SPG(G.graph.edge_subgraph(subgraph_edges), {})

def deg_n_highlight(maximum: int, deg: int):
    G = SPG.by_max(maximum)
    for index, node in enumerate(G.graph):
        if G.graph.degree(node) == deg:
            G.colors[index] = 'blue' if re.match(r'P(\d*)', node) is not None else 'red'
    return G

def embed_graph(maximum: int, maximum_sub: int):
    G = SPG.by_max(maximum)
    G_sub = SPG.by_max(maximum_sub)
    for index, node in enumerate(G.graph):
        if node not in G_sub.graph:
            G.colors[index] = 'blue' if re.match(r'P(\d*)', node) is not None else 'red'
    return G

def unlooped_lifetime(node: str):
    maximum = 4
    while True:
        g = SPG.by_max(maximum)
        while res := g.rot():
            if node in res:
                return maximum
        maximum += 1

def substructrue_diamond(maximum: int, sum_lower_bound: int):
    G = SPG.by_max(maximum)
    diamond_nodes = set()
    diamond_leading_nodes = set()
    diamonds = []
    for index, node in enumerate(G.graph):
        if node not in diamond_leading_nodes and (sum_matched := re.match(r'S(\d*)', node)):
            diamond_leading_nodes.add(node)
            if int(sum_matched[1]) >= sum_lower_bound:
                for neighbor1, neighbor2 in combinations(G.graph.neighbors(node), 2):
                    for nn in G.graph.neighbors(neighbor1):
                        if nn not in diamond_leading_nodes and nn in G.graph.neighbors(neighbor2) \
                                and int(re.match(r'S(\d*)', nn)[1]) >= sum_lower_bound:
                            diamond_nodes.update([node, neighbor1, neighbor2, nn])
                            diamonds.append([node, neighbor1, neighbor2, nn])
    return diamonds, diamond_nodes

@timing
def main():
    # xs = [(n,SPG_stats(n,('game_life','last_leaves'))) for n in range(2, 1000)]
    # print(xs)
    # xs = [x[0] for n in range(10, 1000) if not ((x := (n, SPG_stats(n, ('game_life',))))[1][0] % 2 or x[1][0] == 6)]
    # print(xs)
    # print(f'the longest game life is {max(xs)}')
    # # SPG_stats(20,('chains',))[0].plot()
    # ll98 = SPG_stats(98, ('last_leaves',))[0]
    # lc98 = longest_chain(98)
    # lc98.plot(options={'node_color': ['red' if node in ll98 else 'pink' if node.startswith('S') else 'lightblue' for node in lc98.graph]})
    # ll99 = SPG_stats(99, ('last_leaves',))[0]
    # lc99 = longest_chain(99)
    # lc99.plot(2, options={'node_color': ['red' if node in ll99 else 'pink' if node.startswith('S') else 'lightblue' for node in lc99.graph]})
    # G1 = deg_n_highlight(20, 3)
    # G1_sub = deg_n_nodes(20, 3)
    # G1.plot(4)
    # G1_sub.plot(5)

    # G2 = embed_graph(26, 19)
    # G3 = embed_graph(27, 19)
    # G4 = embed_graph(28, 19)
    # G2.plot(6)
    # G3.plot(7)
    # G4.plot(8)

    # print(nlife := unlooped_lifetime('S10'))
    # G5 = SPG.by_max(nlife, lambda node: node == 'S10')
    # G5.plot(9)

    n = 50
    xs = [len(substructrue_diamond(50, i)[0]) for i in range(4, n)]
    ys = [math.log(d/(n-i/2), n) for i in range(4, 2*n) if (d := len(substructrue_diamond(n, i)[0])) > 0]
    print(ys)
    # G6 = SPG.by_max(20, lambda node: node in substructrue_diamond(20, 20)[1])
    # G6.plot(10)


main()
plt.show()