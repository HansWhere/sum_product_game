import math
import re

import numpy as np
import networkx as nx
from itertools import combinations_with_replacement, combinations, chain
import matplotlib.pyplot as plt
from typing import *
from time import time
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit

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

def deg_n_prod_nodes(maximum: int, deg: int, sum_lower_bound: int):
    G = SPG.by_max(maximum)
    counter = 0
    subgraph_edges = set()
    for node in G.graph:
        if G.graph.degree(node) == deg and re.match(r'P(\d*)', node) is not None:
            for neighbor in G.graph.neighbors(node):
                if neighbor < sum_lower_bound:
                    break
            else:
                counter += 1
                for edge in G.graph.edges(node):
                    subgraph_edges.add(edge)
    return counter, SPG(G.graph.edge_subgraph(subgraph_edges), {})

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
                            diamond_nodes.update([node, nn, neighbor1, neighbor2])
                            diamonds.append([node, nn, neighbor1, neighbor2])
    return diamonds, diamond_nodes

def substructrue_chains(maximum: int):
    G = SPG.by_max(maximum)
    chain_xs = []
    while leaves := G.rot():
        chain_xs += leaves
    return chain_xs

def induced_sum_diamond_diagram(maximum: int):
    graph = nx.Graph()
    for s_edge in [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 2)[0]]:
        graph.add_edge(*s_edge)
    return graph

def diamond_upper_curve(a_plus_b: int):
    ...

def diamond_sum_nodes_til(maximum: int):
    xs_s = [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 0)[0]]
    xs_til_s = [[p[0] + p[1], p[1] - p[0]] for p in xs_s]
    return xs_til_s

def diamond_scatter_til(maximum: int, color: str = 'red'):
    xas_til_s = np.array(diamond_sum_nodes_til(maximum))
    ps_til_s = xas_til_s.transpose()
    plt.scatter(ps_til_s[0], ps_til_s[1], c=color)

def diamond_scatter_til_parity(maximum: int, AmB_bound: int):
    xs_til_s = [p for p in diamond_sum_nodes_til(maximum) if p[1] <= AmB_bound]
    ps_til_s_same = np.array([p for p in xs_til_s if p[0] % 2 == 0]).transpose()
    ps_til_s_diff = np.array([p for p in xs_til_s if p[0] % 2 == 1]).transpose()
    plt.scatter(ps_til_s_same[0], ps_til_s_same[1], c='red')
    plt.scatter(ps_til_s_diff[0], ps_til_s_diff[1], c='blue')

def density_of_tails(maximum: int, AmB_bound: int):
    all_diamonds = [p for p in substructrue_diamond(maximum, 0)[0] if int(p[0][1:]) - int(p[1][1:]) <= AmB_bound]
    diamond_fishes = [p for p in substructrue_diamond(maximum, 0)[0] if (int(p[0][1:]) - int(p[1][1:])) % 2 == 0]
    return len(diamond_fishes) / len(all_diamonds)
    # xs = [p for p in diamond_sum_nodes_til(maximum) if p[1] <= AmB_bound]
    # xs_same = [p for p in xs if p[0] % 2 == 0]
    # return len(xs_same)/len(xs)

def diamond_minimize_B(maximum: int):
    return max(*diamond_sum_nodes_til(maximum), key=lambda p: p[1])

def eight_factors(ApB: int, AmB: int):
    ...

@timing
def main():
    N = 30
    # plt.figure(99,igsize=(6,6))
    # gN = SPG.by_max(N, highlights_cond=lambda p: p in substructrue_diamond(N,0)[1])
    # gN.plot(99)
    # all_diamonds = substructrue_diamond(N, 0)[0]
    # diamond_fishes = [p for p in substructrue_diamond(N, 0)[0] if (int(p[0][1:]) - int(p[1][1:])) % 2 == 0]
    # print(len(diamond_fishes)/len(all_diamonds))

    # xs = [len(substructrue_diamond(N, i)[0]) for i in range(4, 2*N)]
    # xs_sp = [[[int(p[1:]) for p in ps] for ps in substructrue_diamond(N, i)[0]] for i in range(4, 2 * N)]
    # xs_s = [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(N, 0)[0]]
    # xs_til_s = [[p[0]+p[1], p[1]-p[0]] for p in xs_s]

    # xs_til_s = [[int(ps[0][1:]),int(ps[1][1:])] for ps in substructrue_diamond(N, 0)[0]]

    # plt.figure(99, figsize=(6,6))
    # nas = np.arange(4, 2*N).reshape(-1,1)
    # xas = np.array(xs)
    # xas_s = np.array(xs_s)
    # xas_til_s = np.array(xs_til_s)
    # print(xas_s)
    # mid_index = len([1 for x in xs if x > math.floor(xas[0]/2)])
    # print(mid_index)
    # dxas = np.array(xs) - np.array(xs[1:]+[0])
    # plt.plot(nas, xas, 'go-', label='diamonds#', linewidth=2)

    # plt.figure(100, figsize=(6, 6))
    # plt.plot(nas, dxas, 'ro-', label='diff diamonds#', linewidth=2)

    #  fplt.figure(101,igsize=(6,6))
    # ps_s = xas_s.transpose()
    # ps_til_s = xas_til_s.transpose()
    # plt.scatter(ps_s[0], ps_s[1])
    # plt.scatter(ps_til_s[0], ps_til_s[1])

    # plt.figure(102, figsize=(6,6))
    # induce_diag = induced_sum_diamond_diagram(N)
    # nx.draw_networkx(induce_diag, nx.nx_agraph.graphviz_layout(induce_diag))

    # print(max(diamond_sum_nodes_til(150), key=lambda p: p[1]))
    # print([p for p in diamond_sum_nodes_til(100) if p[1] == 60])
    # print([p for p in diamond_sum_nodes_til(100) if p[0] == 126])

    # plt.figure(101, figsize=(6, 6))
    #
    # bound_slope = max((p[1]-4)/(p[0]-30) for p in diamond_sum_nodes_til(500) if p[0] != 30)
    # print(bound_slope)
    # x = np.linspace(30, 400, 100)
    # y = bound_slope * (x-30) + 4
    x = np.linspace(0, 400, 100)
    y = x - 4 * x ** (3/4)
    plt.plot(x, y)
    diamond_scatter_til_parity(200,400)
    print(density_of_tails(200,400))
    # diamond_scatter_til(500, color='violet')
    # diamond_scatter_til(250, color='blue')
    # diamond_scatter_til(200, color='green')
    # diamond_scatter_til(150, color='yellow')
    # diamond_scatter_til(100, color='orange')
    # diamond_scatter_til(50, color='red')
    # plt.plot(x, y)

    # N = 250
    # g300 = SPG.by_max(300)
    # print(diamond_minimize_B(300))
    # plt.figure(102, figsize=(6, 6))
    # plt.scatter(range(20,N), [diamond_minimize_B(i)[1] for i in range(20, N)])


main()
plt.show()