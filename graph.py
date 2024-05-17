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
        G = SPG(graph, edge_labels)
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
        self.colors = ['pink' if node.startswith('S') else 'lightblue' for node in self.graph]
        self.edge_labels = dict([(key, self.edge_labels[key]) for key in self.edge_labels.keys() if
                                 key[0] not in res and key[1] not in res])
        print(self.edge_labels)
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

class Diamond:
    def __init__(self, A, a1, a2, B, b1, b2):
        self.A = A
        self.a1 = a1
        self.a2 = a2
        self.B = B
        self.b1 = b1
        self.b2 = b2
        self.P1 = (A**2 - a1**2) // 4  # = (B**2 - b1**2) // 4
        self.P2 = (A**2 - a2**2) // 4  # = (B**2 - b2**2) // 4

    @staticmethod
    def from_SSPP(S1, S2, P1, P2):
        return Diamond(A=S1,
                       a1=math.sqrt(S1 ** 2 - 4 * P1),
                       a2=math.sqrt(S1 ** 2 - 4 * P2),
                       B=S2,
                       b1=math.sqrt(S2 ** 2 - 4 * P1),
                       b2=math.sqrt(S2 ** 2 - 4 * P2),
                       )

    @staticmethod
    def from_4nodes(dia):
        return Diamond.from_SSPP(*[int(node[1:]) for node in dia])

    def __repr__(self):
        return f"(A={self.A}, a1={self.a1}, a2={self.a2}, B={self.B}, b1={self.b1}, b2={self.b2}, P1={self.P1}, P2={self.P2})"

def induced_sum_diamond_diagram(maximum: int):
    graph = nx.Graph()
    for s_edge in [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 2)[0]]:
        graph.add_edge(*s_edge)
    return graph


def diamond_upper_curve(a_plus_b: int):
    ...

# {s = x + y, d = x - y}, Tsd represents this kind of replacement
def diamond_sum_nodes_Tsd(maximum: int):
    xs_s = [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 0)[0]]
    xs_Tsd_s = [[p[0] + p[1], p[1] - p[0]] for p in xs_s]
    return xs_Tsd_s


def estimate_epsilon(maximum: int):
    # to find out the constant epsilon such that the line "sqrt(D) = sqrt(S) + epsilon" bounds the scatter
    xs_s = [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 0)[0]]
    epsilon = maximum
    result_p = ()
    for dia in substructrue_diamond(maximum, 0)[0]:
        # print(epsilon)
        s0 = int(dia[0][1:])
        s1 = int(dia[1][1:])
        if (next_epsilon := math.sqrt(int(s0 + s1) - math.sqrt(s0 + s1))) < epsilon:
            epsilon = next_epsilon
            result_p = dia
    return epsilon, Diamond.from_4nodes(result_p)


def diamond_sum_nodes_AB(maximum: int):
    return [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 0)[0]]


def diamond_sum_nodes_sqTsd(maximum: int):
    xs_s = [[int(p[1:]) for p in ps[0:2]] for ps in substructrue_diamond(maximum, 0)[0]]
    xs_sqTsd_s = [[math.sqrt(p[0] + p[1]), math.sqrt(p[1] - p[0])] for p in xs_s]
    return xs_sqTsd_s


def diamond_scatter_AB(maximum: int, color: str = 'red'):
    xas_AB_s = np.array(diamond_sum_nodes_AB(maximum))
    ps_AB_s = xas_AB_s.transpose()
    plt.scatter(ps_AB_s[0], ps_AB_s[1], c=color)


def diamond_scatter_Tsd(maximum: int, color: str = 'red'):
    x = np.linspace(0, 4*maximum, 100)
    y = np.array([(-math.sqrt(xi) + 2*math.sqrt(maximum)) ** 2 for xi in x])
    xas_Tsd_s = np.array(diamond_sum_nodes_Tsd(maximum))
    ps_Tsd_s = xas_Tsd_s.transpose()
    plt.scatter(ps_Tsd_s[0], ps_Tsd_s[1], c=color)
    plt.plot(x,y)


def diamond_scatter_sqTsd(maximum: int, color: str = 'red'):
    xas_sqTsd_s = np.array(diamond_sum_nodes_sqTsd(maximum))
    ps_sqTsd_s = xas_sqTsd_s.transpose()
    plt.scatter(ps_sqTsd_s[0], ps_sqTsd_s[1], c=color)


def diamond_sqS_vs_sqD_scatter(maximum: int, scatter_color: str = 'red'):
    sqS = np.linspace(0, 2 * math.sqrt(maximum), 100)
    sqD = np.array([-sqS_i + 2 * math.sqrt(maximum) for sqS_i in sqS])
    diamond_scatter_sqTsd(maximum, color=scatter_color)
    plt.plot(sqS, sqD)


def diamond_scatter_sd(maximum: int, color: str = 'red'):
    xas_Tsd_s = np.array(diamond_sum_nodes_Tsd(maximum))
    ps_Tsd_s = xas_Tsd_s.transpose()
    plt.scatter(ps_Tsd_s[0], ps_Tsd_s[0] * ps_Tsd_s[1] / maximum, c=color)


def diamond_scatter_Tsd_parity(maximum: int, AmB_bound: int):
    xs_Tsd_s = [p for p in diamond_sum_nodes_Tsd(maximum) if p[1] <= AmB_bound]
    ps_Tsd_s_same = np.array([p for p in xs_Tsd_s if p[0] % 2 == 0]).transpose()
    ps_Tsd_s_diff = np.array([p for p in xs_Tsd_s if p[0] % 2 == 1]).transpose()
    plt.scatter(ps_Tsd_s_same[0], ps_Tsd_s_same[1], c='red')
    plt.scatter(ps_Tsd_s_diff[0], ps_Tsd_s_diff[1], c='blue')


def diamond_scatter_Tsd_parity_sqrt(maximum: int, AmB_bound: int):
    xs_Tsd_s = [p for p in diamond_sum_nodes_Tsd(maximum) if p[1] <= AmB_bound]
    ps_Tsd_s_same = np.array([[math.sqrt(p[0]), math.sqrt(p[1])] for p in xs_Tsd_s if p[0] % 2 == 0]).transpose()
    ps_Tsd_s_diff = np.array([[math.sqrt(p[0]), math.sqrt(p[1])] for p in xs_Tsd_s if p[0] % 2 == 1]).transpose()
    plt.scatter(ps_Tsd_s_same[0], ps_Tsd_s_same[1], c='red')
    plt.scatter(ps_Tsd_s_diff[0], ps_Tsd_s_diff[1], c='blue')


def density_of_tails(maximum: int, AmB_bound: int):
    all_diamonds = [p for p in substructrue_diamond(maximum, 0)[0] if int(p[0][1:]) - int(p[1][1:]) <= AmB_bound]
    diamond_fishes = [p for p in substructrue_diamond(maximum, 0)[0] if (int(p[0][1:]) - int(p[1][1:])) % 2 == 0]
    return len(diamond_fishes) / len(all_diamonds)
    # xs = [p for p in diamond_sum_nodes_Tsd(maximum) if p[1] <= AmB_bound]
    # xs_same = [p for p in xs if p[0] % 2 == 0]
    # return len(xs_same)/len(xs)


def diamond_minimize_B(maximum: int):
    return max(*diamond_sum_nodes_Tsd(maximum), key=lambda p: p[1])


def eight_factors(ApB: int, AmB: int):
    ...


def life_time_of_sum_node(maximum: int):
    G = SPG.by_max(maximum)
    life_sums = [0 for _ in range(2 * maximum + 1)]
    current_turn = 0
    while leaves := G.rot():
        for leaf in leaves:
            if sum_matched := re.match(r'S(\d*)', leaf):
                print(leaf)
                life_sums[int(sum_matched[1])] = current_turn
        current_turn += 1
    for loop_node in G.graph:
        if sum_matched := re.match(r'S(\d*)', loop_node):
            print(loop_node, "!")
            life_sums[int(sum_matched[1])] = -1
    return life_sums


def minimized_N_to_make_sum_node_immortal(maximum_N: int):
    minN_of_sums = [0 for _ in range(2 * maximum_N + 1)]
    visited_sums = set()
    for maximum in range(maximum_N):
        G = SPG.by_max(maximum)
        while G.rot():
            pass
        for loop_node in G.graph:
            if loop_node not in visited_sums \
                    and (sum_matched := re.match(r'S(\d*)', loop_node)):
                visited_sums.add(loop_node)
                minN_of_sums[int(sum_matched[1])] = maximum - ...
    return minN_of_sums


#
# class Diamond:
#     def __init__(self, A, B, a1, a2, b1, b2):
#         self.A = A      # A is the larger sum node
#         self.B = B      # B is the smaller sum node
#         self.a1 = a1
#         self.a2 = a2
#         self.b1 = b1
#         self.b2 = b2
#         self.P1 = p1    # p1 = (A/2)^2 - (a1/2)^2 = (B/2)^2 - (b1/2)^2
#         self.P2 = p2    # p2 = (A/2)^2 - (a2/2)^2 = (B/2)^2 - (b2/2)^2
#
#
#     def from_sd_point(self, ): # s = A + B, d = A - B


@timing
def main():
    # ===================================
    N = 200
    Epsilon = math.sqrt(30) - 2

    diamond_scatter_AB(250, color='blue')
    diamond_scatter_AB(200, color='green')
    diamond_scatter_AB(150, color='yellow')
    diamond_scatter_AB(100, color='orange')

    # ===================================

    # ===================================
    # N = 200
    # Epsilon = math.sqrt(30) - 2
    # x = np.linspace(0, 4*N, 100)
    # y = np.array([(-math.sqrt(xi) + 2*math.sqrt(N)) ** 2 for xi in x])
    # y_left = np.array([(math.sqrt(xi) - Epsilon) ** 2 for xi in x])

    # diamond_scatter_Tsd(500, color='violet')
    # diamond_scatter_Tsd(250, color='blue')
    # diamond_scatter_Tsd(200, color='green')
    # diamond_scatter_Tsd(150, color='yellow')
    # diamond_scatter_Tsd(100, color='orange')
    # plt.plot(x, y)
    # plt.plot(x, y_left)
    # ===================================
    # print(estimate_epsilon(200))      # 3.4641016151377544
    # print(estimate_epsilon(300))     # 3.4641016151377535
    #  N=1000 -> 4.9520474982524485
    # ===================================
    # N = 200
    # Epsilon = 3.4641016151377535  # min(sqrt(A+B) - sqrt(A-B))
    # sqS = np.linspace(0, 2 * math.sqrt(N), 100)
    # sqD_left = sqS - Epsilon
    # plt.plot(sqS, sqD_left)
    # diamond_sqS_vs_sqD_scatter(250, scatter_color='blue')
    # diamond_sqS_vs_sqD_scatter(200, scatter_color='green')
    # diamond_sqS_vs_sqD_scatter(150, scatter_color='yellow')
    # diamond_sqS_vs_sqD_scatter(100, scatter_color='orange')

main()
plt.show()
