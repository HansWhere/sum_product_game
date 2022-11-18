import networkx as nx
from itertools import combinations_with_replacement, chain
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
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @staticmethod
    def by_max(maximum: int):
        return SPG(nx.Graph(
            (f'S{i + j}', f'P{i * j}')
            for i, j in combinations_with_replacement(range(2, maximum+1), 2)
        ))

    def copy(self) -> 'SPG':
        return SPG(self.graph.copy())

    def plot(self, num: int = 1, figsize: Tuple[int, int] = (6, 6)):
        options = {
            'node_color': ['pink' if node.startswith('S') else 'lightblue' for node in self.graph],
            'node_size': 600,
            'font_size': 10,
            'width': .8,
            'with_labels': True,
        }
        plt.figure(num, figsize)
        nx.draw(self.graph, nx.nx_agraph.graphviz_layout(self.graph), **options)

    def leaves(self) -> List:
        return [i for i in self.graph if self.graph.degree(i) <= 1]

    def rot(self) -> List:
        self.graph.remove_nodes_from(res := self.leaves())
        return res

    def succ(self) -> 'SPG':
        ret = self.graph.copy()
        ret.remove_nodes_from(self.leaves())
        return SPG(ret)

def game_life(maximum: int) -> int:
    G = SPG.by_max(maximum)
    count = 0
    while G.rot():
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
            if 'chains' in modes or 'dropped_nodes' in modes:
                dropped_nodes = chain(dropped_nodes, iter(last_leaves))
        else:
            res = []
            if 'game_life' in modes:
                res.append(life_count)
            if 'last_leaves' in modes:
                res.append(last_leaves)
            if 'dropped_nodes' in modes:
                res.append(dropped_nodes)
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

@timing
def main():
    xs = [(n,SPG_stats(n,('game_life','last_leaves'))) for n in range(2, 100)]
    print(xs)
    print(f'the longest game life is {max(xs)}')
    SPG_stats(20,('chains',))[0].plot()
    print(list(SPG_stats(99,('dropped_nodes',))[0]))
main()
plt.show()