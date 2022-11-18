# !apt-get install python3-dev graphviz libgraphviz-dev pkg-config
# !pip install pygraphviz

#############################################################
# no edge labels, but runs faster and is clearer for larger n
#############################################################

import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import math

MAX = 98  # @param {type:"integer"}

# Building the graph
G = nx.Graph()
for i, j in it.combinations_with_replacement(range(2, MAX + 1), 2):
    sum_node = f'S{i + j}'  # 'S6' for i=2, j=3
    prod_node = f'P{i * j}'
    G.add_edge(sum_node, prod_node)

color_map = ['pink' if node.startswith('S') else 'lightblue' for node in G]
options = {
    'node_color': color_map,
    'node_size': 600,
    'font_size': 10,
    'width': .8,
    'with_labels': True,
}

plt.figure(1, figsize=(16, 6))  # Add 1, 2,.. for additional figures
nx.draw(G, nx.nx_agraph.graphviz_layout(G), **options)

# Removing the leaves
leaves = [node for node in G if G.degree(node) <= 1]
G.remove_nodes_from(leaves)

# Drawing the graph without leaves
color_map = ['pink' if node.startswith('S') else 'lightblue' for node in G]
options['node_color'] = color_map

plt.figure(2, figsize=(10, 6))
nx.draw(G, nx.nx_agraph.graphviz_layout(G), **options)

# leaves left after 4 levels of removal
i = 2
for i in range(2, 5):
    leaves = [node for node in G if G.degree(node) <= 1]
    G.remove_nodes_from(leaves)
    i += 1

print(leaves)

#################
# has edge labels
#################

import networkx as nx
import itertools as it
import matplotlib.pyplot as plt

MAX = 99  # @param {type:"integer"}

# Building the graph
G = nx.Graph()
edge_labels = {}
for i, j in it.combinations_with_replacement(range(2, MAX + 1), 2):
    edge = (f'S{i + j}', f'P{i * j}')
    edge_labels[edge] = f'{i}:{j}'
    G.add_edge(*edge, _=(i, j))

color_map = ['pink' if node.startswith('S') else 'lightblue' for node in G]
options = {
    'node_color': color_map,
    'node_size': 600,
    'font_size': 10,
    'width': .8,
    'with_labels': True,
}

pos = nx.nx_agraph.graphviz_layout(G)

plt.figure(1, figsize=(25, 10))  # Add 1, 2,.. for additional figures
nx.draw(G, pos, **options)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Removing the leaves
leaves = [node for node in G if G.degree(node) <= 1]
G.remove_nodes_from(leaves)

# Drawing the graph without leaves
color_map = ['pink' if node.startswith('S') else 'lightblue' for node in G]
options['node_color'] = color_map

plt.figure(2, figsize=(15, 10))
nx.draw(G, pos, **options)

nx.draw_networkx_edge_labels(G, pos, edge_labels={x: edge_labels[x] for x in G.edges})
plt.show()

#########################################################
# function that gives you the various partitions of a sum
#########################################################

import math


def sums(n, numb):
    answer = set()
    p1 = math.floor(numb / 2)
    p2 = math.ceil(numb / 2)
    while p1 >= 2 and p1 <= n:
        answer.add(tuple((p1, p2)))
        p1 = p1 - 1
        p2 = p2 + 1
    return answer


sums(99, 168)

# gives products from pairs of numbers
tuple(a * b for a, b in sums(10, 6))

###################################
# longest chain length
###################################

import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import math


def max_chain(MIN, MAX):
    for n in range(MIN, MAX + 1):
        # Building the graph
        G = nx.Graph()
        for i, j in it.combinations_with_replacement(range(2, n + 1), 2):
            sum_node = f'S{i + j}'  # 'S6' for i=2, j=3
            prod_node = f'P{i * j}'
            G.add_edge(sum_node, prod_node)

        # finding the longest chain length

        leaves = [node for node in G if G.degree(node) <= 1]
        j = -1
        while len(leaves) > 0:
            leaves = [node for node in G if G.degree(node) <= 1]
            G.remove_nodes_from(leaves)
            j += 1
            if len(leaves) == 0 and j > 7:
                print(n, j)
                print()


max_chain(10, 301)
