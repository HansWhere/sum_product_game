import networkx as nx
import itertools as it
import matplotlib.pyplot as plt

MAX = 10

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

plt.figure(1, (16, 6))  # Add 1, 2,.. for additional figures
nx.draw(G, nx.nx_agraph.graphviz_layout(G), **options)

# Removing the leaves
# leaves = list(filter(lambda n: G.degree(n) <= 1, G))
leaves = [node for node in G if G.degree(node) <= 1]
G.remove_nodes_from(leaves)

# Drawing the graph without leaves
color_map = ['pink' if node.startswith('S') else 'lightblue' for node in G]
options['node_color'] = color_map

plt.figure(2, (6, 10))
nx.draw(G, nx.nx_agraph.graphviz_layout(G), **options)

# leaves left after 4 levels of removal
# i = 2
for i in range(2, 5):
    leaves = [node for node in G if G.degree(node) <= 1]
    G.remove_nodes_from(leaves)
    # i += 1

plt.show()

plt.show()