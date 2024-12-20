"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('../')

############## Task 1
G = nx.read_edgelist('datasets/CA-HepTh.txt', delimiter='\t')
n_nodes = G.number_of_nodes()
print(f'Number of nodes: {n_nodes}')
n_edges = G.number_of_edges()
print(f'Number of edges: {n_edges}')

############## Task 2
cc = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print(f'Number of connected components: {len(cc)}')
largest_c_subgraph = G.subgraph(cc[0])
n_nodes_largest = largest_c_subgraph.number_of_nodes()
print(f'Number of nodes of the largest connected component: {n_nodes_largest}, {n_nodes_largest/n_nodes} of the total')
n_edges_largest = largest_c_subgraph.number_of_edges()
print(f'Number of edges of the largest connected component: {n_edges_largest}, {n_edges_largest/n_edges} of the total')

'''
Output:

Number of nodes: 9877
Number of edges: 25998
Number of connected components: 429
Number of nodes of the largest connected component: 8638, 0.8745570517363572 of the total
Number of edges of the largest connected component: 24827, 0.9549580736979768 of the total
'''