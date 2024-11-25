"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans

G = nx.read_edgelist('../datasets/CA-HepTh.txt', delimiter='\t')

############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k, d=10):
    # 1. A is the adjacency matrix of the graph
    A = nx.adjacency_matrix(G)
    # 2. Compute Laplacian matrix
    n_nodes = A.shape[0]
    I = eye(n_nodes)
    D_inverse = diags(1 / A.sum(axis=0))
    L_rw = I - D_inverse @ A
    # 3. Eigenvalue decomposition
    eigenvalues, eigenvectors = eigs(L_rw, d, which='SM')
    idx = np.argsort(eigenvalues)
    U = np.vstack(eigenvectors[:, idx]).real
    # 4. Apply kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    G_nodes = list(G) 
    clustering = {G_nodes[i]:kmeans.labels_[i] for i in range(n_nodes)}

    return clustering

############## Task 4
cc = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
Giant = G.subgraph(cc[0])
clustering = spectral_clustering(Giant, 50)
# print(clustering)

############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    clusters = {}
    for key, value in clustering.items():
        clusters.setdefault(value, []).append(key)
    
    modularity = 0
    for c in clusters.values():
        cluster = G.subgraph(c)
        l_cluster = cluster.number_of_edges()
        d_cluster = sum([G.degree[n] for n in cluster])
        modularity += (l_cluster/m) - (d_cluster/(2*m))**2

    return modularity

############## Task 6
# Real clustering
print(f"Spectral clustering: {modularity(Giant, clustering)}")
# Random clustering
random_clustering = {node:randint(0, 49) for node in list(G)}
print(f"Random clustering: {modularity(Giant, random_clustering)}")

'''
Output:

Spectral clustering: 0.49875509330544726
Random clustering: 0.00023744283860352596
'''