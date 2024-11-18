"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# import matplotlib.pyplot as plt
# def visualize_graph(G, filename):
#     plt.figure(figsize=(12, 5))
#     nx.draw(G, with_labels=True, node_color='lightblue', 
#             node_size=500, font_size=16, font_weight='bold')    
#     plt.savefig(filename)

############## Task 7
#load Mutag dataset
def load_dataset():
    dataset = TUDataset(root='tmp/MUTAG', name='MUTAG')
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]
    y = [data.y.item() for data in dataset]
    return Gs, y

Gs, y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]

    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test

# P4 = nx.Graph()
# P4.add_nodes_from(range(4))
# P4.add_edge(0,1)
# P4.add_edge(1,2)
# P4.add_edge(2,3)
# visualize_graph(P4, 'glet.png')

# C4 = nx.Graph()
# C4.add_nodes_from(range(4))
# C4.add_edge(0,1)
# C4.add_edge(1,2)
# C4.add_edge(2,3)
# C4.add_edge(3,0)
# visualize_graph(C4, 'fig.png')

# print(shortest_path_kernel([P4, C4], [P4, C4]))


############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(Gs_train), 4))

    for i, G in enumerate(Gs_train):
        G_nodes = list(G)
        # sample graphlets from G
        for _ in range(n_samples):
            sampled_nodes = np.random.choice(G_nodes, 3, replace=False)
            graphlet = G.subgraph(sampled_nodes)
            for j, glet in enumerate(graphlets):
                phi_train[i, j] += int(nx.is_isomorphic(graphlet, glet))
    #print(phi_train)

    phi_test = np.zeros((len(Gs_test), 4))
    for i, G in enumerate(Gs_test):
        G_nodes = list(G)
        # sample graphlets from G
        for _ in range(n_samples):
            sampled_nodes = np.random.choice(G_nodes, 3, replace=False)
            graphlet = G.subgraph(sampled_nodes)
            for j, glet in enumerate(graphlets):
                phi_test[i, j] += int(nx.is_isomorphic(graphlet, glet))
    #print(phi_test)

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 9
K_train_graphlet, K_test_graphlet = graphlet_kernel(G_train, G_test)

# G1 = nx.Graph()
# G1.add_nodes_from(range(4))
# G1.add_edge(0,1)
# G1.add_edge(1,2)
# G1.add_edge(2,3)
# G1.add_edge(3,0)

# G2 = nx.Graph()
# G2.add_nodes_from(range(4))

# print(graphlet_kernel([G1], [G2])[1])

############## Task 10
# SP
clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)
y_pred_sp = clf_sp.predict(K_test_sp)
print(f'Accuracy with shortest path kernel: {accuracy_score(y_test, y_pred_sp)}')

# Graphlets
clf_graphlet = SVC(kernel='precomputed')
clf_graphlet.fit(K_train_graphlet, y_train)
y_pred_graphlet = clf_graphlet.predict(K_test_graphlet)
print(f'Accuracy with graphlet kernel: {accuracy_score(y_test, y_pred_graphlet)}')