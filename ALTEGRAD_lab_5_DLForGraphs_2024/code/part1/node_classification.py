"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network
nx.draw_networkx(G, node_color=y)


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy with deepwalk embeddings: {accuracy_score(y_test, y_pred)}')


############## Task 8
# Generates spectral embeddings
def spectral_embeddings(G, d=10):
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

    return U

embeddings_spectral = spectral_embeddings(G, 2)
X_train_spectral = embeddings_spectral[idx_train,:]
X_test_spectral = embeddings_spectral[idx_test,:]

clf_spectral = LogisticRegression().fit(X_train_spectral, y_train)
y_pred_spectral = clf_spectral.predict(X_test_spectral)
print(f'Accuracy with spectral embeddings: {accuracy_score(y_test, y_pred_spectral)}')


'''
Output:

Number of nodes: 34
Number of edges: 78
Generating walks
100%|██████████| 34/34 [00:00<00:00, 411.52it/s]
Training word2vec

Accuracy with deepwalk embeddings: 1.0
Accuracy with spectral embeddings: 0.8571428571428571
'''