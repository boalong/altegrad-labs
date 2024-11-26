"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    
    ##################
    # your code here #
    ##################
    ns = np.random.randint(10, 21, 100)
    for n in ns[:50]:
        Gs.append(nx.fast_gnp_random_graph(n, 0.2))
        y.append(0)
    for n in ns[50:]:
        Gs.append(nx.fast_gnp_random_graph(n, 0.4))
        y.append(1)

    permuted_indices = np.random.permutation(100)

    Gs = [Gs[i] for i in permuted_indices]
    y = [y[i] for i in permuted_indices]

    return list(Gs), list(y)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)