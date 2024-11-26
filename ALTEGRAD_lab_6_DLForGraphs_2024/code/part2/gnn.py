"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim

from models import GNN
from utils import create_dataset, sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 200
batch_size = 8
n_hidden_1 = 16
n_hidden_2 = 32
n_hidden_3 = 32
learning_rate = 0.01

# Generates synthetic dataset
Gs, y = create_dataset()
n_class = np.unique(y).size

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

N_train = len(G_train)
N_test = len(G_test)

# Initializes model and optimizer
model = GNN(1, n_hidden_1, n_hidden_2, n_hidden_3, n_class, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Trains the model
for epoch in range(epochs):
    t = time.time()
    model.train()
    
    train_loss = 0
    correct = 0
    count = 0
    for i in range(0, N_train, batch_size):
        adj_batch = list()
        idx_batch = list()
        y_batch = list()

        ############## Task 7
        
        ##################
        # your code here #
        ##################
        len_batch = min(batch_size, N_train-i)
        for j in range(i, i+len_batch):
            nj = G_train[j].number_of_nodes()
            A_tilde = nx.adjacency_matrix(G_train[j]) + sp.identity(nj)
            adj_batch.append(A_tilde)
            idx_batch.extend([j-i]*nj)
            y_batch.append(y_train[j])
        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.ones((adj_batch.shape[0], 1))
        
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)
        y_batch = torch.LongTensor(y_batch).to(device)
        features_batch = torch.FloatTensor(features_batch).to(device)

        optimizer.zero_grad()
        output = model(features_batch, adj_batch, idx_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(train_loss / count),
              'acc_train: {:.4f}'.format(correct / count),
              'time: {:.4f}s'.format(time.time() - t))
        
print('Optimization finished!')

# Evaluates the model
model.eval()
test_loss = 0
correct = 0
count = 0
for i in range(0, N_test, batch_size):
    adj_batch = list()
    idx_batch = list()
    y_batch = list()

    ############## Task 7
    
    ##################
    # your code here #
    ##################
    len_batch = min(batch_size, N_test-i)
    for j in range(i, i+len_batch):
        nj = G_test[j].number_of_nodes()
        A_tilde = nx.adjacency_matrix(G_test[j]) + sp.identity(nj)
        adj_batch.append(A_tilde)
        idx_batch.extend([j-i]*nj)
        y_batch.append(y_test[j])
    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.ones((adj_batch.shape[0], 1))

    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)
    idx_batch = torch.LongTensor(idx_batch).to(device)
    y_batch = torch.LongTensor(y_batch).to(device)
    features_batch = torch.FloatTensor(features_batch).to(device)

    output = model(features_batch, adj_batch, idx_batch)
    loss = loss_function(output, y_batch)
    test_loss += loss.item() * output.size(0)
    count += output.size(0)
    preds = output.max(1)[1].type_as(y_batch)
    correct += torch.sum(preds.eq(y_batch).double())

print('loss_test: {:.4f}'.format(test_loss / count),
      'acc_test: {:.4f}'.format(correct / count),
      'time: {:.4f}s'.format(time.time() - t))


'''
Epoch: 0191 loss_train: 0.2963 acc_train: 0.8556 time: 0.1514s
Optimization finished!
loss_test: 0.1281 acc_test: 1.0000 time: 0.1426s
'''