import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(' ')
            labels.append(content[0].split(':')[0])
            docs.append(' '.join(content[1:]))
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab

path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx, doc in enumerate(docs):
        G = nx.Graph()
        for i, doci in enumerate(doc):
            if doci not in G.nodes():
                G.add_node(doci)
                G.nodes[doci]['label'] = vocab[doci]
        for i, doci in enumerate(doc):
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    G.add_edge(doci, doc[j])
                else:
                    break
        
        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, LovaszTheta
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations
G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label')) # your code here #
G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label')) # your code here #

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=False) # your code here #

# Construct kernel matrices
K_train = gk.fit_transform(G_train) # your code here #
K_test = gk.transform(G_test) # your code here #

#Task 13

# Train an SVM classifier and make predictions
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Evaluate the predictions
print("Accuracy, Weisfeiler-Lehman subtree kernel:", accuracy_score(y_pred, y_test))


#Task 14
from grakel.kernels import (
    RandomWalk, 
    RandomWalkLabeled, 
    PyramidMatch, 
    NeighborhoodHash, 
    ShortestPath, 
    ShortestPathAttr, 
    GraphletSampling, 
    SubgraphMatching,
    HadamardCode,
    NeighborhoodSubgraphPairwiseDistance,
    LovaszTheta,
    SvmTheta,
    Propagation,
    PropagationAttr,
    OddSth,
    MultiscaleLaplacian,
    VertexHistogram,
    EdgeHistogram,
    GraphHopper,
    CoreFramework,
    WeisfeilerLehmanOptimalAssignment
)

from time import time
def evaluate_kernel(kernel_class, G_train, G_test, y_train, y_test, **kernel_params):
    try:
        start = time()
        # Initialize kernel with parameters
        gk = kernel_class(**kernel_params)
        
        # Print graph format for debugging
        print(f"\nTesting {kernel_class.__name__}")
        
        # Compute kernel matrices
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)
        
        # Train and evaluate
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        
        acc = accuracy_score(y_pred, y_test)
        print(f"Accuracy, {kernel_class.__name__}: {acc:.3f}")
        print(f"Time: {time()-start}")
        return acc
        
    except Exception as e:
        print(f"Error with {kernel_class.__name__}: {str(e)}")
        return None

kernels_to_test = [(kernel, {}) for kernel in [
    # PyramidMatch,
    # NeighborhoodHash, 
    # ShortestPath, 
    # GraphletSampling, 
    SvmTheta,
    # Propagation,
    # OddSth,
    VertexHistogram
    # CoreFramework,
    # WeisfeilerLehmanOptimalAssignment
]]

kernels_to_test.append((HadamardCode, {"n_iter":1}))

for kernel_class, params in kernels_to_test:
    evaluate_kernel(kernel_class, G_train, G_test, y_train, y_test, **params)


'''
Output:

Vocabulary size:  7186
Example of graph-of-words representation of document
Accuracy, Weisfeiler-Lehman subtree kernel: 0.858

Testing SvmTheta
Accuracy, SvmTheta: 0.308
Time: 41.327484130859375

Testing VertexHistogram
Accuracy, VertexHistogram: 0.864
Time: 1.2865419387817383

Testing HadamardCode
Accuracy, HadamardCode: 0.864
Time: 55.0876624584198
'''