"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################
    X_train = []
    y_train = []

    n_digits = np.random.randint(1, 11, size=n_train)
    for n_digitsi in n_digits:
        X_traini = np.random.randint(1, 11, size=n_digitsi)
        y_train.append(np.sum(X_traini))
        X_train.append(np.pad(X_traini, (max_train_card - n_digitsi, 0), mode='constant'))

    return np.array(X_train), np.array(y_train)


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    n_test = 200000

    X_test = []
    y_test = []

    for n_digitsi in range(5, 101, 5):
        X_testis = np.random.randint(1, 11, size=(10000, n_digitsi))
        y_test.append(np.sum(X_testis, axis=1))
        X_test.append(X_testis)

    return X_test, y_test
