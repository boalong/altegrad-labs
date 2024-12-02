"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################
        x_batch = X_test[i][j:min(j+batch_size, n_samples_per_card), :]
        x_batch = torch.tensor(x_batch, dtype=torch.int).to(device)

        y_pred_deepsets.append(deepsets(x_batch))
        y_pred_lstm.append(lstm(x_batch))
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(np.round(y_pred_deepsets), y_test[i]) #your code here
    mae_deepsets = mean_absolute_error(np.round(y_pred_deepsets), y_test[i]) #your code here
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(np.round(y_pred_lstm), y_test[i]) #your code here
    mae_lstm = mean_absolute_error(np.round(y_pred_lstm), y_test[i]) #your code here
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
# your code here #
##################
abscisses = range(5, 101, 5)
plt.plot(abscisses, results['deepsets']['acc'], label='Accuracy, DeepSets')
plt.plot(abscisses, results['lstm']['acc'], label='Accuracy, LSTM')
plt.xlabel('Cardinality of the input set')
plt.legend()
plt.savefig('acc.png')
plt.show()
plt.plot(abscisses, results['deepsets']['mae'], label='MAE, DeepSets')
plt.plot(abscisses, results['lstm']['mae'], label='MAE, LSTM')
plt.xlabel('Cardinality of the input set')
plt.legend()
plt.savefig('mae.png')
plt.show()

'''
LSTM performs well only when the input has cardinality
10 like in the train examples, contrary to Deepsets which 
performs well no matter the cardinality
'''
