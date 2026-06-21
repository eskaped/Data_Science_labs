import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# a) Generazione dei dati ------------------------------------------------------
N_pairs = 10000
pairs = np.random.randint(low=0, high=1000, size=(N_pairs, 2))
pairs_sums = pairs.sum(axis=1)

# b) Definizione della rete neurale, con PyTorch -------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].unsqueeze(0)    # rende la shape (1, ) invece di dover riscalare ogni volta con .view(-1, 1)
    
# Implementazione di una parte della richiesta del punto d), poichè andrebbe fatta a questo livello del codice
pairs_train, pairs_test, sums_train, sums_test = train_test_split(pairs, pairs_sums, test_size=0.20, random_state=4028987551)

# Creazione dei DataSet
train_dataset = Dataset(pairs_train, sums_train)
test_dataset  = Dataset(pairs_test, sums_test)

# Creazione dei DataLoader per gestire i batch
# Durante l'addestramento si desidera che il modello veda i dati in un ordine diverso a ogni epoca, anche se in realtà questi dati sono già casuali
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Nella fase di test non si aggiornano i pesi, quindi l'ordine dei campioni non cambia il risultato finale
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__()
        # La rete contiene 2 layers nascosti, densi, con 64 neuroni e ReLU come funzione di attivazione
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Applico ReLU dopo il primo layer
        x = torch.relu(self.fc2(x)) # Applico ReLU dopo il secondo layer
        x = self.fc3(x) # Il layer finale non ha funzione di attivazione
        return x