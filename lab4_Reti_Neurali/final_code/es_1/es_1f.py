import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchsummary import summary
import torch.optim as optim

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
        x = torch.relu(self.fc1(x)) # Applicazione di ReLU dopo il primo layer
        x = torch.relu(self.fc2(x)) # Applicazione di ReLU dopo il secondo layer
        x = self.fc3(x) # Il layer finale non ha funzione di attivazione
        return x
    
# c) Configurazione del processo di addestramento ------------------------------
# Impostazione del dispositivo su cui eseguire il codice (GPU, se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNN().to(device=device)
loss_function = nn.MSELoss()    # Impostazione di MSE come funzione costo
optimiser = optim.Adam(model.parameters(), lr=0.001)    # Impostazione di Adam come ottimizzatore

# Funzione per stampare il modello e visualizzare il numero totale di parametri
def visualizza_modello(model, input_size):
    print("Struttura della rete neurale:")
    print(model)
    print("\nRiassunto del modello:")
    summary(model, input_size=input_size)

# visualizza_modello(model, input_size=(2, ))

# d) Addestramento del modello -------------------------------------------------
epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()   # Impostazione del modello in modalità addestramento
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)   # Spostamento  dei dati sul dispositivo

        optimiser.zero_grad()                   # Azzeramento dei gradienti
        outputs= model(x_batch)                 # Forward pass
        loss = loss_function(outputs, y_batch)  # Calcolo della perdita
        loss.backward()                         # Backward pass
        optimiser.step()                        # Aggiornamento dei pesi

        running_loss += loss.item() * x_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# e) Valutazione sul test set: predizione di nuovi dati ------------------------
model.eval()    # Impostazione del modello in modalità valutazione
test_loss = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        predictions = model(x_batch)    # Forward pass
        loss = loss_function(predictions, y_batch)  # Calcolo della funzione costo sul batch
        test_loss += loss.item() * x_batch.size(0)

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.6f}")

# Predizione su una coppia di numeri casuali, richiesta dall'esercizio
random_pair = np.random.randint(low=0, high=1000, size=(1, 2))
random_tensor = torch.from_numpy(random_pair).float().to(device)
with torch.no_grad():
    prediction = model(random_tensor)
print(f"Esempio casuale: {random_pair[0][0]} + {random_pair[0][1]} = \t Reale: {random_pair.sum()} \t Predetto: {prediction.item():.2f}")

# f) Lista di test e confronto predizione - somma reale ------------------------
test_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [13, 985], [897, 26], [999, 0], [0, 999], [998, 996], [999, 999]])
test_sums = test_pairs.sum(axis=1)
test_tensor = torch.from_numpy(test_pairs).float().to(device)
with torch.no_grad():
    predictions = model(test_tensor)

for pair, real, pred in zip(test_pairs, test_sums, predictions):
    print(f"{pair[0]} + {pair[1]} = \t Reale: {real} \t Predetto: {pred.item():.2f}")