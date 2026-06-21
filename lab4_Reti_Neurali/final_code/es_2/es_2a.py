import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# a) Realizzazione di una rete neurale che determini il segno (+, -, 0) della somma di due numeri, sulla base del codice implementato per l'esercizio 1

# Generazione dei dati
N_pairs = 10000
pairs = np.random.randint(low=-999, high=1000, size=(N_pairs, 2))
# Il + 1 è necessario perchè Cross Entropy si aspetta che le classi siano indici interi non negativi, ma np.sign restituisce -1, 0, 1 e -1 non è un indice valido
# Quindi si rimappano le labels aggiungendo +1 a tutto: -1 -> 0 (negativo), 0 -> 1 (zero), +1 -> 2 (positivo)
pairs_sums_signs = np.sign(pairs.sum(axis=1)) + 1

# Definizione della rete neurale con PyTorch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.long)  # Cross Entropy richiede target (labels) di tipo long, cioè intero a 64 bit
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
pairs_train, pairs_test, signs_train, signs_test = train_test_split(pairs, pairs_sums_signs, test_size=0.20, random_state=4028987551)

# Creazione dei DataSet
train_dataset = Dataset(pairs_train, signs_train)
test_dataset  = Dataset(pairs_test, signs_test)

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
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Applicazione di ReLU dopo il primo layer
        x = torch.relu(self.fc2(x)) # Applicazione di ReLU dopo il secondo layer
        x = self.fc3(x) # Il layer finale non ha funzione di attivazione
        return x
    
# Configurazione del processo di addestramento
# Impostazione del dispositivo su cui eseguire il codice (GPU, se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNN().to(device=device)
loss_function = nn.CrossEntropyLoss()    # Impostazione della Cross Entropy come funzione costo, adatta a problemi di classificazione
optimiser = optim.Adam(model.parameters(), lr=0.001)    # Impostazione di Adam come ottimizzatore

# Funzione per stampare il modello e visualizzare il numero totale di parametri
def visualizza_modello(model, input_size):
    print("Struttura della rete neurale:")
    print(model)
    print("\nRiassunto del modello:")
    summary(model, input_size=input_size)

visualizza_modello(model, input_size=(2, ))

# Addestramento del modello
epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()   # Impostazione del modello in modalità addestramento
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)   # Spostamento dei dati sul dispositivo

        optimiser.zero_grad()                   # Azzeramento dei gradienti
        outputs= model(inputs)                  # Forward pass
        loss = loss_function(outputs, labels)   # Calcolo della perdita
        loss.backward()                         # Backward pass
        optimiser.step()                        # Aggiornamento dei pesi

        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Grafico dell'andamento della perdita durante il training
fig, ax = plt.subplots()
ax.plot(train_losses, color='red', label='Training Loss')

ax.set_xlabel("Epoca", fontsize=22, labelpad=12)
ax.set_ylabel("Funzione di perdita", fontsize=22, labelpad=12)
ax.tick_params(axis='both', labelsize=18)
ax.set_xticks(np.linspace(0, 100, 11))
ax.legend(fontsize=20)
ax.set_yscale('log')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.savefig('images/training_2log.png', dpi=300)
plt.show()