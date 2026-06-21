import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split

# a) Realizzazione di una rete neurale che determini il segno (+, +, 0) della somma di due numeri, sulla base del codice implementato per l'esercizio 1

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

# visualizza_modello(model, input_size=(2, ))

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

# b) Valutazione sul test set e calcolo dell'accuratezza -----------------------
model.eval()    # Impostazione del modello in modalità valutazione
correct = 0
total = 0
misclassified = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        predictions = model(inputs)    # Forward pass
        _, predicted = torch.max(predictions, 1)    # Previsione della classe con punteggio più alto

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Salvataggio dei numeri mal classificati
        for i in range(inputs.size(0)):
            if predicted[i] != labels[i]:
                misclassified.append((inputs[i].cpu(), predicted[i].item(), labels[i].item()))

accuracy = 100 * correct / total
print(f"Accuracy sul test set: {accuracy:.2f}%")

# Predizione su una coppia di numeri casuali, richiesta dall'esercizio
random_pair = np.random.randint(low=-999, high=1000, size=(1, 2))
random_sign = int(np.sign(random_pair.sum()))
random_tensor = torch.from_numpy(random_pair).float().to(device)
with torch.no_grad():
    prediction = model(random_tensor)
    _, predicted = torch.max(prediction, 1)
    predicted_class = predicted.item()
print(f"Esempio casuale: {random_pair[0][0]} + {random_pair[0][1]} = {random_pair.sum()}")
print(f"Segno reale: {random_sign} \t Segno predetto: {predicted_class - 1}")

# Lista di test e confronto predizione - segno reale ------------------------
test_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1], [13, 985], [897, 26], [-15, -698], [-950, -32], [-999, 999], [-1, 1], [999, 999]])
test_signs = np.sign(test_pairs.sum(axis=1)) + 1
test_tensor = torch.from_numpy(test_pairs).float().to(device)
with torch.no_grad():
    predictions = model(test_tensor)
    _, predicted = torch.max(predictions, 1)

for pair, real, pred in zip(test_pairs, test_signs, predicted):
    print(f"{pair[0]} + {pair[1]} = {pair.sum()}\t Segno reale: {real - 1} \t Segno predetto: {(pred.item() - 1)}")