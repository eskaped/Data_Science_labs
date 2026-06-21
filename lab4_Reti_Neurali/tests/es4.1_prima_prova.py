import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

# DEVO NORMALIZZARE I DATASET DI TRAINING E TEST PERCHè Con valori così grandi,
#  i pesi iniziali casuali producono output enormemente distanti dai target, 
# quindi la MSE iniziale è altissima e il learning rate molto piccolo non basta a farla scendere.

# NOTA: non basta vedere la MSE della parte di test per capire il comportamento
# della rete, bisogna fare dei test dando al modello trainato coppie di numeri e 
# vedere se riesce a sommarli. "Da vedere fanno schifo entrambe" - Di Sante sulle 
# reti (quella con MSE enorme e quella con MSE piccola), penso perchè bisogna vedere
# quanto cala la MSE in modo percentuale rispetto al valore iniziale, non in modo
# numerico assoluto. Con questa interpretazione la rete con dataset normalizzati
# è quasi peggio di quella non normalizzata perchè scende meno percentualmente.



# 1. GENERAZIONE DEL DATASET ------------------------------------------------------
# Genero 10^4 coppie di numeri interi casuali tra 0 e 999
N_couples = 10000
couples_arr = []
sum_arr = []

for n in range(N_couples):
    rand_1 = np.random.randint(999)
    rand_2 = np.random.randint(999)
    couples_arr.append((rand_1, rand_2))
    sum_arr.append(rand_1 + rand_2)

# Impostiamo il dispositivo su cui eseguire il codice (GPU, se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definiamo la classe Dataset di PyTorch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(Y,dtype=torch.float32)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
# Creiamo i DataSet usando l’80% dei dati per l’addestramento e il 20% per la
# validazione
train_dataset = Dataset(np.array(couples_arr[:8000], dtype=np.float32) / 999.0, np.array(sum_arr[:8000], dtype=np.float32) / 1998.0)
test_dataset = Dataset(np.array(couples_arr[8000:], dtype=np.float32) / 999.0, np.array(sum_arr[8000:], dtype=np.float32) / 1998.0)
# train_dataset  = Dataset(couples_arr[:8000], sum_arr[:8000])
# test_dataset  = Dataset(couples_arr[8000:], sum_arr[8000:])

# Creiamo i DataLoader per gestire i batch
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# 2. DEFINIZIONE DELLA RETE NEURALE DENSA --------------------------------------

class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__() # questo comando si può togliere da 
        # python3 in poi se non sbaglio.
        # La rete contiene due layer nascosti densi con 64 neuroni ciascuno,
        # funzione di attivazione ReLU e un layer di output denso a un neurone.
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. FUNZIONE DI PERDITA E OTTIMIZZATORE ---------------------------------------
# Alleno il modello con ottimizzatore Adam e funzione di perdita MSE
model = DenseNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Stampiamo il modello e visualizziamo il numero totale di parametri
def model_visualiser(model, input_size):
    print("Neural Network structure:\n", model)
    print("\nModel summary:\n")
    summary(model, input_size = input_size)

# L'input size è la forma di un singolo input che entra nella rete, quindi in
# questo caso corrisponde a (2,) perchè un singolo elemento è una coppia di numeri
model_visualiser(model, input_size = (2,))

# 4. CICLO DI TRAINING ---------------------------------------------------------
# Faccio il ciclo di training con 100 epoche, come da richiesta

epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train() # Modello impostato in modalità training
    running_loss = 0.
    for inputs, labels in train_loader:
        # Sposto i dati sul dispositivo
        inputs, labels = inputs.to(device), labels.to(device)
        # Azzero i gradienti
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calcolo la perdita
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Aggiorno i pesi
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
# Visualizziamo l'andamento della perdita durante il training
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Andamento della perdita durante il training')
plt.legend()
plt.show()

# 5. VALUTAZIONE SUL DATASET DI TEST -------------------------------------------

model.eval()   # Modello impostato in modalità valutazione 
test_loss = 0.

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)


test_loss /= len(test_loader.dataset)
print(f'Test MSE: {test_loss:.6f}')