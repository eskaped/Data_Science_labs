import numpy as np
from sklearn import datasets, linear_model
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# a) Scarico il dataset MNIST di sklearn -----------------------------------------------------------------------------------------------------------------
# in questo modo python non verifica il certificato SSL. Non è una pratica sicura in generale ma per scaricare un dataset pubblico da una fonte nota è ok
ssl._create_default_https_context = ssl._create_unverified_context

# fetch_openml restituisce un pandas DataFrame:
# X: 70000 righe (numero di immagini) e 784 colonne (numero di features, 28 x 28, una per ogni pixel)
# Y: vettore di 70000 elementi (numero di immagini), ognuno dei quali è il numero scritto, da 0 a 9
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

# Converto il dataset a matrice e vettore di NumPy perchè è più comodo da usare per il laboratorio
X = X.to_numpy()
y = y.to_numpy()

# Converto y da un vettore di stringhe (così arriva dal dataset di sklearn) a un vettore di int64, il tipo standard di int usato da NumPy
y = y.astype(np.int64)

# b) Divido il dataset in parte di train e parte di test secondo le istruzioni ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000, test_size=10000, random_state=3843637271)

# c) Normalizziamo le caratteristiche a media zero e varianza uno. ---------------------------------------------------------------------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
# fit_transform sul train è la composizione di due passaggi in uno, andrebbe fatto il fit solo sui dati di train e poi il transform su tutto
# L'obiettivo è evitare data leakage (cross contamination), perchè il test deve simulare dati mai visti dal modello
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Shape X_train:', X_train_scaled.shape, '; Shape X_test:', X_test_scaled.shape)