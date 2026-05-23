import numpy as np
from sklearn import datasets, linear_model
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib   # per salvare e caricare il modello addestrato
import os   # per controllare se il file del modello addestrato esiste già

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
# NOTA: HO ELIMINATO TUTTI I PASSAGGI PERCHÈ SI FANNO NEL PUNTO d) PER SALVARE LO SCALER INSIEME AL MODELLO QUANDO LO ADDESTRO

# d) Creazione e addestramento di un modello di regressione logistica, con il metodo del gradiente stocastico per l'ottimizzazione -----------------------
# Se il modello è già stato addestrato e salvato su disco, lo carico direttamente per evitare di riaddestrarlo ad ogni esecuzione
# La regola generale è: tutto ciò che trasforma i dati prima di entrare nel modello deve essere salvato insieme al modello. In questo caso bisogna salvare il modello e lo scaler
if os.path.exists('mnist_model.pkl'):
    model = joblib.load('mnist_model.pkl')  # carica il modello salvato
    scaler = joblib.load('mnist_scaler.pkl')    # carica lo scaler salvato
    X_train_scaled = scaler.transform(X_train)  # uso solo transform e non fit_transform perchè lo scaler è già stato fittato sul training set durante il primo addestramento
    X_test_scaled = scaler.transform(X_test)
    print('Modello e scaler caricati da disco.')
else:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'mnist_scaler.pkl')   # salva lo scaler prima di addestrare

    # tol è necessario per fermare l'addestramento: ad ogni iterazione, SAG controlla se il miglioramento della loss rispetto all'iterazione precedente è diventato più piccolo di tol
    model = linear_model.LogisticRegression(solver='sag', max_iter=500, verbose=1, tol=1e-3)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, 'mnist_model.pkl')   # salva il modello addestrato su disco
    print('Modello addestrato e salvato su disco.')

# IMPORTANTE: convergence after 462 epochs took 379 seconds con tol=1e-3, mentre prima con tol=1e-2 ho avuto convergence after 43 epochs took 36 seconds

# e) Calcolo la sparsità (=percentuale di pesi non nulli) e l'accuratezza (=frazione di esempi classificati correttamente sul totale) --------------------
# Un modello si dice "sparso" quando molti dei suoi coefficienti sono esattamente zero, cioè il modello ha imparato a ignorare completamente certe feature. 
# La sparsità si calcola come la percentuale di pesi non nulli rispetto al totale. 
sparsity = np.mean(model.coef_ == 0) * 100
print("Sparsità (% pesi nulli): ", sparsity)

accuracy = model.score(X_test_scaled, y_test)
print("Accuratezza: ", accuracy)