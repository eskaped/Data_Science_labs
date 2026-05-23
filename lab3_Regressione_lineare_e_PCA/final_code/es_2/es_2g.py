import numpy as np
from sklearn import datasets, linear_model
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib   # per salvare e caricare il modello addestrato
import os   # per controllare se il file del modello addestrato esiste già
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['text.usetex'] = True

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

# g) Applicazione della PCA per ottenere un'immagine utilizzando una frazione del numero di dimensioni (784) ---------------------------------------------
ks = [5, 10, 20, 30, 100]  # il numero di componenti principali da conservare
image_index = 42
single_image = X_test_scaled[image_index].reshape(1, -1)   # estraggo solo l'immagine che mi interessa, mantenendo la dimensione 2D (1, 784)
original_pixels = scaler.inverse_transform(single_image).reshape(28, 28)

# VARIANTE CON GRAFICO 2 X 5, scartata
# fig, ax = plt.subplots(2, 5)
# for i in range(len(ks)):
#     pca_i = PCA(n_components=ks[i])
#     pca_i.fit(X_train_scaled)   # solo fit su tutto il dataset di train
#     reduced_image = pca_i.transform(single_image)   # proietto solo un'immagine
#     reconstructed_image = pca_i.inverse_transform(reduced_image)    # ricostruisco solo un'immagine
#     reconstructed_pixels = scaler.inverse_transform(reconstructed_image).reshape(28, 28)

#     # mostro quanta varianza nel dataset originale è contenuta nelle prime k componenti principali
#     # print(f"Varianza spiegata nelle prime {ks[i]} componenti: ", pca_i.explained_variance_ratio_)
#     # print(f"Somma varianza spiegata nelle prime {ks[i]} componenti: ", np.sum(pca_i.explained_variance_ratio_))

#     ax[0, i].imshow(original_pixels, cmap='gray')
#     ax[1, i].imshow(reconstructed_pixels, cmap='gray')
    
#     ax[0, i].set_title("Immagine originale", fontsize=14)
#     ax[1, i].set_title(f"Immagine ricostruita con \n{ks[i]} componenti principali", fontsize=14)
#     ax[0, i].axis('off')
#     ax[1, i].axis('off')
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
# plt.subplots_adjust(left=0.05, right=0.95, top=0.92, wspace=0.3)
# plt.savefig('images/2_g.png', dpi=300)
# plt.show()

# VARIANTE CON GRAFICO 1 X 6
fig, ax = plt.subplots(1, 6)
ax[0].imshow(original_pixels, cmap='gray')
ax[0].set_title("Immagine originale", fontsize=20, pad=30)
ax[0].axis('off')
for i in range(len(ks)):
    pca_i = PCA(n_components=ks[i])
    pca_i.fit(X_train_scaled)   # solo fit su tutto il dataset di train
    reduced_image = pca_i.transform(single_image)   # proietto solo un'immagine
    reconstructed_image = pca_i.inverse_transform(reduced_image)    # ricostruisco solo un'immagine
    reconstructed_pixels = scaler.inverse_transform(reconstructed_image).reshape(28, 28)

    ax[i+1].imshow(reconstructed_pixels, cmap='gray')
    ax[i+1].set_title(f"Immagine ricostruita\ncon {ks[i]} componenti\nprincipali", fontsize=20, pad=12)
    ax[i+1].axis('off')
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, wspace=0.15)
plt.savefig('images/2_g_(1x6).png', dpi=300)
plt.show()