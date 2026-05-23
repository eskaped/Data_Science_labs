import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# a) Recupero il dataset di volti da Virtuale -----------------------------------------------------------------------------------------------------------------
mat_contents = scipy.io.loadmat(os.path.join('allFaces.mat'))

# Le facce sono immagini di 168 x 192 pixels, quindi ogni foto è un punto in uno spazio 32256-dimensionale
faces = mat_contents['faces']   # faces.shape = (32256, 2410), cioè abbiamo 2410 foto di facce (colonne), ognuna composta da 32256 pixels (messi in colonna)

m = int(mat_contents['m'].item())
n = int(mat_contents['n'].item())

# nfaces = numero di foto per faccia diversa, cioè numero di foto di facce per persona diversa
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

# c) Utilizzo la PCA per decomporre il dataset in autofacce, e visualizzo le prime k autofacce per interpretarne il risultato ---------------------------------
# Uso le prime 36 persone come dati di train, le rimanenti 2 come dato di test. Il numero di immagini di test scende a 2282
trainingFaces = faces[:, :np.sum(nfaces[:36])]   # np.sum(nfaces[:number]) è l'indice della colonna in faces che corrisponde all'inizio del blocco di foto di quella persona

# Prima di applicare la PCA bisogna SEMPRE riscalare il dataset togliendo la media
avgFace = np.mean(trainingFaces, axis=1)    # avgFace.shape = (32256,)
X = trainingFaces - np.tile(avgFace,(trainingFaces.shape[1], 1)).T

# Applico la PCA eseguendola manualmente
if not os.path.exists('svd_results.npz'):
    # Calcolo lento, eseguito solo la prima volta per il salvataggio (circa 40 secondi)
    U, S, V_T = np.linalg.svd(X.T, full_matrices=False)
    np.savez('svd_results.npz', U=U, S=S, VT=V_T)
    print("SVD calcolata e salvata.")
else:
    # Caricamento veloce, tutte le volte successive
    data = np.load('svd_results.npz')
    U, S, V_T = data['U'], data['S'], data['VT']
    print("SVD caricata da file.")

# Solo la prima foto della persona 38
testFace = faces[:, np.sum(nfaces[:37])]

testFace_centered = testFace - avgFace

# e) Valuto la perdita di informazione in base al numero di autofacce utilizzate, mostrando la differenza tra immagine originale e ricostruzione al variare di k.
# Per valutare la perdita di informazione al variare del numero di autofacce utilizzate per la ricostruzione creo un grafico:
k_range = np.arange(0, 2282, 1)
errors = []
for r in k_range:
    coefs_r = V_T[:r, :] @ testFace_centered
    reconFace_r = avgFace + V_T[:r, :].T @ coefs_r
    mse = np.mean((testFace - reconFace_r)**2)
    errors.append(mse)
fig, ax = plt.subplots()
ax.plot(k_range, errors, color='black')
for k_val, label in zip([23, 288], [r'\shortstack{90\% varianza \\ di training\\ (k=23)}', r'\shortstack{99\% varianza\\ di training\\ (k=288)}']):
    ax.axvline(x=k_val, color='firebrick', linestyle='--', alpha=0.6)
    ax.annotate(label, xy=(k_val, max(errors)*0.7), 
                xytext=(k_val + 20, max(errors)*0.75), fontsize=16)
ax.set_xlabel("Numero di autofacce k", fontsize=24, labelpad=12)
ax.set_ylabel("MSE", fontsize=24, labelpad=12)
ax.set_xticks(np.arange(0, 2500, 250))
ax.tick_params(axis='both', labelsize=18)
ax.grid(linestyle='--', color='darkgrey')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.09, bottom=0.1, right=0.98, top=0.97)
plt.savefig('images/3_e_mse.png', dpi=300)
plt.show()