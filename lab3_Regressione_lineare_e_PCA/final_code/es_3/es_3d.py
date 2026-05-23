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

# d) Carico una nuova foto di un volto in primo piano, la rendo compatibile col dataset e la proietto nello spazio delle autofacce per calcolare i coefficienti 
# di espansione lineare. Poi la ricostruisco a partire da k autofacce, variando k per analizzare l'accuratezza della ricostruzione ----------------------------

# Tutte le 64 foto della persona 38, testFaces.shape = (32256, 64)
testFaces = faces[:np.sum(nfaces[:37]), :np.sum(nfaces[:38])] # np.sum(nfaces[:number]) è l'indice della colonna in faces che corrisponde all'inizio del blocco di foto di quella persona
# Solo la prima foto della persona 38
testFace = faces[:, np.sum(nfaces[:37])]

# Proiezione nello spazio delle autofacce:
# Come prima cosa centro l'immagine di test sottraendo la media calcolata sul dataset di training.
testFace_centered = testFace - avgFace
# Il secondo passaggio è calcolare i coefficienti di espansione
# r = 10
# coefs = V_T[:r, :] @ testFace_centered
# Il terzo passaggio è ricostruire l'immagine risommando le autofacce pesate dai coefficienti, e riaggiungendo la media:
# reconFace = avgFace + V_T[:r, :].T @ coefs

# Valuto al variare di r
rs = [5, 50, 100, 500, V_T.shape[0]]    # V_T.shape[0] = 2282, cioè tutte le righe disponibili, la ricostruzione completa
fig, ax = plt.subplots(1, 6)
ax[5].imshow(np.reshape(testFace, (m, n)).T, cmap='gray')
ax[5].set_title("Immagine originale", fontsize=18, pad=30)
ax[5].axis('off')
for i in range(len(rs)):
    coefs_r = V_T[:rs[i], :] @ testFace_centered
    reconFace_r = avgFace + V_T[:rs[i], :].T @ coefs_r

    ax[i].imshow(np.reshape(reconFace_r, (m, n)).T, cmap='gray')

    ax[i].set_title(f"Volto ricostruito \ncon {rs[i]} componenti\nprincipali", fontsize=18)
    ax[i].axis('off')

ax[4].set_title(r"\shortstack{Ricostruzione completa \\ ($\neq$ originale)}", fontsize=18, pad=15)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.15)
plt.savefig('images/3_d.png', dpi=300)
plt.show()