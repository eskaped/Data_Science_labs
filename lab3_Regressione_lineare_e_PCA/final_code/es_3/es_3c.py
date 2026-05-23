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
# axis=1 vuol dire che la media è svolta attraverso le colonne per ogni riga, cioè per ogni pixel fa la media su tutte le immagini
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

# U.shape = (2282, 2282), S.shape = (2282,) V_T.shape = (2282, 32256), cioè le colonne di V (V_T è V trasposto!) sono gli autovettori della matrice di covarianza, cioè sono le componenti principali

fig, ax = plt.subplots(4, 4)
k = 4
for i in range(k):
    for j in range(k):
        # indice progressivo dell'autofaccia: per ogni riga i fissa, j scorre da 0 a k-1 riempiendo le celle da sinistra a destra; poi i incrementa e si passa alla riga successiva
        index = i * k + j
        # MATLAB (da cui viene il .mat) salva le matrici in column-major order (colonna per colonna), mentre NumPy usa row-major order (riga per riga). 
        # Per questo il reshape(m, n) da solo produrrebbe un'immagine ruotata/trasposta, e il .T corregge l'orientamento
        ax[i, j].imshow(np.reshape(V_T[index, :],(m, n)).T, cmap='gray')
        ax[i, j].set_title(f"PCA {index + 1}", fontsize=16)
        ax[i, j].axis('off')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.25, bottom=0.02, right=0.75, top=0.95, wspace=0.1)
plt.savefig('images/3_c1.png', dpi=300)
plt.show()

# Grafico della varianza cumulativa delle componenti principali in funzione del loro indice: questo giustifica perché bastano poche autofacce per rappresentare bene le facce.
fig, ax = plt.subplots()
num_sing_val = 2282
variance_explained = S**2 / np.sum(S**2)    # frazione di varianza spiegata da ogni componente, somma a 1
cumulative_variance = np.cumsum(variance_explained) # cumulative_variance.shape = (2282,)

ax.plot(np.arange(num_sing_val), cumulative_variance, color='black')

thresholds = [0.50, 0.90, 0.99]
for thresh in thresholds:
    # searchsorted trova il primo indice in cui cumvar supera thresh
    index = np.searchsorted(cumulative_variance, thresh)  # indice 0-based
    # +1 perché l'indice è 0-based ma il numero di componenti è 1-based
    n_thresh = index + 1  # numero di componenti (per l'annotazione)

    ax.axhline(y=thresh, color='firebrick', linestyle='--', alpha=0.6)
    ax.axvline(x=index, color='firebrick', linestyle='--', alpha=0.6)   # posizione sulla curva (0-based)
    # annotazione al punto di intersezione
    ax.annotate(rf'{thresh*100:.0f}\%: {n_thresh} componenti', xy=(index, thresh), xytext=(n_thresh + 10, thresh - 0.04), fontsize=16)

ax.set_xlabel('Indice delle autofacce', fontsize=24, labelpad=12)
ax.set_ylabel('Percentuale di varianza spiegata', fontsize=24, labelpad=12)
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: rf'${x*100:.0f}\%$'))
ax.tick_params(axis='both', labelsize=18)
ax.grid(linestyle='--', color='darkgrey')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.tight_layout()
plt.savefig('images/3_c2.png', dpi=300)
plt.show()

# Visualizzazione di avgFace (faccia media) accanto alla prima autofaccia
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(np.reshape(avgFace, (m, n)).T, cmap='gray')
# ax[1].imshow(np.reshape(V_T[0, :], (m, n)).T, cmap='gray')

# ax[0].set_title("Faccia media del dataset", fontsize=22, pad=20)
# ax[1].set_title(f"Prima autofaccia" , fontsize=22, pad=20)
# # ax[1].set_title(f"Prima autofaccia \n({variance_explained[0]} varianza spiegata)" , fontsize=22, pad=20)
# ax[0].axis('off')
# ax[1].axis('off')

# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
# plt.subplots_adjust(left=0.03, bottom=0.08, right=0.97, top=0.9)
# plt.savefig('3_c_avg.png', dpi=300)
# plt.show()