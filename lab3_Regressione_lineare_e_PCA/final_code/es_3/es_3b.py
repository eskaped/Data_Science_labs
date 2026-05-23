import numpy as np
import os
import scipy.io

# a) Recupero il dataset di volti da Virtuale -----------------------------------------------------------------------------------------------------------------
mat_contents = scipy.io.loadmat(os.path.join('allFaces.mat'))

# Le facce sono immagini di 168 x 192 pixels, quindi ogni foto è un punto in uno spazio 32256-dimensionale
faces = mat_contents['faces']   # faces.shape = (32256, 2410), cioè abbiamo 2410 foto di facce (colonne), ognuna composta da 32256 pixels (messi in colonna)

m = int(mat_contents['m'].item())
n = int(mat_contents['n'].item())

# nfaces = numero di foto per faccia diversa, cioè numero di foto di facce per persona diversa
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

# b) Carico il dataset e appiattisco le immagini da matrici a vettori ----------------------------------------------------------------
# Attenzione: il dataset è organizzato come una matrice dove ogni colonna è un'immagine già appiattita: 168×192 = 32256 pixel messi in un vettore.
# nfaces è un array di 38 elementi (una per persona), che dice quante foto ci sono per ciascuna persona. La somma fa 2410.
print(f"Ogni immagine è già un vettore di {faces.shape[0]} pixel (={m}x{n})")
print(f"Totale immagini: {faces.shape[1]}, distribuite su {len(nfaces)} persone")