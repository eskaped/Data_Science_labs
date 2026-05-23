import numpy as np
import os
import scipy.io

# a) Recupero il dataset di volti da Virtuale -----------------------------------------------------------------------------------------------------------------
mat_contents = scipy.io.loadmat(os.path.join('allFaces.mat'))

# Le facce sono immagini di 168 x 192 pixels, quindi ogni foto è un punto in uno spazio 32256-dimensionale
faces = mat_contents['faces']   # faces.shape = (32256, 2410), cioè abbiamo 2410 foto di facce (colonne), ognuna composta da 32256 pixels (messi in colonna)
print(faces, "faces.shape = ", faces.shape)

m = int(mat_contents['m'].item())
n = int(mat_contents['n'].item())
print("m = ", m, "\nn = ", n) # (m, n) = (168, 192)

# nfaces = numero di foto per faccia diversa, cioè numero di foto di facce per persona diversa
nfaces = np.ndarray.flatten(mat_contents['nfaces'])
print("Number of faces images = ", nfaces)    # sommano a 2410 (com'è giusto che sia), 9 persone su 38 hanno meno di 64 foto (in numero variabile)