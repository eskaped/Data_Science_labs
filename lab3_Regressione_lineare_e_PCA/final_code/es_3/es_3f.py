import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
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


# Solo la prima foto della persona 37
testFace = faces[:, np.sum(nfaces[:36])]

testFace_centered = testFace - avgFace

# f) Espando l’immagine di un altro (s)oggetto, un cane, nella base delle autofacce. Quante autofacce sono necessarie per una rappresentazione accurata?
im_dog = Image.open("theo.jpeg")
im_plant = Image.open("houseplant.jpg")
# print(im.format, im.size, im.mode)  # JPEG (1035, 976) RGB
im_dog_grey = im_dog.convert('L')
im_plant_grey = im_plant.convert('L')
# Attenzione: Pillow usa la convenzione (larghezza, altezza) per le dimensioni, l'inverso di NumPy che usa (righe, colonne) cioè (altezza, larghezza). 
# Le immagini del dataset sono m=168 pixel di larghezza e n=192 pixel di altezza, quindi pass a .resize() la tupla (168, 192), cioè (m, n). 
im_dog_grey_resized = im_dog_grey.resize((m, n))    # osservo una lieve distorsione dovuta al cambio di aspect ratio
im_plant_grey_resized = im_plant_grey.resize((m, n))    # osservo una lieve distorsione dovuta al cambio di aspect ratio
im_np_dog = np.array(im_dog_grey_resized)
im_np_plant = np.array(im_plant_grey_resized)
im_vector_dog = im_np_dog.T.flatten()   # shape: (32256,), compatibile con le colonne di faces
im_vector_plant = im_np_plant.T.flatten()   # shape: (32256,), compatibile con le colonne di faces

# Procedo a proiettare nello spazio delle autofacce, esattamente come prima:
im_centered_dog = im_vector_dog - avgFace
im_centered_plant = im_vector_plant - avgFace
# Valuto al variare di r
rs = [50, 200, 500, 1000, V_T.shape[0]]    # V_T.shape[0] = 2282, cioè tutte le righe disponibili, la ricostruzione completa

fig, ax = plt.subplots(2, 6)
ax[0, 5].imshow(np.reshape(im_vector_dog, (m, n)).T, cmap='gray')
ax[0, 5].set_title("Immagine originale", fontsize=18, pad=15)
ax[0, 5].axis('off')
ax[1, 5].imshow(np.reshape(im_vector_plant, (m, n)).T, cmap='gray')
ax[1, 5].set_title("Immagine originale", fontsize=18, pad=15)
ax[1, 5].axis('off')
for i in range(len(rs)):
    coefs_r_dog = V_T[:rs[i], :] @ im_centered_dog
    coefs_r_plant = V_T[:rs[i], :] @ im_centered_plant
    reconFace_r_dog = avgFace + V_T[:rs[i], :].T @ coefs_r_dog
    reconFace_r_plant = avgFace + V_T[:rs[i], :].T @ coefs_r_plant
    ax[0, i].imshow(np.reshape(reconFace_r_dog, (m, n)).T, cmap='gray')
    ax[0, i].set_title(f"Immagine ricostruita\ncon $k = {rs[i]}$", fontsize=18)
    ax[0, i].axis('off')
    ax[1, i].imshow(np.reshape(reconFace_r_plant, (m, n)).T, cmap='gray')
    ax[1, i].set_title(f"Immagine ricostruita\ncon $k = {rs[i]}$", fontsize=18)
    ax[1, i].axis('off')

ax[0, 4].set_title(r"\shortstack{Ricostruzione completa \\ ($\neq$ originale)}", fontsize=18)
ax[1, 4].set_title(r"\shortstack{Ricostruzione completa \\ ($\neq$ originale)}", fontsize=18)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.15)
plt.savefig('images/3_f_both.png', dpi=300)
plt.show()

# Per finire, visualizziamo nello stesso grafico la MSE della 38esima faccia e delle immagini del cane e della pianta al fine di confrontare meglio i valori numerici
k_range = np.arange(0, 2282, 1)
errors_face = []
errors_im_dog = []
errors_im_plant = []
for r in k_range:
    coefs_r_face = V_T[:r, :] @ testFace_centered
    coefs_r_im_dog = V_T[:r, :] @ im_centered_dog
    coefs_r_im_plant = V_T[:r, :] @ im_centered_plant
    reconFace_r_face = avgFace + V_T[:r, :].T @ coefs_r_face
    reconFace_r_im_dog = avgFace + V_T[:r, :].T @ coefs_r_im_dog
    reconFace_r_im_plant = avgFace + V_T[:r, :].T @ coefs_r_im_plant
    mse_face = np.mean((testFace - reconFace_r_face)**2)
    mse_im_dog = np.mean((im_vector_dog - reconFace_r_im_dog)**2)
    mse_im_plant = np.mean((im_vector_plant - reconFace_r_im_plant)**2)
    errors_face.append(mse_face)
    errors_im_dog.append(mse_im_dog)
    errors_im_plant.append(mse_im_plant)
fig, ax = plt.subplots()
ax.plot(k_range, errors_face, color='black', label='Volto dal test set')
ax.plot(k_range, errors_im_dog, color='royalblue', label='Cane (fuori dominio)')
ax.plot(k_range, errors_im_plant, color='forestgreen', label='Pianta (fuori dominio)')
for k_val, label in zip([23, 288], [r'\shortstack{90\% varianza \\ di training\\ (k=23)}', r'\shortstack{99\% varianza\\ di training\\ (k=288)}']):
    ax.axvline(x=k_val, color='firebrick', linestyle='--', alpha=0.6)
    ax.annotate(label, xy=(k_val, max(errors_face)*0.7), 
                xytext=(k_val + 20, max(errors_face)*3), fontsize=16)
ax.legend(fontsize=18)
ax.set_xlabel("Numero di autofacce k", fontsize=24, labelpad=12)
ax.set_ylabel("MSE", fontsize=24, labelpad=12)
ax.set_xticks(np.arange(0, 2500, 250))
ax.set_yticks(np.arange(0, 22500, 2500))
ax.tick_params(axis='both', labelsize=18)
ax.grid(linestyle='--', color='darkgrey')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.tight_layout()
plt.savefig('images/3_f_mse_confronto.png', dpi=300)
plt.show()