import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['text.usetex'] = True


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('miofratelloèfigliounico.JPG')

X = rgb2gray(img/255.)

U, S, Vh = np.linalg.svd(X, full_matrices=True)


# PRIMO GRAFICO, SVD DELL'IMMAGINE COMPLETA --------------------------------------------------------------------
# fig, ax = plt.subplots(1, 2)
# ax[0].grid(linestyle = '--')
# ax[0].set_xlabel('Index', fontsize=18)
# ax[0].set_ylabel('Singular values', fontsize=18)
# ax[0].set_xticks([0, 200, 400, 600, 800, 1000])
# ax[0].tick_params(axis='both', labelsize=16)
# ax[0].plot(S, 'limegreen')
# ax[1].imshow(X, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# ax[1].tick_params(axis='both', labelsize=16)

# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()

# plt.savefig('fig_2b_1.png', dpi = 300, bbox_inches = 'tight')
# print(S)


# SECONDO GRAFICO, CON DIVISIONE 3X3 PER MOSTRARE LA SVD A 3 DIVERSI GRADI DI ACCURATEZZA -----------------------
fig, ax = plt.subplots(3, 3, figsize = (10, 6), constrained_layout = True)

r_arr = [10, 50, 200]

for i in np.arange(len(r_arr)):
    U_tilde = U[:, :r_arr[i]]
    S_tilde = S[:r_arr[i]]
    Vh_tilde = Vh[:r_arr[i], :]
    X_tilde = U_tilde @ np.diag(S_tilde) @ Vh_tilde

    ax[i, 0].set_xlabel('Index', fontsize=13)
    ax[i, 0].set_ylabel('Singular values', fontsize=13)
    ax[i, 0].tick_params(axis='both', labelsize=12)
    ax[i, 0].set_yscale('log')
    ax[i, 0].grid(linestyle = '--')
    ax[i, 0].plot(S_tilde, 'limegreen')

    cumulative_energy_S = np.cumsum(S)/np.sum(S)
    cumulative_energy_S_tilde = np.sum(S_tilde)/np.sum(S)
    ax[i, 1].set_xlabel('Index', fontsize= 13)
    ax[i, 1].set_ylabel('Cumulative Energy', fontsize=13)
    ax[i, 1].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax[i, 1].tick_params(axis='both', labelsize=12)
    ax[i, 1].set_ylim([0, 1.1])
    ax[i, 1].grid(linestyle = '--')
    ax[i, 1].hlines(y = cumulative_energy_S_tilde, xmin = 0, xmax = 1000, color = 'r')
    ax[i, 1].plot(cumulative_energy_S, 'darkorange')

    ax[i, 2].imshow(X_tilde, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    ax[i, 2].set_xlabel('Pixels', fontsize=13)
    ax[i, 2].set_ylabel('Pixels', fontsize=13)
    ax[i, 2].set_xticks([0, 200, 400, 600, 800, 1000])
    ax[i, 2].set_yticks([0, 200, 400, 600, 800, 1000])
    ax[i, 2].tick_params(axis='both', labelsize=10)

plt.savefig('fig_2b_2.png', dpi = 300, bbox_inches = 'tight')

plt.show()