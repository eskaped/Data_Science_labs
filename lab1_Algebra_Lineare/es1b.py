import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# N = [100, 50, 33, 20]
# n= [1, 2, 3, 5]
N = [1000, 400, 80, 10]
n= [8, 20, 100, 800]
colors = ['b', 'r', 'g', 'orange']

# using sharex=True makes the zoom/pan affects all plots, and limits stay identical for all plots
fig, ax = plt.subplots(2, 2, sharex= True, sharey= True)

# VOLENDO: axes = ax.flatten()   # turns [[ax00, ax01],[ax10, ax11]] → [ax0, ax1, ax2, ax3]

graphs = []
for i in np.arange(2):
    for j in np.arange(2):
        ax[i, j].set_aspect('equal', adjustable= 'box')
        ax[i, j].grid(linestyle = '--')
        ax[i, j].set_xlabel(r'$Re(\frac{\lambda}{\sqrt{n}})$', fontsize=16)
        ax[i, j].set_ylabel(r'$Im(\frac{\lambda}{\sqrt{n}})$', fontsize=16)
        # ax[i, j].set_xticks([-1, -0.5, 0, 0.5, 1])
        # ax[i, j].set_yticks([-1, -0.5, 0, 0.5, 1])
        ax[i, j].set_xticks([-1, 0, 1])
        ax[i, j].set_yticks([-1, 0, 1])
        ax[i, j].tick_params(axis= 'both', labelsize= 16, labelleft=True, labelbottom=True)
        graphs.append(ax[i, j])


# verifying eigenvalues/sqrt(n) distribution -----------------------------------------------------------------

for i in np.arange(len(N)):
    for num_matrices in np.arange(N[i]):
        X = np.random.normal(0, 1, size=(n[i], n[i]))
        eigenvalues, eigenvectors = np.linalg.eig(X)
        graphs[i].plot(np.real(eigenvalues/np.sqrt(n[i])), np.imag(eigenvalues/np.sqrt(n[i])), '.', markersize = 0.5, color = colors[i])

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.subplots_adjust(wspace=0.02, hspace = 0.350)
plt.savefig('fig_1b_2.png', dpi = 300, bbox_inches = 'tight')

plt.show()