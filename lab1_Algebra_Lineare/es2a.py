import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(2, 4)

eig = []
sing = []

N = 100
ε_arr = np.linspace(0, 0.00001, N)

for ε in ε_arr:
    A = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [ε, 0, 0, 0]])
    singularvals = np.linalg.svd(A, compute_uv = False)
    eigenvals, eigenvects = np.linalg.eig(A)

    eig.append(eigenvals)
    sing.append(singularvals)

eig_arr = np.array(eig)
sing_arr = np.array(sing)
print(sing_arr)

for i in np.arange(4):
    ax[0, i].plot(ε_arr, np.real(eig_arr[:, i]), 'limegreen')
    ax[0, i].plot(ε_arr, np.imag(eig_arr[:, i]),'orange')
    ax[1, i].plot(ε_arr, np.real(sing_arr[:, i]), 'limegreen')

    #graphics ------------------------------------------------------
    s = str(i+1)
    ax[0, i].set_xlabel('$\\epsilon$', fontsize = 18)
    ax[0, i].set_ylabel(rf'$\lambda_{{{i+1}}}$', fontsize=18)
    ax[0, i].set_xticks([0.0e-5, 0.2e-5, 0.4e-5, 0.6e-5, 0.8e-5, 1.0e-5])
    ax[0, i].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    ax[0, i].tick_params(axis='both', labelsize=15)
    ax[0, i].grid(linestyle = '--')
    ax[0, i].set_ylim(-0.1, 0.1)

    ax[1, i].set_xlabel('$\\epsilon$', fontsize = 18)
    ax[1, i].set_ylabel(rf'$\sigma_{{{i+1}}}$', fontsize = 18)
    ax[1, i].set_xticks([0.0e-5, 0.2e-5, 0.4e-5, 0.6e-5, 0.8e-5, 1.0e-5])
    ax[1, i].set_yticks([0, 1, 2, 3, 4, 5])
    ax[1, i].tick_params(axis='both', labelsize=15)
    ax[1, i].grid(linestyle = '--')
    ax[1, i].set_ylim(0, 5)

ax[1, 3].set_ylim(0, 0.00001)
ax[1, 3].set_yticks([0.0e-5, 0.2e-5, 0.4e-5, 0.6e-5, 0.8e-5, 1.0e-5])

plt.subplots_adjust(left=0.075, bottom=0.075, right=0.98, top=0.945, wspace=0.45, hspace=0.26)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.savefig('fig_2a1.png', dpi = 300, bbox_inches = 'tight')
# plt.tight_layout()

plt.show()