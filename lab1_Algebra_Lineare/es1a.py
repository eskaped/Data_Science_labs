import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


A = np.array([[-1, 1], [-1, -1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print('eigenvalues : ', eigenvalues)
print('eigenvectors : ', eigenvectors)

fig, ax = plt.subplots(2, 2)

eigenvectors_inv = np.linalg.inv(eigenvectors)

N = 100
a00 = []
a01 = []
a10 = []
a11 = []

for t in np.linspace(0, 5, N) :
    D = np.diag(np.exp(eigenvalues * t))
    expAt = eigenvectors @ D @ eigenvectors_inv
    # print(expAt)
    a00.append(expAt[0, 0])
    a01.append(expAt[0, 1])
    a10.append(expAt[1, 0])
    a11.append(expAt[1, 1])

a00_arr = np.array(a00)
a01_arr = np.array(a01)
a10_arr = np.array(a10)
a11_arr = np.array(a11)


ax[0, 0].plot(np.linspace(0, 5, N), np.real(a00_arr), 'limegreen')
ax[0, 0].plot(np.linspace(0, 5, N), np.imag(a00_arr), 'darkorange')
ax[0, 1].plot(np.linspace(0, 5, N), np.real(a01_arr), 'limegreen')
ax[0, 1].plot(np.linspace(0, 5, N), np.imag(a01_arr), 'darkorange')
ax[1, 0].plot(np.linspace(0, 5, N), np.real(a10_arr), 'limegreen')
ax[1, 0].plot(np.linspace(0, 5, N), np.imag(a10_arr), 'darkorange')
ax[1, 1].plot(np.linspace(0, 5, N), np.real(a11_arr), 'limegreen')
ax[1, 1].plot(np.linspace(0, 5, N), np.imag(a11_arr), 'darkorange')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
for i in range(2) : 
    for j in range(2) :
        ax[i, j].set_xlabel('t', fontsize = 14)
        ax[i, j].set_ylabel(rf'$e^{{At}}_{{{i, j}}}$', fontsize = 16)
        ax[i, j].tick_params(axis = "both", labelsize= 13)
        ax[i, j].set_xlim(0, 5)
        ax[i, j].set_ylim(-1, 1)
        ax[i, j].grid(linestyle = '--')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.savefig('fig_1a.png', dpi = 300, bbox_inches = 'tight')

plt.show()