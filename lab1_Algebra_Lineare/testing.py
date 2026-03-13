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
    ax[0, i].plot(ε_arr, np.real(eig_arr[:, i]), 'r')
    ax[0, i].plot(ε_arr, np.imag(eig_arr[:, i]),'b')
    ax[1, i].plot(ε_arr, np.real(sing_arr[:, i]), 'r')
    # ax[0, i].set_xlabel(r"$ \alpha $") either use raw strings (r"$blablabla$") or escape backslashes ("$\\blablabla$")
    ax[0, i].set_xlabel("$\\alpha $")

    #graphics ------------------------------------------------------
    ax[0, 1].set

plt.show()

# print(u"\u221A")