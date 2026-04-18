import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()


N = 200
# Generazione con loop di python, più lenta perchè NumPy è progettato per lavorare velocemente in modo vettoriale
# X = []
# Y = []

# for i in np.arange(N):
#     x = np.random.rand()
#     ε = np.random.uniform(low = -0.2, high = 0.2)
#     y = np.sin(2 * np.pi * x) + ε
#     X.append(x)
#     Y.append(y)

# X_arr = np.array(X)
# Y_arr = np.array(Y)

np.random.seed(425526971) #BIG prime number
# Generazione vettoriale, se passo N a np.random.rand() questa crea in automatico un array di N elementi
X = np.random.rand(N)
# notare che o scrivo np.random.rand(N, -0.2, 0.2) oppure devo mettere la keyword in tutti gli argomenti, altrimenti non funziona
ε = np.random.uniform(low = -0.2, high = 0.2, size=N)
Y = np.sin(2 * np.pi * X) + ε

x_var = np.linspace(0, 1, N)

ax.plot(X, Y, '.', color='blue', label='Punti generati')
ax.plot(x_var, np.sin(2 * np.pi * x_var), color='red', label='Curva reale')
ax.legend(fontsize=22)

ax.set_xlabel(rf'$x_{{i}}$', fontsize=28)
ax.set_ylabel(rf'$y_{{i}}$', fontsize=28)
ax.tick_params(axis = "both", labelsize= 20)
ax.grid(linestyle = '--')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
# ax.set_aspect('equal', adjustable='box')
# plt.axis('equal')
plt.savefig('1_sine.png', dpi = 300)

plt.show()