import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# fig, ax = plt.subplots(1, 2)
fig, ax = plt.subplots()


# Generating the dataset -------------------------------------------------------
N = 200
D = 8

η = 0.01
etas = [0.001, 0.01, 0.1, 0.3, 1.0]
# etas = [0.1, 0.2, 0.3, 0.4, 1.0] eta di prova per testare l'algoritmo
colors = ['gold', 'orange', 'green', 'blue', 'purple']
epochs = 100000
x_var = np.linspace(0, 1, N)

np.random.seed(425526971) #BIG prime number
X = np.random.rand(N)
ε = np.random.uniform(low = -0.2, high = 0.2, size=N)
Y = np.sin(2 * np.pi * X) + ε

# Polynomial regression model --------------------------------------------------
w_0 = np.random.uniform(low=-0.5, high=0.5, size=(D + 1, ))

def Polynomial(x,w, D):
    return (np.vander(x, D + 1, increasing=True) @ w).ravel()

def Cost(N, x, y, w):
    return (1/(2*N)) * np.sum((y - Polynomial(x, w, D)) ** 2)

# Gradient function ------------------------------------------------------------
def Gradient(N, x, y, w, D):
    return (1/N) * (Polynomial(x, w, D) - y) @ np.vander(x, D + 1, increasing=True)

# Grdadient Descent function ---------------------------------------------------
def Gradient_Descent(w, η, epochs):
    cost_gd = []
    w = w.copy()
    for j in range(epochs):
        Δw = - η * Gradient(N, X, Y, w, D)
        w += Δw
        cost_gd.append(Cost(N, X, Y, w))
    return w, cost_gd

for eta_index, eta in enumerate(etas):
    w_trained , cost = Gradient_Descent(w_0, eta, epochs)
    ax.plot(x_var, Polynomial(x_var, w_trained, D), color=colors[eta_index], label=rf'$\eta = {eta}$', alpha=0.6)
# ax.plot(x_var, Polynomial(x_var, Gradient_Descent(w_0, η, epochs), D), color='blue', label='Predizione del modello')
ax.plot(x_var, np.sin(2 * np.pi * x_var), color='red', label=rf'Curva reale') # seno ideale senza rumore
ax.set_xlabel(rf'$x$', fontsize=30)
ax.set_ylabel(rf'$y$', fontsize=30)
ax.tick_params(axis ='both', labelsize=20)
ax.grid(linestyle='--')
ax.legend(fontsize=24)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.savefig('3_poly_regression_with_gd.png', dpi = 300)

plt.show()