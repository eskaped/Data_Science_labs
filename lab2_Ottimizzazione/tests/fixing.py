import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)
# fig, ax = plt.subplots(5, 2)

N = 200
D = 5
η = 0.08
etas = [0.001, 0.01, 0.05, 0.1, 0.5]
epochs = 1000000
gd_evo_interval = 5000
# ottengo buoni risultati con eta=0.05, 10^6 iterazioni; eta=0.08 e stesse iter.; già un po' peggio (forse) con eta=0.1
# già con eta=0.6 si vede che il modello si sta adattando ai dati perchè all'estremità destra curva
# in realtà fino a eta=0.9 sembra che il modello funzioni abbastanza bene avvicinandosi al seno, ma con eta maggiore scoppia

cost_gd = []
x_var = np.linspace(0, 1, N)


np.random.seed(425526971) #BIG prime number
# Generazione vettoriale, se passo N a np.random.rand() questa crea in automatico un array di N elementi
X = np.random.rand(N)
# notare che o scrivo np.random.rand(N, -0.2, 0.2) oppure devo mettere la keyword in tutti gli argomenti, altrimenti non funziona
ε = np.random.uniform(low = -0.2, high = 0.2, size=N)
Y = np.sin(2 * np.pi * X) + ε

# Polynomial regression model --------------------------------------------------
w_0 = np.random.uniform(low = -0.5, high = 0.5, size=(D + 1, ))

Phi = np.vander(X, D + 1, increasing=True)

# attenzione: usare D + 1 come numero di righe di w e di colonne di Polynomial (vander) per avere il giusto numero di gradi
def Polynomial(Phi, w):
    # w = np.random.uniform(low = -0.5, high = 0.5, size=((D + 1 ), 1)) # DA COMMENTARE E DA CAMBIARE w_0 CON D+1
    # Polynomial ritorna un vettore colonna!!! (N, 1) NOT ANYMORE, LOL, see row below
    # using .ravel() to reshape the (N, 1) 2D array otherwise returned by the function to a (N, ) 1D vector in order to fix shape issues
    # with the gradient function, which would otherwise return a Nx(D+1) matrix instead of a (1, D+1) vector
    return (Phi @ w).ravel()


# def Cost(N, x, y, w):
#     #può darsi che serva specificare l'asse su cui sommare perchè sto usando vettori colonna (N, 1) e non quelli standard di NumPy (N, )
#     #falso, alla fine ho fatto con vettori (N, ) e pertanto non serve specificare axis=1
#     return (1/(2*N)) * np.sum((y - Polynomial(x, w, D)) ** 2)

def Cost(Phi, y, w):
    N = len(y)
    y_pred = Phi @ w
    return (1/(2*N)) * np.sum((y - y_pred)**2)

# def Gradient(N, x, y, w, D):
#     # d = np.arange(D + 1)
#     # return (1/N) * np.sum((y - Polynomial(x, w, D))@((-x) ** d))
#     # sto facendo (y_pred - y) per implementare già il segno meno che viene fuori con il gradiente
#     return (1/N) * (Polynomial(x, w, D) - y) @ np.vander(x, D + 1, increasing=True)

def Gradient(Phi, y, w):
    N = len(y)
    y_pred = Phi @ w
    return (1/N) * Phi.T @ (y_pred - y)

# def Gradient_Descent(w, η, epochs, interval):
#     # cost_gd = []
#     w = w.copy()
#     for j in range(epochs):
#         if (j % interval == 0):
#             ax[0].plot(x_var, Polynomial(x_var, w, D), alpha=0.3)
#         cost_gd.append(Cost(N, X, Y, w))
#         Δw = - η * Gradient(N, X, Y, w, D)
#         w += Δw
#     return w

def Gradient_Descent(Phi, y, w, η, epochs, interval):
    w = w.copy()
    cost_gd = []

    for j in range(epochs):
        # if j % interval == 0:
        #     ax[0].plot(x_var, Phi @ w, alpha=0.3)
        if j % interval == 0:
            ax[0].plot(x_grid, Phi_grid @ w, alpha=0.2)
        cost_gd.append(Cost(Phi, y, w))

        grad = Gradient(Phi, y, w)
        w -= η * grad

    return w, cost_gd

# print(Gradient(N, X, Y, w_0))
# print(np.shape(Gradient(N, X, Y, w_0)))
# print(np.shape(w_0))
# print(np.shape(Y))
# print(np.shape(Polynomial(X, w_0, D)))

# manager = plt.get_current_fig_manager()

# ax[0].plot(x_var, Polynomial(x_var, Gradient_Descent(Phi, Y, w_0, η, epochs, gd_evo_interval), D), 'bw')
# ax[0].plot(x_var, np.sin(2 * np.pi * x_var), 'r') # seno ideale senza rumore
# ax[0].scatter(X, Y, color='black', s=10)

# for eta in range(len(etas)):
#     w_0 = np.random.uniform(low = -0.5, high = 0.5, size=(D + 1, ))
#     ax[eta, 0].plot(x_var, Polynomial(x_var, Gradient_Descent(w_0, eta, epochs, gd_evo_interval), D), 'b')
#     ax[eta, 0].plot(x_var, np.sin(2 * np.pi * x_var), 'r')
#     ax[eta, 1].plot(cost_gd, label=f"η={eta}")

# x_cost = np.linspace(0, epochs, epochs)
# ax[1].set_yscale('log')
# ax[1].plot(cost_gd)

# manager.full_screen_toggle()


# plt.show()
# =========================
# SORT dei dati per plotting
# =========================
idx = np.argsort(X)
X_sorted = X[idx]
Y_sorted = Y[idx]

# =========================
# GRIGLIA LISCA PER IL MODELLO
# =========================
x_grid = np.linspace(0, 1, 300)
Phi_grid = np.vander(x_grid, D + 1, increasing=True)

# =========================
# TRAINING
# =========================
w_final, cost_gd = Gradient_Descent(Phi, Y, w_0, η, epochs, gd_evo_interval)

# =========================
# PREVISIONE MODELLO
# =========================
y_grid = Phi_grid @ w_final

# =========================
# PLOT RISULTATI
# =========================
ax[0].scatter(X_sorted, Y_sorted, color='black', s=10, label="Dati")
ax[0].plot(x_grid, np.sin(2*np.pi*x_grid), 'r', label="Funzione vera")
ax[0].plot(x_grid, y_grid, 'b', label="Modello appreso")

ax[0].legend()

# =========================
# COSTO
# =========================
ax[1].plot(cost_gd)
ax[1].set_yscale('log')
ax[1].set_title("Convergenza del costo")
ax[1].set_xlabel("Iterazioni")
ax[1].set_ylabel("J(w)")

plt.show()