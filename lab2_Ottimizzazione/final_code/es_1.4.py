import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(1, 2)
# fig, ax = plt.subplots(5, 2)

N = 200
D = 8
η = 0.8
etas = [0.001, 0.01, 0.05, 0.1, 0.5]
epochs = 100000
gd_evo_interval = epochs // 10
x_var = np.linspace(0, 1, N)


np.random.seed(425526971) #BIG prime number
# Generazione vettoriale, se passo N a np.random.rand() questa crea in automatico un array di N elementi
X = np.random.rand(N)
# notare che o scrivo np.random.rand(N, -0.2, 0.2) oppure devo mettere la keyword in tutti gli argomenti, altrimenti non funziona
ε = np.random.uniform(low=-0.2, high=0.2, size=N)
Y = np.sin(2 * np.pi * X) + ε

# Polynomial regression model --------------------------------------------------
w_0 = np.random.uniform(low=-0.5, high=0.5, size=(D + 1, ))

# attenzione: usare D + 1 come numero di righe di w e di colonne di Polynomial (vander) per avere il giusto numero di gradi
def Polynomial(x, w, D):
    # w = np.random.uniform(low = -0.5, high = 0.5, size=((D + 1 ), 1)) # DA COMMENTARE E DA CAMBIARE w_0 CON D+1
    # Polynomial ritorna un vettore colonna!!! (N, 1) NOT ANYMORE, LOL, see row below
    # using .ravel() to reshape the (N, 1) 2D array otherwise returned by the function to a (N, ) 1D vector in order to fix shape issues
    # with the gradient function, which would otherwise return a Nx(D+1) matrix instead of a (1, D+1) vector
    return (np.vander(x, D + 1, increasing=True) @ w).ravel()


def Cost(N, x, y, w):
    #può darsi che serva specificare l'asse su cui sommare perchè sto usando vettori colonna (N, 1) e non quelli standard di NumPy (N, )
    #falso, alla fine ho fatto con vettori (N, ) e pertanto non serve specificare axis=1
    return (1/(2*N)) * np.sum((y - Polynomial(x, w, D)) ** 2)

def Gradient(N, x, y, w, D):
    # sto facendo (y_pred - y) per implementare già il segno meno che viene fuori con il gradiente
    return (1/N) * (Polynomial(x, w, D) - y) @ np.vander(x, D + 1, increasing=True)

def Gradient_Descent(w, η, epochs, interval):
    cost_gd = []
    w = w.copy()
    for j in range(epochs):
        if (j % interval == 0):
            ax[0].plot(x_var, Polynomial(x_var, w, D), alpha=0.3)
        Δw = - η * Gradient(N, X, Y, w, D)
        cost_gd.append(Cost(N, X, Y, w))
        w += Δw
    return w, cost_gd


w_trained, cost = Gradient_Descent(w_0, η, epochs, gd_evo_interval)

# PLOTTING ---------------------------------------------------------------------
ax[0].plot(x_var, Polynomial(x_var, w_trained, D), color='blue', linewidth=2, label=rf'$\eta = {η}$')
ax[0].plot(x_var, np.sin(2 * np.pi * x_var), color='red', label=rf'Curva reale') # seno ideale senza rumore
ax[0].scatter(X, Y, color='black', s=5, alpha=0.7, label='Punti generati')

ax[0].set_xlabel(rf'$x$', fontsize=24)
ax[0].set_ylabel(rf'$y$', fontsize=24)
ax[0].tick_params(axis ='both', labelsize=20)
ax[0].grid(linestyle='--')
ax[0].legend(fontsize=22)

ax[1].plot(cost, color='green')

ax[1].set_xlabel(rf'$Numero\ di\ iterazioni$', fontsize=24)
ax[1].set_ylabel(rf'$Funzione\ costo$', fontsize=24)
ax[1].tick_params(axis ='both', labelsize=20)
# ax[1].set_yscale('log') # ha senso per evidenziare meglio la riduzione esponenziale della funzione costo, altrimenti va giù in poche iterazioni e poi resta a zero
ax[1].grid(linestyle='--')

# for eta in range(len(etas)):
#     w_0 = np.random.uniform(low = -0.5, high = 0.5, size=(D + 1, ))
#     ax[eta, 0].plot(x_var, Polynomial(x_var, Gradient_Descent(w_0, eta, epochs, gd_evo_interval), D), 'b')
#     ax[eta, 0].plot(x_var, np.sin(2 * np.pi * x_var), 'r')
#     ax[eta, 1].plot(cost_gd, label=f"η={eta}")

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.subplots_adjust(left=0.09, right=0.97, top=0.93)
plt.savefig('4_monitoraggio_convergenza.png', dpi = 300)

plt.show()