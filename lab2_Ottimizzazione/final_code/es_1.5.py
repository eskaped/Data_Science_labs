import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(1, 2)

N = 200
# M = 50 # minibatches' length
batch_sizes = [1, 10, 50, 100, 200]
D = 8
η = 0.08
epochs = 100000
gd_evo_interval = epochs // 10
threshold = 0.01
x_var = np.linspace(0, 1, N)


np.random.seed(425526971) #BIG prime number
# Generazione vettoriale, se passo N a np.random.rand() questa crea in automatico un array di N elementi
X = np.random.rand(N)
# notare che o scrivo np.random.rand(N, -0.2, 0.2) oppure devo mettere la keyword in tutti gli argomenti, altrimenti non funziona
ε = np.random.uniform(low = -0.2, high = 0.2, size=N)
Y = np.sin(2 * np.pi * X) + ε

# Polynomial regression model --------------------------------------------------
w_0 = np.random.uniform(low = -0.5, high = 0.5, size=(D + 1, ))

# attenzione: usare D + 1 come numero di righe di w e di colonne di Polynomial (vander) per avere il giusto numero di gradi
def Polynomial(x, w, D):
    # using .ravel() to reshape the (N, 1) 2D array otherwise returned by the function to a (N, ) 1D vector in order to fix shape issues
    # with the gradient function, which would otherwise return a Nx(D+1) matrix instead of a (1, D+1) vector
    return (np.vander(x, D + 1, increasing=True) @ w).ravel()

def Cost(N, x, y, w):
    return (1/(2*N)) * np.sum((y - Polynomial(x, w, D)) ** 2)

def Gradient(x, y, w, D):
    # sto facendo (y_pred - y) per implementare già il segno meno che viene fuori con il gradiente
    return (1/len(x)) * (Polynomial(x, w, D) - y) @ np.vander(x, D + 1, increasing=True)

def Stochastic_Gradient_Descent(w, η, epochs, interval, M):
    cost_gd = []
    w = w.copy()
    converged = False
    for epoch in range(epochs):
        # if (epoch % interval == 0):
        #     ax[0].plot(x_var, Polynomial(x_var, w, D), alpha=0.3)
        permutation = np.random.permutation(len(X))
        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]
        for j in range(0, N, M):
            Δw = - η * Gradient(shuffled_X[j: j + M], shuffled_Y[j: j + M], w, D)
            w += Δw
        # il costo va calcolato sull'intero dataset dopo ogni epoca, non su un random batch o sulla somma dei batch!!!
        cost = Cost(N, X, Y, w)
        cost_gd.append(cost)
        if (not converged) and (cost <= threshold):
            print(rf'SGD con batch size M = {M} converge in {epoch} epoche')
            converged = True
    return w, cost_gd

# w_trained, cost = Stochastic_Gradient_Descent(w_0, η, epochs, gd_evo_interval, M)

for M in batch_sizes:
    w_trained, cost = Stochastic_Gradient_Descent(w_0, η, epochs, gd_evo_interval, M)
    ax[0].plot(x_var, Polynomial(x_var, w_trained, D), label=rf'$M = {M}$', linewidth=1, alpha=0.8)
    ax[1].plot(cost, label=rf'$M = {M}$')



# # PLOTTING ---------------------------------------------------------------------
# ax[0].plot(x_var, Polynomial(x_var, w_trained, D), color='blue', linewidth=2, label=rf'$\eta = {η}$')
ax[0].plot(x_var, np.sin(2 * np.pi * x_var), color='red', label=rf'Curva reale', linewidth=2) # seno ideale senza rumore
ax[0].scatter(X, Y, color='black', s=5, alpha=0.7, label='Punti generati')

ax[0].set_xlabel(rf'$x$', fontsize=24)
ax[0].set_ylabel(rf'$y$', fontsize=24)
ax[0].tick_params(axis ='both', labelsize=20)
ax[0].grid(linestyle='--')
ax[0].legend(fontsize=20)

# ax[1].plot(cost, color='green')

ax[1].set_xlabel(rf'$Numero\ di\ iterazioni$', fontsize=24)
ax[1].set_ylabel(rf'$Funzione\ costo$', fontsize=24)
ax[1].tick_params(axis ='both', labelsize=20)
# # ax[1].set_yscale('log') # ha senso per evidenziare meglio la riduzione esponenziale della funzione costo, altrimenti va giù in poche iterazioni e poi resta a zero
ax[1].grid(linestyle='--')
ax[1].legend(fontsize=20)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.subplots_adjust(left=0.09, right=0.97, top=0.93)
plt.savefig('5_minibatch_sgd_confronto.png', dpi = 300)

plt.show()