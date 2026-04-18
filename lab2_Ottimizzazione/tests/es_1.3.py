import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)


# Generating the dataset ---------------------------------------------------
N = 200
D = 8

X = []
Y = []

for i in range(N):
    x = np.random.rand()
    ε = np.random.uniform(low = -0.2, high = 0.2)
    y = np.sin(2 * np.pi * x) + ε
    X.append(x)
    Y.append(y)
X_arr = np.array(X)
Y_arr = np.array(Y)

# Polynomial regression model ----------------------------------------------
w = np.random.uniform(low = -0.5, high = 0.5, size=(D, 1))
# w = [0, 2*np.pi, 0, -((2*np.pi)**3)/6, 0, ((2*np.pi)**5)/120, 0, -((2*np.pi)**7)/5040]

def Polynomial(x, w, D):
    y = 0
    for d in range(D):
        y += w[d] * (x ** d)
    y_arr = np.array(y)
    return y_arr

# Cost function J(w) --------------------------------------------------------
def Cost(N, x, y, w, D):
    cost_i = 0
    for i in range(N):
        cost_i += ((y[i] - Polynomial(x[i], w, D)) **2)
    return (1/N) * cost_i

# # Cost function J(w) -----------------------------------------------------------
# def Cost(N, x, y, param, D):
#     for i in range(y.shape[1]):
#         sum = 0
#         sum += ((y - Polynomial(x, param, D).T) * (y - 
#                                                    Polynomial(x, param, D).T) )[i]
#     return (1/(2*N)) * sum
#     # cost_i = 0
#     # for i in range(N):
#     #     cost_i += ((y[i] - Polynomial(x[i], param, D)) **2)
#     # return (1/N) * cost_i

# Gradient function ---------------------------------------------------------
def Grad(N, x, y, w, D):
    grad = []
    for j in range(D + 1):
        sum_i = 0
        for i in range(N):
            sum_i += (y[i] - Polynomial(x[i], w, D) * (- (x[i] **j)))
        grad.append((1/N) * sum_i)
    return np.array(grad)

# grad = []
    # for d in range(D):
    #     grad.append((1/N) * (y - Polynomial(x, w, D))@(-x ** d))
    # return np.array(grad)


def Gradient_Descent(param, epochs = 200, η = 0.01):
    cost = []
    for j in range(epochs):
        ax[0].plot(X_arr, Polynomial(X_arr, param, D), '+')
        Δ_param = - η * Grad(N, X_arr, Y_arr, param, D)
        param += Δ_param
        # print(Cost(N, X_arr, Y_arr, param, D))
        cost.append(Cost(N, X_arr, Y_arr, param, D))
    cost_arr = np.array(cost)
    return (param, cost_arr)

# def Gradient_Descent(param, epochs = 200, η = 0.01):
#     for j in range(epochs):
#         # NOTA: PER PLOTTARE VANDER BISOGNA FARE LA TRASPOSTA (????) COMMENTO LA RIGA PERCHè CON TANTE ITERAZIONI CI METTE UN FOTTIO A DISEGNARE
#         # ax[0].plot(X_arr, Polynomial(X_arr, param, D).T, '+')
#         param_arr.append(param)
#         Δ_param = - η * Grad(N, X_arr, Y_arr, param, D)
#         param += Δ_param
#         # print(Cost(N, X_arr, Y_arr, param, D))
#     #     cost.append(Cost(N, X_arr, Y_arr, param, D))
#     # cost_arr = np.array(cost)
#     # return (param, cost_arr)
#     return param



ax[0].plot(X_arr, Polynomial(X_arr, Gradient_Descent(w)[0], D), 'b+')

x_var = np.linspace(0, 1, N)
ax[0].plot(x_var, np.sin(2 * np.pi * x_var), 'r')

ax[1].plot(Gradient_Descent(w)[1], 'r+')

plt.show()


def gd(grad, init, n_epochs=1000, eta=10**-4, noise_strength=0):
    #This is a simple optimizer
    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    v=0
    for j in range(n_epochs):
        noise=noise_strength*np.random.randn(params.size)
        v=-eta*(np.array(grad(params))+noise)
        params=params+v
        param_traj[j+1,]=params
    return param_traj