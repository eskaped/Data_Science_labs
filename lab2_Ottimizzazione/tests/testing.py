import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Generating the dataset ---------------------------------------------------
N = 200
D = 14

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
    # y_arr = np.array(y)
    # return y_arr
    return y

def Grad(N, x, y, w, D):
    grad = []
    for j in range(D):
        sum_i = 0
        for i in range(N):
            sum_i += (y[i] - Polynomial(x[i], w, D) * (- (x[i] **j)))
        grad.append((1/N) * sum_i)
    return np.array(grad)

def Gradient_Descent(param, epochs = 100000, η = 0.1):
    for j in range(epochs):
        ax.plot(x_var, Polynomial(x_var, param, D), '+')
        Δ_param = - η * Grad(N, X_arr, Y_arr, param, D)
        param += Δ_param
    return param


x_var = np.linspace(0, 1, N)

ax.plot(x_var, Polynomial(x_var, Gradient_Descent(w), D), 'g+')
ax.plot(X_arr, Y_arr, '.')

ax.plot(x_var, np.sin(2 * np.pi * x_var), 'r')
plt.show()