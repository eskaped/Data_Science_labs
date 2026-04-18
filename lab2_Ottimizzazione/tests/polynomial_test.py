import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

D = 8
x = np.linspace(-10, 10, 5)

# x_py = [1., 2., 3., 4.]
# x = np.array(x_py)
# print(x)
w = np.random.uniform(low = -0.5, high = 0.5, size=(D, 1))
# w_py = [3, 5, 7]
# w = np.array(w_py)
# print(w)

def Polynomial(x, w, D):
    y = 0
    for d in range(D):
        y += w[d] * (x ** d)
    # y_arr = np.array(y)
    # return y_arr
    return y

def Poly_Vander(x, w, D):
    return (np.vander(x, D, True) @ w).T

# print(Polynomial(x, w, D))
# print(Poly_Vander(x, w, D))


ax[0].plot(x, Polynomial(x, w, D))
ax[1].plot(x, Poly_Vander(x, w, D).T)   # NOTA: PER PLOTTARE VANDER BISOGNA FARE LA TRASPOSTA (????)
plt.show()