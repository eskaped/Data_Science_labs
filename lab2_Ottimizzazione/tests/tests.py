import numpy as np

# x = np.array([0, 1, 2, 3, 4])
# print(np.shape(x))
# print(x ** 2)
N = 100
D = 9

def Polynomial(x, w, D):
    w = np.random.uniform(low = -0.5, high = 0.5, size=((D + 1 ), 1)) # DA COMMENTARE E DA CAMBIARE w_0 CON D+1
    # Polynomial ritorna un vettore colonna!!! (N, 1)
    return (np.vander(x, D + 1, increasing=True) @ w)

def Cost(N, x, y, w):
    #può darsi che serva specificare l'asse su cui sommare perchè sto usando vettori colonna (N, 1) e non quelli standard di NumPy (N, )
    return (1/(2*N)) * np.sum((y - Polynomial(x, w, D + 1)) ** 2, axis=1)

def Gradient(N, x, y, w):
    d = np.arange(D + 1)
    return (1/N) * np.sum((y - Polynomial(x, w, D + 1)) @ ((-x) ** d))

X = np.random.rand(N)
print(Gradient(N, X, ))