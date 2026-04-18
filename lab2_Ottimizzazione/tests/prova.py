import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([3, 4, 5])
m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
print(m @ x.T)
print(x @ y)