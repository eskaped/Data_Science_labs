import numpy as np

# a) Generazione dei dati ------------------------------------------------------
N_pairs = 10000
pairs = np.random.randint(low=0, high=1000, size=(N_pairs, 2))
pairs_sums = pairs.sum(axis=1)

print(pairs[0:10])
print(pairs_sums[0:10])