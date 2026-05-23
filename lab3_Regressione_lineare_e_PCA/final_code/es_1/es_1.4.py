import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.electronic_structure.plotter import BSPlotter
plt.rcParams['text.usetex'] = True


api_key = "BBRKH6cJvqXRu17B7ZmW5x4xqNyH0U5v"
mpr = MPRester(api_key)

feature_names = ["formation_energy_per_atom", "energy_above_hull", "density",
            "volume", "nsites", "symmetry.number", "efermi", 
            "total_magnetization", "total_magnetization_normalized_vol", 
            "num_magnetic_sites"]

# NOTA: si trovano 20765 materiali in questo modo
docs = mpr.materials.summary.search(
    num_elements=2,
    fields=["material_id", "formula_pretty", "band_gap", 
            "formation_energy_per_atom", "energy_above_hull", "density",
            "volume", "nsites", "symmetry.number", "efermi", 
            "total_magnetization", "total_magnetization_normalized_vol", 
            "num_magnetic_sites"]
)

def safe(x):
    return x if x is not None else 0.0

X = np.array([
    [
        safe(doc.formation_energy_per_atom),
        safe(doc.energy_above_hull),
        safe(doc.density),
        safe(doc.volume),
        safe(doc.nsites),
        safe(doc.symmetry.number),
        safe(doc.efermi),
        safe(doc.total_magnetization),
        safe(doc.total_magnetization_normalized_vol),
        safe(doc.num_magnetic_sites)
    ]
    for doc in docs])
Y = np.array([safe(doc.band_gap) for doc in docs])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=3843637271)

#ORA USIAMO SOLO IL DATASET DI TRAIN
#scaliamo i dati per poter fare confronti sensati, che non dipendono dalle unità di misura
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_scaled = (X_train - X_mean) / X_std

# aggiungiamo solo ora la colonna di 1 perchè così prima abbiamo scalato le features
X_tilde = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

correlation_with_band_gap = np.array([
    np.corrcoef(X_scaled[:, j], Y_train, rowvar=True)[0, 1] #oppure [1, 0] è uguale, tanto la matrice è simmetrica
    for j in range(X_scaled.shape[1])
])

# IMPLEMENTO LA REGRESSIONE LINEARE nella sua forma chiusa
# PRIMO MODO
optimal_params = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ Y_train

# SECONDO MODO
# β_opt = np.linalg.lstsq((X_tilde.T @ X_tilde), (X_tilde.T @ Y))
β_opt, residuals, rank, sv = np.linalg.lstsq(X_tilde, Y_train, rcond=None)

# IMPLEMENTO IL COEFFICIENTE DI DETERMINAZIONE R^2
X_test_scaled = (X_test - X_mean) / X_std
X_test_tilde = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))
Y_predicted = X_test_tilde @ β_opt

R2_train = r2_score(Y_train, (X_tilde @ β_opt))
R2_test = r2_score(Y_test, (Y_predicted))

# IMPLEMENTO ORA LA REGRESSIONE POLINOMIALE, per vedere come cambia il problema
poly_regression_degree2 = PolynomialFeatures(degree=2, include_bias=False) # grado 2, non aggiunge anche la colonna di 1
poly_regression_degree3 = PolynomialFeatures(degree=3, include_bias=False) # grado 3, non aggiunge anche la colonna di 1

X_poly_train_2 = poly_regression_degree2.fit_transform(X_scaled)
X_poly_test_2 = poly_regression_degree2.transform(X_test_scaled)
X_poly_train_3 = poly_regression_degree3.fit_transform(X_scaled)
X_poly_test_3 = poly_regression_degree3.transform(X_test_scaled)

model_poly_2 = linear_model.LinearRegression()
model_poly_2.fit(X_poly_train_2, Y_train)
model_poly_3 = linear_model.LinearRegression()
model_poly_3.fit(X_poly_train_3, Y_train)

R2_poly_train_2 = r2_score(Y_train, model_poly_2.predict(X_poly_train_2))
R2_poly_test_2 = r2_score(Y_test, model_poly_2.predict(X_poly_test_2))
R2_poly_train_3 = r2_score(Y_train, model_poly_3.predict(X_poly_train_3))
R2_poly_test_3 = r2_score(Y_test, model_poly_3.predict(X_poly_test_3))

print("R^2 on training data of polynomial model with degree=2 =", R2_poly_train_2)
print("R^2 on test data of polynomial model with degree=2 =", R2_poly_test_2)
print("R^2 on training data of polynomial model with degree=3 =", R2_poly_train_3)
print("R^2 on test data of polynomial model with degree=3 =", R2_poly_test_3)