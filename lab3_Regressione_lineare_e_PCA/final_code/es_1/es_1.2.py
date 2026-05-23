import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.electronic_structure.plotter import BSPlotter
plt.rcParams['text.usetex'] = True
plt.rcParams['lines.markersize'] = 5

api_key = "BBRKH6cJvqXRu17B7ZmW5x4xqNyH0U5v"
mpr = MPRester(api_key)

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
β_opt, residuals, rank, sv = np.linalg.lstsq(X_tilde, Y_train, rcond=None)

# IMPLEMENTO IL COEFFICIENTE DI DETERMINAZIONE R^2
X_test_scaled = (X_test - X_mean) / X_std
X_test_tilde = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))
Y_predicted = X_test_tilde @ β_opt

R2_train = r2_score(Y_train, (X_tilde @ β_opt))
R2_test = r2_score(Y_test, (Y_predicted))

print(R2_train, R2_test)

# Visualizzazione di dove il modello sbaglia di più, se a materiali con basso gap o alto gap
fig, ax = plt.subplots()

plt.scatter(Y_test, Y_predicted, marker='.', color='royalblue', alpha=0.5, label="Predizioni del modello")
ax.plot([0, 10], [0, 10], color='firebrick', linestyle='--', linewidth=1, label=r"Predizione perfetta $\hat{y} = y$")
ax.set_xlim(-0.5, 9)
ax.set_ylim(-3, 5)
ax.set_xlabel("Band gap reale, da predire", fontsize=26, labelpad=12)
ax.set_ylabel("Band gap predetto dal modello lineare", fontsize=26, labelpad=16)
ax.tick_params(axis ='both', labelsize=20)
ax.set_xticks(np.arange(0, 9, 1))
ax.minorticks_on()
ax.grid(linestyle='--', color='darkgrey', alpha=0.6)
ax.legend(fontsize=20, markerscale=3)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.1, right=0.98, top=0.98)
plt.savefig('images/1_2.png', dpi = 300)
plt.show()