import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.electronic_structure.plotter import BSPlotter
plt.rcParams['text.usetex'] = True

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

#scaliamo i dati per poter fare confronti sensati, che non dipendono dalle unità di misura
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# aggiungiamo solo ora la colonna di 1 perchè così prima abbiamo scalato le features
X_tilde = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# X_mean = X.mean(axis=0)
# X_centered = X - X.mean
# Σ = X_centered.T @ X_centered / (len(docs) - 1) # ATTENZIONE: funziona solo su dati centrati, andrebbero centrati prima!!!
# Σ = np.cov(X, rowvar=False, bias=False) # così fa tutto numpy da solo

# correlation_matrix = np.corrcoef(X_tilde_scaled, rowvar=False)

correlation_with_band_gap = np.array([
    np.corrcoef(X_scaled[:, j], Y, rowvar=True)[0, 1] #oppure [1, 0] è uguale, tanto la matrice è simmetrica
    for j in range(X_scaled.shape[1])
])

# print(correlation_with_band_gap)

# Visualizzazione
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["firebrick" if c >= 0 else "royalblue" for c in correlation_with_band_gap]

# Grafico in verticale invece che in orizzontale, per rendere leggibili i nomi delle features
feature_names_graph = ["Formation energy per atom", "Energy above hull", "Density",
            "Volume", "Number of sites", "Symmetry Number", "Fermi energy", 
            "Total magnetization", "Total magnetization \n normalized for volume", 
            "Number of magnetic sites"]
bars = ax.barh(feature_names_graph[::-1], correlation_with_band_gap[::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=1.0)
ax.set_xlabel("Correlazione di Pearson con band gap", fontsize=26, labelpad=12)
ax.set_xlim(-1, 1)
ax.set_yticks(np.arange(len(feature_names_graph[::-1])), feature_names_graph[::-1], fontsize=16)
ax.tick_params(axis ='x', labelsize=22)
for bar, value in zip(bars, correlation_with_band_gap[::-1]):
    # Piccolo offset per staccare il testo dalla barra
    offset = 0.02 if value >= 0 else -0.02
    # Allineamento: 'left' se positivo (testo parte dal bordo destro),
    # 'right' se negativo (testo finisce al bordo sinistro)
    ha = 'left' if value >= 0 else 'right'
    ax.text(
        value + offset,                          # posizione x
        bar.get_y() + bar.get_height() / 2,      # centro verticale della barra
        f'{value:.2f}',                          # testo formattato a 2 decimali
        va='center', ha=ha, fontsize=16
    )

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.2, top=0.98, right=0.98, bottom=0.12)
plt.savefig('images/1_1.png', dpi = 300)
plt.show()