import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
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

# IMPLEMENTO LE REGOLARIZZAZIONI ------------------------------------------------------------------------------------------
# Scelgo la griglia per il parametro di regolarizzazione alpha ("grid search" con "cross-validation CV")
# scala logaritmica per avere un numero uguale di punti per ogni ordine di grandezza, esplorando in modo uniforme lo spazio
alphas = np.logspace(-3, 3, 100)

Ridge = linear_model.RidgeCV(alphas=alphas)
Lasso = linear_model.LassoCV(alphas=alphas)

Ridge.fit(X_scaled, Y_train)
Lasso.fit(X_scaled, Y_train)

print("Miglior parametro di regolarizzazione Ridge:", Ridge.alpha_)
print("Miglior parametro di regolarizzazione LASSO:", Lasso.alpha_)

print("Coefficienti Ridge:", Ridge.coef_)
print("Coefficienti LASSO:", Lasso.coef_)

R2_Ridge = r2_score(Y_test, Ridge.predict(X_test_scaled))
R2_Lasso = r2_score(Y_test, Lasso.predict(X_test_scaled))

print(R2_Ridge)
print(R2_Lasso)

# Grafico per visualizzare come variano i coefficienti tra Ridge e Lasso
feature_names_graph = ["Formation energy per atom", "Energy above hull", "Density",
            "Volume", "Number of sites", "Symmetry Number", "Fermi energy", 
            "Total magnetization", "Total magnetization \n normalized for volume", 
            "Number of magnetic sites"]

df = pd.DataFrame({
    "LASSO": Lasso.coef_,
    "Ridge": Ridge.coef_,
    "Lineare": β_opt[1:]
}, index=feature_names_graph)

fig, ax = plt.subplots()
df = df.iloc[::-1] # inverto le righe del dataframe per plottare le features dall'alto verso il basso e non viceversa!
df.plot(kind='barh', ax=ax, color=['forestgreen', 'darkorange', 'royalblue'], width=0.7)
ax.axvline(0, color="black", linewidth=1.0)
ax.set_xlabel("Coefficienti ottimali ottenuti con regressione lineare\ne regolarizzazioni Ridge e LASSO", fontsize=22, labelpad=12)
ax.tick_params(axis='both', labelsize=18)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=20)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.2, bottom=0.14, right=0.98, top=0.97)
plt.savefig('images/1_3.png', dpi = 300)
plt.show()


# Questo fa la stessa cosa delle funzioni RidgeCV e LassoCV ma serve fare un for esplicito per poter visualizzare i regularization paths, perchè non riuscirei ad 
# ottenere tutte le informazioni che mi servono per i grafici da RidgeCV e LassoCV.
# Si noti che non c'è incoerenza tra gli algoritmi di RidgeCV e LassoCV e quest'implementazione con il for perchè questo è esattamente ciò che le funzioni già pronte fanno al loro interno
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()

train_errors_ridge = []
train_errors_lasso = []
test_errors_ridge = []
test_errors_lasso = []
coefs_ridge = []
coefs_lasso = []

for α in alphas:
    ridge.set_params(alpha=α)
    ridge.fit(X_scaled, Y_train) # non serve passare X_tilde con la colonna di 1
    coefs_ridge.append(ridge.coef_)

    # uso R^2 come stima della performance della predizione
    train_errors_ridge.append(ridge.score(X_scaled, Y_train))
    test_errors_ridge.append(ridge.score(X_test_scaled, Y_test))

    lasso.set_params(alpha=α)
    lasso.fit(X_scaled, Y_train)
    coefs_lasso.append(lasso.coef_)

    # uso R^2 come stima della performance della predizione
    train_errors_lasso.append(lasso.score(X_scaled, Y_train))
    test_errors_lasso.append(lasso.score(X_test_scaled, Y_test))


fig, ax = plt.subplots(1, 2)

# Converto le liste in matrici numpy: shape (n_alphas, n_features), ogni colonna è una feature, ogni riga un valore di alpha
coefs_ridge_arr = np.array(coefs_ridge)
coefs_lasso_arr = np.array(coefs_lasso)
# Plotto curva per curva, assegnando la label corrispondente
for i, feature_name in enumerate(feature_names_graph):
    ax[0].semilogx(alphas, coefs_ridge_arr[:, i], label=feature_name)
    ax[1].semilogx(alphas, coefs_lasso_arr[:, i], label=feature_name)

ax[0].set_xlabel(r"Parametro di regolarizzazione $\alpha$", fontsize=22, labelpad=6)
ax[0].set_ylabel(r"Coefficienti $\beta$", fontsize=22, labelpad=12)
ax[0].set_title(r"Ridge — Percorso di regolarizzazione", fontsize=24, pad=12)
ax[0].axhline(0, color="black", linewidth=0.8)
ax[0].tick_params(axis='both', labelsize=18)

ax[1].set_xlabel(r"Parametro di regolarizzazione $\alpha$", fontsize=22, labelpad=6)
ax[1].set_ylabel(r"Coefficienti $\beta$", fontsize=22, labelpad=12)
ax[1].set_title(r"Lasso — Percorso di regolarizzazione", fontsize=24, pad=12)
ax[1].axhline(0, color="black", linewidth=0.8)
ax[1].tick_params(axis='both', labelsize=18)

# Prendo gli handle solo da ax[0]: le label sono le stesse in entrambi i pannelli
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), fontsize=16)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.subplots_adjust(left=0.1, bottom=0.23, right=0.98, top=0.93, wspace=0.25)
plt.savefig('images/1_3.1.png', dpi=300)
plt.show()