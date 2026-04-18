import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# fig, ax = plt.subplots(2, 2)
fig, ax = plt.subplots()

degrees = [1, 2, 3, 4]
colors = ['red', 'orange', 'blue', 'green']
x = np.linspace(-10, 10, 100)

# VERSIONE INIZIALE, non ottimizzata per la ripetizione su grande scala
# w = np.random.uniform(low = -0.5, high = 0.5, size=(D + 1, 1))
# def Polynomial(x, w, D):
#     y = 0
#     for d in range(D):
#         y += w[d] * (x ** d)
#     y_arr = np.array(y)
#     return y_arr

# VERSIONE FINALE, con np.vander()
# attenzione: usare D + 1 come numero di righe di w e di colonne di Polynomial (vander) per avere il giusto numero di gradi
def Polynomial(x,w, D):
    # Polynomial ritorna un vettore colonna!!! (N, 1) # ATTENZIONE: con .ravel() non più!
    return (np.vander(x, D + 1, increasing=True) @ w).ravel()

for D in degrees:
    w = np.random.uniform(low=-0.5, high=0.5, size=((D + 1 ), ))
    ax.plot(x, Polynomial(x, w, D), color=colors[D - 1], label=rf'D = {D}')
 

ax.set_ylim(-10, 10)
ax.set_xlabel(rf'$x$', fontsize = 30)
ax.set_ylabel(rf'$\hat{{y}}(x; \mathbf{{w}})$', fontsize = 30)
ax.tick_params(axis = 'both', labelsize= 20)
ax.grid(linestyle = '--')
# ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=20)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.savefig('2_polynomials.png', dpi = 300)

plt.show()