# Import
import numpy as np
from matplotlib import pyplot as plt
from Inv_scattering.Main_multi_soliton_invers_scat import Scatt_invers_multi_soliton





A = [1, 5, 2]
delta = [5, 10, 20]

N = np.arange(10, 80, 1)  # on commence à 1 pour éviter N=0
scattering_data = " "

x = 100
t = 50

erreur_array = np.zeros(len(N))
i=0
for n in N:
    N_interpol_courb = n
    print("N_interpol_courb :", N_interpol_courb)
    q_x, _,_,erreur = Scatt_invers_multi_soliton(
        int(N_interpol_courb),
        A,
        delta,
        x,
        t,
        scattering_data,
        sauvgarde=False
    )


    erreur_array[i] = erreur[0] / (100 * np.abs(q_x))
    i +=1

# Création de deux sous-graphes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graphe linéaire
ax1.scatter(N, erreur_array, color='r', marker='o', s=20,
            label='Erreur relative')
ax1.set_xlabel(r'$N_{\mathrm{interpol}}$')
ax1.set_ylabel('Erreur relative (%)')
ax1.set_title('Échelle linéaire')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Graphe logarithmique
ax2.scatter(N, erreur_array, color='b', marker='o', s=20,
            label='Erreur relative')
ax2.set_xlabel(r'$N_{\mathrm{interpol}}$')
ax2.set_ylabel('Erreur relative (%)')
ax2.set_title('Échelle logarithmique')
ax2.set_yscale('log')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()