# Import
import numpy as np
import matplotlib.pyplot as plt
from Collocation_chebychev import Cheb_point, Diff_cheb_1
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa, multi_soliton_scat_data,
                            multi_soliton_phys, try_pole, conv_amp_defa_to_pole_norm,  Theta, delta )
from Contours import (Dico_courbe, Saut_cercle, Rayon_cercles, visualisation_contour_cercle, Add_saut_cercle_to_dico,
                      X_phys_glob_G_glob_W_glob)
from Cauchy_transform import C_plus_assambalge_borne, C_moin_assambalge_borne, Operateur

# Import de donnée
N_pole = 3
N_interpol = 200 * 4 * N_pole
# `dico` contient des dictionnaires : NumPy l'enregistre donc comme un
# tableau d'objets Python et nécessite explicitement l'autorisation du pickle.
# Ne l'utiliser que pour un fichier `.npz` produit par ce projet et de source
# fiable, car le pickle peut exécuter du code lors du chargement.
data = np.load("Scattering_invers_multi_soliton_" + str(N_pole) + "pole_" +
               str(N_interpol) + ".npz", allow_pickle=True)
lst_dico = data["dico"]
X_glob = data["X_glob"]
U_phys = data["U_phys"]

# Visualisation

plt.figure(figsize=(7, 6))

plt.scatter(X_glob.real, X_glob.imag, c=np.abs(U_phys[:, 0]), s=12, cmap="viridis")

plt.colorbar(label="|U_1|")
#plt.axis("equal")
plt.grid()
plt.title("Amplitude de U_1 sur le contour")
plt.xlabel("Re z")
plt.ylabel("Im z")
plt.show()

plt.figure(figsize=(7, 6))

plt.scatter(X_glob.real, X_glob.imag, c=np.abs(U_phys[:, 1]), s=12, cmap="viridis")

plt.colorbar(label="|U_2|")
#plt.axis("equal")
plt.grid()
plt.title("Amplitude de U_2 sur le contour")
plt.xlabel("Re z")
plt.ylabel("Im z")
plt.show()
