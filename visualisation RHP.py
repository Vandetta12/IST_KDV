# Import
import numpy as np
import matplotlib.pyplot as plt
from numpy import dtypes
from Collocation_chebychev import Cheb_point, Diff_cheb_1
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa, multi_soliton_scat_data,
                            multi_soliton_phys, try_pole, conv_amp_defa_to_pole_norm,  Theta, Delta )
from Contours import (Dico_courbe, Saut_cercle, Rayon_cercles, visualisation_contour_cercle, Add_saut_cercle_to_dico,
                      X_phys_glob_G_glob_W_glob_x_glob)
from Cauchy_transform import C_plus_assambalge_borne, C_moin_assambalge_borne, Operateur, Evaluation_cauchy_grid

# Fonctions
def Visualisation_solution_RHP(U_phys, lst_dico, z_0_list,  save, load, N_grid, x_lim, y_lim):
    if load == 0:

        X, Y = np.meshgrid(np.linspace(x_lim[0], x_lim[1], N_grid), np.linspace(y_lim[0], y_lim[1], N_grid))
        Z = X + Y * 1j
        dist = 10 ** (-3)
        Phi1, Phi2 = Evaluation_cauchy_grid(Z, U_phys, lst_dico, dist)

        np.savez(save, dico=lst_dico, U_phys=U_phys, Z=Z, Real_grid=(X, Y), phy=(Phi1, Phi2), N_grid=N_grid)

    if load == 1:
        data = np.load(save, allow_pickle=True)
        lst_dico = data["dico"]
        U_phys = data["U_phys"]
        Z = data["Z"]
        X , Y = data["Real_grid"][0] , data["Real_grid"][1]
        Phi1, Phi2 = data["phy"][0] , data["phy"][1]
        N_grid = data["N_grid"]

    else :
        print("Erreur : choisissez 1 ou 0, en fonction de si vous voulez charger un fichier ou non.")

    Phi1 += 1
    Phi2 += 1
    module1 = np.abs(Phi1)
    module2 = np.abs(Phi2)
    phase1 = np.angle(Phi1)
    phase2 = np.angle(Phi2)

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, module1, cmap="viridis", shading="auto")
    plt.scatter([0,0,0], z_0_list, c="red", s=12)
    plt.scatter([0,0,0], -z_0_list, c="red", s=12)
    plt.colorbar(label="|Phi1|")
    plt.contour(X, Y, module1, colors="white")

    plt.grid()
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, module2, cmap="viridis", shading="auto")
    plt.scatter([0,0,0], z_0_list, c="red", s=12)
    plt.scatter([0,0,0], -z_0_list, c="red", s=12)
    plt.colorbar(label="|Phi2|")
    plt.contour(X, Y, module2, colors="white")
    plt.grid()
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, phase1, cmap="twilight", shading="auto")
    plt.scatter([0,0,0], z_0_list, c="red", s=12)
    plt.scatter([0,0,0], -z_0_list, c="red", s=12)
    plt.colorbar(label="arg(Phi1)")
    plt.contour(X, Y, phase1, colors="white")
    plt.grid()
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, phase2, cmap="twilight", shading="auto")
    plt.scatter([0,0,0], z_0_list, c="red", s=12)
    plt.scatter([0,0,0], -z_0_list, c="red", s=12)
    plt.colorbar(label="arg(Phi2)")
    plt.contour(X, Y, phase2, colors="white")
    plt.grid()
    plt.axis("equal")
    plt.show()

    return







# Import de donnée
N_pole = 3
N_interpol = 100 * 4 * N_pole
data = np.load("Scattering_invers_multi_soliton_3pole_3600_en__x_5.01_t_0.npz", allow_pickle=True)
lst_dico = data["dico"]
X_glob = data["X_glob"]
U_phys = data["U_phys"]
z_list = data["z_list"]

# Visualisation

plt.figure(figsize=(7, 6))

plt.scatter(X_glob.real, X_glob.imag, c=np.abs(U_phys[:, 0]), s=12, cmap="viridis")

plt.colorbar(label="|U_1|")
#plt.axis("equal")
plt.grid()
plt.title("Amplitude de U_1 sur le contour")
plt.xlabel("Re z")
plt.ylabel("Im z")
plt.axis("equal")
plt.show()

plt.figure(figsize=(7, 6))

plt.scatter(X_glob.real, X_glob.imag, c=np.abs(U_phys[:, 1]), s=12, cmap="viridis")

plt.colorbar(label="|U_2|")
#plt.axis("equal")
plt.grid()
plt.title("Amplitude de U_2 sur le contour")
plt.xlabel("Re z")
plt.ylabel("Im z")
plt.axis("equal")
plt.show()

N_grid = 100

fichier = ("Scattering_invers_multi_soliton_" + str(N_pole) + "_pole_" + str(N_interpol)
           + "_visualisation_" + str(N_grid) + ".npz")

Visualisation_solution_RHP(U_phys, lst_dico, z_list, fichier, 0, N_grid, (-3, 3), (-3, 3))