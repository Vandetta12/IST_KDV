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
from Cauchy_transform import C_plus_assambalge_borne, C_moin_assambalge_borne


# import des donnée de scattering
A = [2.4 , 1, 8]
delta = [10, 0, 5]
scattering_data = np.load("Scattering_data_trois_soliton_A_" + str(A) + "_Delt_" + str(delta) + ".npz" )
X_Fourrier = scattering_data["X_Fourrier"]
X_cheb = scattering_data["X_cheb"]
L = scattering_data["L"]
rho_points = scattering_data["rho_points"]
int_rho = scattering_data["int_rho"]
Q_cheb = scattering_data["Q_cheb"]
rho_array = scattering_data["rho_array"]
z_0_im = scattering_data["z_0_im"]
C_list = scattering_data["C_list"]

N_pole = len(z_0_im)
S = np.linspace(int_rho[0],int_rho[1],rho_points)

x, t = 10 , 0

# Def du contour

N_interpol_courb = 200
N_interpol = N_interpol_courb * 4 * N_pole
r = Rayon_cercles(z_0_im)
list_dico_plus = []
list_dico_moin = []

theta_arc_1 = (0, np.pi)
theta_arc_2 = (np.pi, 2 * np.pi)

for jj in range(N_pole):

    Mob1_A, Mob1_B, Mob1_C, Mob1_D = Mobius_arc(z_0_im[jj] * 1j, r, theta_arc_1[0], theta_arc_1[1])
    Mob1 = (Mob1_A, Mob1_B, Mob1_C, Mob1_D)
    Mob2_A, Mob2_B, Mob2_C, Mob2_D = Mobius_arc(z_0_im[jj] * 1j, r, theta_arc_2[0], theta_arc_2[1])
    Mob2 = (Mob2_A, Mob2_B, Mob2_C, Mob2_D)
    dico1 = Dico_courbe(N_interpol_courb, Mob1, 1)
    Add_saut_cercle_to_dico(dico1, C_list[jj], z_0_im[jj] , x,t )
    dico2 = Dico_courbe(N_interpol_courb, Mob2, 1)
    Add_saut_cercle_to_dico(dico2, C_list[jj], z_0_im[jj] , x, t)
    list_dico_plus.append(dico1)
    list_dico_plus.append(dico2)


    Mob1_A, Mob1_B, Mob1_C, Mob1_D = Mobius_arc(-z_0_im[jj] * 1j, r, theta_arc_1[1], theta_arc_1[0])
    Mob1 = (Mob1_A, Mob1_B, Mob1_C, Mob1_D)
    Mob2_A, Mob2_B, Mob2_C, Mob2_D = Mobius_arc(-z_0_im[jj] * 1j, r, theta_arc_2[1], theta_arc_2[0])
    Mob2 = (Mob2_A, Mob2_B, Mob2_C, Mob2_D)
    dico1 = Dico_courbe(N_interpol_courb, Mob1, 1)
    Add_saut_cercle_to_dico(dico1, C_list[jj], -z_0_im[jj] , x, t)
    dico2 = Dico_courbe(N_interpol_courb, Mob2, 1)
    Add_saut_cercle_to_dico(dico2, C_list[jj], -z_0_im[jj] , x, t)
    print("dico1 G 5:", dico1['G'][5,:,:])
    list_dico_moin.append(dico1)
    list_dico_moin.append(dico2)

liste_dico = list_dico_plus + list_dico_moin


visualisation_contour_cercle(liste_dico, z_0_im)

Cplus = C_plus_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)

Cmoin = C_moin_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)

X_glob, G_glob, W_glob = X_phys_glob_G_glob_W_glob(liste_dico, N_interpol, N_interpol_courb)



