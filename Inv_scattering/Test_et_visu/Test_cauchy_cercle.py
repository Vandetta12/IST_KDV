# Import
import numpy as np
import matplotlib.pyplot as plt
from Collocation_chebychev import Cheb_point, Diff_cheb_1
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa, multi_soliton_scat_data,
                            multi_soliton_phys, try_pole, conv_amp_defa_to_pole_norm,  Theta )
from Contours import Dico_courbe, Saut_cercle, Rayon_cercles, visualisation_contour_cercle
from Cauchy_transform import C_plus_assambalge_borne, C_moin_assambalge_borne

# code

N_interpol_courb = 100
N_interpol = N_interpol_courb * 2
z = 0
r = 1
theta_arc_1 = (0, np.pi)
theta_arc_2 = (np.pi, 2 * np.pi)


Mob1_A, Mob1_B, Mob1_C, Mob1_D = Mobius_arc(z, r, theta_arc_1[0], theta_arc_1[1])
Mob1 = (Mob1_A, Mob1_B, Mob1_C, Mob1_D)
Mob2_A, Mob2_B, Mob2_C, Mob2_D = Mobius_arc(z, r, theta_arc_2[0], theta_arc_2[1])
Mob2 = (Mob2_A, Mob2_B, Mob2_C, Mob2_D)
dico1 = Dico_courbe(N_interpol_courb, Mob1, 1)
dico2 = Dico_courbe(N_interpol_courb, Mob2, 1)

liste_dico = [dico1, dico2]

visualisation_contour_cercle(liste_dico, [0])

Cplus = C_plus_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)
Cmoin = C_moin_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)


u_phys = np.zeros(N_interpol, dtype=complex)
u_phys[0:N_interpol_courb] = liste_dico[0]['x_phys']
u_phys[N_interpol_courb:] = liste_dico[1]['x_phys']


for m in range(20):
    c_plus_u = Cplus @ (u_phys ** m)
    c_moin_u = Cmoin @ (u_phys ** m)

    print("m =", m)
    print("erreur :", max(np.abs(c_plus_u - (u_phys ** m))))
    print("max c_plus_u :", max(np.abs(c_plus_u)))
    print("max c_moin_u :", max(np.abs(c_moin_u)))
    print("min c_plus_u :", min(np.abs(c_plus_u)))
    print("min c_moin_u :", min(np.abs(c_moin_u)))
