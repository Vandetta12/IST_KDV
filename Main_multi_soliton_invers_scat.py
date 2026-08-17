# Import
import numpy as np
from Mobius_mpas import (Mobius_arc)
from Contours import (Dico_courbe, Rayon_cercles, visualisation_contour_cercle, Add_saut_cercle_to_dico,
                      X_phys_glob_G_glob_W_glob_x_glob, Int_cheb_mult_contour)
from Cauchy_transform import C_moin_assambalge_borne, Operateur, Non_homogène_deriv_x
from Fonction_utile import multi_soliton_phys, conv_amp_defa_to_pole_norm
from matplotlib import pyplot as plt

def Scatt_invers_multi_soliton(N_interpol_courb, A, delta, x, t,scattering_data, sauvgarde = False):
    print("x :", x)
    print("t :", t)
    #import
    rho_points = scattering_data["rho_points"]
    int_rho = scattering_data["int_rho"]
    z_0_im = scattering_data["z_0_im"]
    N_pole = len(z_0_im)
    z_0_im_alt = []
    C_list_alt = []

    for ii in range(N_pole):
        z_0, C_0 = conv_amp_defa_to_pole_norm(A[ii], delta[ii])
        z_0_im_alt.append(z_0)
        C_list_alt.append(C_0 * 1j)
    C_list = scattering_data["C_list"]
    #print("C_list, C_list_alt :", C_list, C_list_alt)
    #print("z_0_im, z_0_im_alt :", z_0_im, z_0_im_alt)

    C_list = C_list_alt.copy()
    z_0_im = z_0_im_alt.copy()
    # Def du contour

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
        Add_saut_cercle_to_dico(dico1, C_list[jj], z_0_im[jj], x, t)
        dico2 = Dico_courbe(N_interpol_courb, Mob2, 1)
        Add_saut_cercle_to_dico(dico2, C_list[jj], z_0_im[jj], x, t)
        list_dico_plus.append(dico1)
        list_dico_plus.append(dico2)

        Mob1_A, Mob1_B, Mob1_C, Mob1_D = Mobius_arc(-z_0_im[jj] * 1j, r, theta_arc_1[1], theta_arc_1[0])
        Mob1 = (Mob1_A, Mob1_B, Mob1_C, Mob1_D)
        Mob2_A, Mob2_B, Mob2_C, Mob2_D = Mobius_arc(-z_0_im[jj] * 1j, r, theta_arc_2[1], theta_arc_2[0])
        Mob2 = (Mob2_A, Mob2_B, Mob2_C, Mob2_D)
        dico1 = Dico_courbe(N_interpol_courb, Mob1, 1)
        Add_saut_cercle_to_dico(dico1, C_list[jj], -z_0_im[jj], x, t)
        dico2 = Dico_courbe(N_interpol_courb, Mob2, 1)
        Add_saut_cercle_to_dico(dico2, C_list[jj], -z_0_im[jj], x, t)
        list_dico_moin.append(dico1)
        list_dico_moin.append(dico2)

    liste_dico = list_dico_plus + list_dico_moin

    # Visualisation du contour
    # visualisation_contour_cercle(liste_dico, z_0_im)

    # Operateur de Cauchy
    Cmoin = C_moin_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)

    # Coordonnée et matrice de saut globale
    X_glob, G_glob, W_glob, W_x_glob = X_phys_glob_G_glob_W_glob_x_glob(liste_dico, N_interpol, N_interpol_courb)

    # Construction du problème RHP discretiser sur les contour et des équation linéaire associé
    op, b = Operateur(W_glob, Cmoin, N_interpol)

    U = np.linalg.solve(op, b)                 # Résolution de celui-ci
    U_phys = U.reshape(-1, 2)                  # On remmmet le vecteur mis a plat en deux colones pour les deux composantes

    # Construction et résolution de la dérivée partielle en x de RHP
    b_x = Non_homogène_deriv_x(W_x_glob, Cmoin, N_interpol, U_phys)

    U_phys_x_plat = np.linalg.solve(op, b_x)
    U_phys_x = U_phys_x_plat.reshape(-1, 2)    # Ideme que pour U_phys


    # Reconstruction de la solutio en x,t par intégration sur le contour
    integ_U_x = Int_cheb_mult_contour(liste_dico, U_phys_x, 0)

    #print("int U_x :", integ_U_x)
    q_x = -integ_U_x * 2 * 1j
    #print("q_x :", q_x)

    # Verification
    q_theorique = multi_soliton_phys(A, delta, np.array([x]), t)
    #print("q_theorique :", q_theorique[0])
    #print("Erreur :", np.abs(q_theorique[0] - q_x) / np.abs(q_theorique[0]) * 100, "%")

    # Sauvgarde
    if sauvgarde == True:

        np.savez("Scattering_invers_multi_soliton_" + str(N_pole) + "pole_" + str(N_interpol) + "_en_"
                 +"_x_" + str(x)+ "_t_" + str(t) + ".npz",
             A=A, delta=delta, X_glob=X_glob, dico=liste_dico, U_phys=U_phys, U_phys_x=U_phys_x, z_list=z_0_im)




    return q_x, U_phys, U_phys_x

# import des donnée de scattering
A = [2.4 , 1, 8]
delta = [10, 0, 5]

scattering_data = np.load("Scattering_data_trois_soliton_A_" + str(A) + "_Delt_" + str(delta) + ".npz" )

# param
N_interpol_courb = 100
T, X = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 100, 100))
Q = np.zeros(T.shape, dtype=complex)
for ii in range(T.shape[0]):
    for jj in range(T.shape[1]):
        Q[ii,jj], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A, delta, X[ii, jj], T[ii,jj], scattering_data)
        print(ii,jj)

np.savez("Q_trois_soliton_A_" + str(A) + "_Delt_" + str(delta) + "_Grid_" + str(T.shape) + ".npz", Q=Q, X=X, T=T)


plt.pcolormesh(T, X, np.real(Q), cmap="plasma", shading="auto")
plt.contour(T, X, np.real(Q), colors="white")
plt.colorbar(label="Q")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

Q_theorique = multi_soliton_phys(A,delta, X,T)

plt.pcolormesh(T, X, Q_theorique, cmap="plasma", shading="auto")
plt.contour(T, X, Q_theorique, colors="white")
plt.colorbar(label="Q_theorique")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

Err= np.abs(Q_theorique - Q)

plt.pcolormesh(T, X, Err, cmap="plasma", shading="auto")
plt.contour(T, X, Err, colors="white")
plt.colorbar(label="Erreur")
plt.xlabel("t")
plt.ylabel("x")
plt.show()


















