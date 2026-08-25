# Imports
import numpy as np
import matplotlib.pyplot as plt
from methodes.Collocation_chebychev import Cheb_point, Diff_cheb_1, residus, rho
from methodes.Hill_eigenvalue import (a, b , Changement_variables, grille, fourier_coeffs_shifted, Fourier_troncage,
                                      Matrice_Hill, Diag_et_filtre, eta_lier)
from misc.Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa,
                                 multi_soliton_scat_data, multi_soliton_phys, try_pole)


# Corps du code
###############
# def des grilles
# ---------------
# Fouriller  (valeur propre L2)
N_Fourrier = 500                                         # paramètre pour le tronquage des mode des Fourrier
M = (2 * N_Fourrier + 1) * 10                      # Nombre de point sur la grille pour la FFT
n, Y = grille(N_Fourrier, M)
X_Fourrier = Changement_variables(Y)

# Chebichev  (Fonction rho)
L = 50                                                 # troncage de la droite pour chebychev
N_cheb = 1001                                          # Nombre de points de collocation
rho_points = 10
int_rho = [-5, 5]
X_cheb, V = Cheb_point(L, N_cheb , oriant = 1, bord = 0, compl = False)
D = Diff_cheb_1(N_cheb)


#######################################################################################################"

# définition du potentiel
# --------------------------
A = [2.4, 1]
delta = [10, 5]




# Soliton à l'instant t = 0
Q_cheb = -multi_soliton_phys(A, delta, X_cheb, 0)
Q_Fourrier = multi_soliton_phys(A, delta, X_Fourrier, 0)



# Calcule de rho sur la droite réelle :
# ------------------------------------

U = np.diag(Q_cheb)
V_inv = np.linalg.inv(V)


D_1 = V @ D @ V_inv / L
D_2 = V @ D @ D @ V_inv / (L ** 2)

S = np.linspace(int_rho[0],int_rho[1],rho_points)
rho_list = []
for s in S:
    print(s)
    rho_s = rho(N_cheb, D_1, D_2, U, Q_cheb, s)
    rho_list.append(rho_s)
rho_array = np.array(rho_list)

#################################################################################################"

# Calcule des valeur prores
# --------------------------

a_val = a(Y)
b_val = b(Y)
Q_val = Q_Fourrier.copy()
k_a, a_chapeau = fourier_coeffs_shifted(a_val, M)
k_b, b_chapeau = fourier_coeffs_shifted(b_val, M)
k_q, Q_chapeau = fourier_coeffs_shifted(Q_val, M)

# Tronquer
k_restraint, a_chapeau = Fourier_troncage(k_a, a_chapeau, N_Fourrier)
_, b_chapeau = Fourier_troncage(k_b, b_chapeau, N_Fourrier)
_, Q_chapeau = Fourier_troncage(k_q, Q_chapeau, N_Fourrier)

H = Matrice_Hill(N_Fourrier, a_chapeau, b_chapeau, Q_chapeau)
valeurs_propres, vecteurs_propres = Diag_et_filtre(H)         # Uniquement les états lier
eta_lier_vecteur, j0 = eta_lier(vecteurs_propres, Y, N_Fourrier)


print("valeurs_propres:", valeurs_propres)

z_0_im = np.sqrt(-valeurs_propres)
print("Poles : ",z_0_im)

# Calcule des résidus de a ( les coefficient de normalisation)

b_z_0 = []
C_list= []
for jj in range(len(z_0_im)):
    j0_cheb = fined_close_to_Fourier_in_cheb(j0[jj], X_cheb, X_Fourrier)
    b_pole, Integ_mu = residus(D_1, D_2, U, Q_cheb, (z_0_im[jj] * 1j), X_cheb, j0_cheb, V_inv, L)
    b_z_0.append(b_pole)
    C_list.append((1j * (b_pole ** 2)) / Integ_mu)

print("b_z_0:", b_z_0)
print("c_z_0:", C_list)


# verification
N_pole = len(z_0_im)
A_cal_list_mes = np.zeros(N_pole)
delta_cal_list_mes = np.zeros(N_pole)
for ll in range(N_pole):
    A_cal_list_mes[ll], delta_cal_list_mes[ll] = conv_pole_norm_to_amp_defa(z_0_im[ll], np.imag(C_list[ll]))

A_cal_list, delta_cal_list = try_pole(A, delta, A_cal_list_mes, delta_cal_list_mes)

for kk in range(N_pole):

    print("Amplitude théorique , déphasage Théprique :", A[kk] , delta[kk])
    print("Amplitude calculée , déphasage calculé :", A_cal_list[kk] , delta_cal_list[kk])
    print("Err amplitude, Err déphasage :", np.abs(A_cal_list[kk] - A[kk]) , np.abs(delta_cal_list[kk] - delta[kk]))

print("Max rho error real :", max(np.abs(np.real(rho_array))))
print("Max rho error im :", max(np.abs(np.imag(rho_array))))

# visualisation
plt.plot(S, np.real(rho_array), color='red', label='real')
plt.plot(S, np.imag(rho_array), color='blue', label='imag')
plt.legend(loc='upper right')
plt.show()


u_reconst = multi_soliton_scat_data(z_0_im, np.imag(C_list), X_cheb, 0)


plt.plot(X_cheb, Q_cheb , color='green', label='Potentiel initial')
plt.plot(X_cheb, u_reconst, color='red', label='Calc', linestyle='--')
plt.xlim(-30, 30)
plt.legend()
plt.show()

# Sauvgarde

np.savez("Scattering_data_trois_soliton_A_" + str(A) + "_Delt_" + str(delta) + "N_cheb_"+str(N_cheb)+"_N_rho_"
         +str(rho_points)+"_" +str(int_rho)+ ".npz" ,
         X_Fourrier=X_Fourrier, X_cheb=X_cheb, L=L, rho_points=rho_points,
         int_rho=int_rho,  Q_cheb=Q_cheb, rho_array=rho_array, z_0_im=z_0_im, C_list=C_list )