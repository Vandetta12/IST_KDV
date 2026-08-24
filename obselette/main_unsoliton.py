# Imports
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from Collocation_chebychev import Cheb_point, Diff_cheb_1, reconstruction, rho
from Hill_eigenvalue import a, b , Changement_variables, grille, fourier_coeffs_shifted, Fourier_troncage, Matrice_Hill, Diag_et_filtre, eta_lier, Int
from verif import rho_verif, Pole_verif

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
L = 100                                                  # troncage de la droite pour chebychev
N_cheb = 1000                                              # Nombre de points de collocation
rho_points = 100
X_cheb, V = Cheb_point(L, N_cheb)

D = Diff_cheb_1(N_cheb)

#######################################################################################################"

# définition du potentiel
# --------------------------
v1 = 4
#v2 = 1.5
deta = 20
# Soliton à l'instant t = 0
#Q_cheb = (v1 / 2) * (np.cosh((np.sqrt(v1) / 2) * (X_cheb ))) ** (-2) + (v2 / 2) * (np.cosh((np.sqrt(v2) / 2) * (X_cheb - deta))) ** (-2)
#Q_Fourrier = (v1 / 2) * (np.cosh((np.sqrt(v1) / 2) * (X_Fourrier ))) ** (-2) + (v2 / 2) * (np.cosh((np.sqrt(v2) / 2) * (X_Fourrier - deta))) ** (-2)
#Q_cheb = (v1 / 2) * (np.cosh((X_cheb))) ** (-2)
#Q_Fourrier = (v1 / 2) * (np.cosh((X_Fourrier))) ** (-2)
Q_cheb = (v1 / 2) * (np.cosh((np.sqrt(v1) / 2) * (X_cheb - deta))) ** (-2)
Q_Fourrier = (v1 / 2) * (np.cosh((np.sqrt(v1) / 2) * (X_Fourrier -deta))) ** (-2)
# Calcule de rho sur la droite réelle :
# ------------------------------------

U = np.diag(Q_cheb)
V_inv = np.linalg.solve(V, np.eye(N_cheb + 1))

D_1 = V @ D @ V_inv / L
D_2 = V @ D @ D @ V_inv / (L ** 2)

S = np.linspace(-10,10,rho_points)
rho_list = []
for s in S:
    print(s)
    rho_s, _ = rho(N_cheb, D_1, D_2, U, Q_cheb, -s)
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
eta_lier_vecteur = eta_lier(vecteurs_propres, Y)

print("valeurs_propres:", valeurs_propres)
#print("vecteurs_propres:", vecteurs_propres)

z_0_im = np.sqrt(-valeurs_propres)
print("Poles : ",z_0_im)

# Calcule des résidus de a ( les coefficient de normalisation)

b_z_0 = []
for jj in range(len(z_0_im)):
    _, b_pole = rho(N_cheb, D_1, D_2, U, Q_cheb, (z_0_im[jj] * 1j))
    b_z_0.append(b_pole)
print("b_z_0:", b_z_0)



C_list,_ = Int(Y, z_0_im, eta_lier_vecteur, X_Fourrier, b_z_0, x_queu=100.0, x_cap=60.0)

print("C_list:", C_list)














# verification
A = v1 / 2
#Pol_verif_array = Pole_verif(A)
#rho_verif_array = rho_verif(A, S)
#for jj in range(min(len(Pol_verif_array), len(z_0_im))):
#    print("Pol_verif_array:", Pol_verif_array[jj])
#    print("Pole", z_0_im[jj])
#    print("diff:", np.abs(np.imag(Pol_verif_array[jj]) - z_0_im[jj]))

print("v1 theorique =", v1)
#print("v2 theorique =", v2)
print("delta theorique =", deta)
v1_cal = 4 * (z_0_im[0] ** 2)
#v2_cal = 4 * (z_0_im[1] ** 2)
delta1 = -np.log((2 * z_0_im[0])/np.imag(C_list[0])) * ((2 * z_0_im[0]) ** (-1))
#delta2 = -np.log((2 * z_0_im[1])/np.imag(C_list[1])) * ((2 * z_0_im[1]) ** (-1))
#Q_Fourrier_verif = (v1_cal / 2) * (np.cosh((np.sqrt(v1_cal) / 2) * (X_Fourrier - delta1 ))) ** (-2) + (v2_cal / 2) * (np.cosh((np.sqrt(v2_cal) / 2) * (X_Fourrier - delta2))) ** (-2)
Q_Fourrier_verif = (v1_cal / 2) * (np.cosh((np.sqrt(v1) / 2) * (X_Fourrier - delta1 ))) ** (-2)
print("v1 Calculer =", v1_cal)
#print("v2 calculer =", v2_cal)
print("Diff v1, diff v2 :", np.abs(v1_cal - v1))
print("delta calculer =", delta1)
print("Diff delta:", np.abs(delta1 - deta))


# visualisation
plt.plot(S, np.real(rho_array), color='red', label='real')
plt.plot(S, np.imag(rho_array), color='blue', label='imag')
#plt.plot(S, np.real(rho_verif_array), color='red', label='real',linestyle=':')
#plt.plot(S, np.imag(rho_verif_array), color='blue', label='imag',linestyle=':')
plt.legend(loc='upper right')
plt.show()
print("Max rho error real :", max(np.abs(np.real(rho_array))))
print("Max rho error im :", max(np.abs(np.imag(rho_array))))

plt.plot(X_Fourrier, Q_Fourrier, color='green', label='Th')
plt.plot(X_Fourrier, Q_Fourrier_verif, color='red', label='Calc', linestyle='--')
plt.xlim(-30, 30)
plt.legend()
plt.show()

