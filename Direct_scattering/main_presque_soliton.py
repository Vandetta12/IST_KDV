# Imports
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from methodes.Collocation_chebychev import Cheb_point, Diff_cheb_1, residus, rho
from methodes.Hill_eigenvalue import a, b , Changement_variables, grille, fourier_coeffs_shifted, Fourier_troncage, Matrice_Hill, Diag_et_filtre, eta_lier
from misc.verif import rho_verif, Pole_verif
from misc.Fonction_utile import fined_close_to_Fourier_in_cheb

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
L = 10                                                  # troncage de la droite pour chebychev
N_cheb = 201                                            # Nombre de points de collocation
rho_points = 200
int_rho = [-5, 5]
X_cheb, V = Cheb_point(L, N_cheb , oriant = 1, bord = 0, compl = False)
D = Diff_cheb_1(N_cheb)
print("centre :", X_cheb[N_cheb//2])
#######################################################################################################"

# définition du potentiel
# --------------------------
v1 = 2.4 * 2
# Soliton à l'instant t = 0
Q_cheb = -(v1 / 2) * (np.cosh(X_cheb) ** (-2))
Q_Fourrier = (v1 / 2) * (np.cosh(X_Fourrier) ** (-2))

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
#print("vecteurs_propres:", vecteurs_propres)

z_0_im = np.sqrt(-valeurs_propres)
print("Poles : ",z_0_im)

# Calcule des résidus de a ( les coefficient de normalisation)

b_z_0 = []
C_list= []
for jj in range(len(z_0_im)):
    j0_cheb = fined_close_to_Fourier_in_cheb(j0[jj], X_cheb, X_Fourrier)
    b_pole, Integ_mu = residus(D_1, D_2, U, Q_cheb, (z_0_im[jj] * 1j), X_cheb, j0_cheb, V_inv, L)
    print("Integ_mu:", Integ_mu)
    b_z_0.append(b_pole)
    C_list.append((1j * (b_pole ** 2)) / Integ_mu)
print("b_z_0:", b_z_0)
print("c_z_0:", C_list)


# verification
A = v1 / 2
Pol_verif_array = Pole_verif(A)
rho_verif_array = rho_verif(A, S)
for jj in range(min(len(Pol_verif_array), len(z_0_im))):
    print("Pol_verif_array:", Pol_verif_array[jj])
    print("Pole", z_0_im[jj])
    print("diff:", np.abs(np.imag(Pol_verif_array[jj]) - z_0_im[jj]))




# visualisation
plt.plot(S, np.real(rho_array), color='red', label='real')
plt.plot(S, np.imag(rho_array), color='blue', label='imag')
plt.plot(S, np.real(rho_verif_array), color='red', label='real',linestyle=':')
plt.plot(S, np.imag(rho_verif_array), color='blue', label='imag',linestyle=':')
plt.legend(loc='upper right')
plt.show()

Error = rho_array - rho_verif_array
print("Max rho error real :", max(np.abs(np.real(Error))))
print("Max rho error im :", max(np.abs(np.imag(Error))))

plt.plot(S, np.real(Error), color='red', label='Err real')
plt.show()
plt.plot(S, np.imag(Error), color='blue', label='Err imag')
plt.show()

plt.plot(X_Fourrier, Q_Fourrier, color='green', label='Th')
#plt.plot(X_Fourrier, Q_Fourrier_verif, color='red', label='Calc', linestyle='--')
plt.xlim(-30, 30)
plt.legend()
plt.show()

