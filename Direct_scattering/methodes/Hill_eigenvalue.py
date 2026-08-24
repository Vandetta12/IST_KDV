# Imports
import numpy as np


# Fonctions
def a(Y):
    """
    Premier coeficient de l'équation
    :param Y: array entre -pi et pi
    :return: un array d'image de l'array Y par la fonction a
    """
    a_list = -np.cos(Y/2) ** 4
    return a_list

def b(Y):
    """
    Second coeficient de l'équation
    :param Y: array entre -pi et pi
    :return: un array d'image de l'array Y par la fonction b
    """
    b_list = (np.cos(Y/2) ** 3) * np.sin(Y/2)
    return b_list

def Changement_variables(Y):
    """
    Changement de variable pour retourner a la droite réelle
    :param Y: array entre -pi et pi
    :return: un array
    """
    X = 2 * np.tan(Y / 2)
    return X



def grille(N ,M ):
    """
    Crée une grille de points sur [-pi, pi]
    :param N: Nombre de modes de Fourrier avent troncage
    :param M: Nombre de point sur la grille pour la FFT
    :return: n et Y deux array
    """
    n = np.arange(-N, N + 1)                            # indices des modes
    Y = 2 * np.pi * (np.arange(M) + 0.5)/M - np.pi     # Grille de points
    return n, Y



def fourier_coeffs_shifted(f_vals, M):
    """
    Calcule hat f_k pour une grille midpoint :
        Y_j = -pi + 2pi (j+1/2)/M

    Convention : f(y) ~ sum_k hat f_k e^{i k y}
    """
    # 1) FFT brute (sur l'index j)
    F = np.fft.fft(f_vals) / M

    # 2) Les modes k associés à la FFT (entiers positifs puis négatifs)
    k = np.fft.fftfreq(M, d=1.0/M)   # ex: [0,1,2,...,-2,-1]
    # k est float mais ce sont des entiers représentés en float
    # (on les convertira après le shift)

    # 3) Correction due au fait que Y_j = -pi + 2pi(j+1/2)/M
    # e^{-ikY_j} = e^{-ik(-pi)} e^{-ik*2pi(j+1/2)/M}
    # => facteur (-1)^k * exp(-i*pi*k/M) devant la FFT.
    phase = np.exp(1j*np.pi*k) * np.exp(-1j*np.pi*k/M)
    F = F * phase

    # 4) Remettre en ordre symétrique
    F = np.fft.fftshift(F)
    ksym = np.fft.fftshift(k).astype(int)

    return ksym, F


def Fourier_troncage(ksym, F_chapeau, N):
    restriction = np.arange(-2*N, 2*N + 1)
    dico = {int(k): i for i, k in enumerate(ksym)}
    return restriction, np.array([F_chapeau[dico[int(k)]] for k in restriction], dtype=complex)

def Matrice_Hill(N, a_chapeau, b_chapeau, Q_chapeau):
    """
    Construit la matrice H
    :param N: parmètre du nombres de modes
    :param a_chapeau: coefficiens de la fonctio a (array)
    :param b_chapeau: coefficiens de la fonctio b (array)
    :param Q_chapeau: coefficiens de la fonctio Q (array)
    :return: ùatrice H
    """
    N_vect = np.arange(N,3*N + 1,1)
    H = np.zeros((2*N + 1, 2*N + 1), dtype=complex)
    for i in N_vect:
        for j in N_vect:
            H[i - N, j - N] = -((j - 2*N) ** 2) * a_chapeau[i - j + (2 * N)] + (1j * (j - 2*N)) * b_chapeau[i - j + (2 * N)] - Q_chapeau[i - j + (2 * N)]
    return H

def Diag_et_filtre(H):
    # Diagonalisation
    eigvals, eigvecs = np.linalg.eig(H)
    # filtrage
    mask = (np.abs(eigvals.imag) < 1e-10) & (eigvals.real < -1e-10)
    lambda_bonne = eigvals.real[mask]
    eigvecs_bonne = eigvecs[:, mask]

    return lambda_bonne, eigvecs_bonne

def eta_lier(vecteur_propres, Y, N):
    """
    reconstruit les états lier à partir des vecteurs propre de l'opérateu réduit sur l'intervalle -pi pi
    :param vecteur_propres:
    :param Y:
    :return:
    """
    E_lier = list(np.zeros(len(vecteur_propres[0,:])))

    j0 = []

    for jj in range(len(vecteur_propres[0,:])):
        m_j = np.zeros_like(Y, dtype=complex)

        for n in range(-N, N + 1):
            m_j += vecteur_propres[n + N, jj] * np.exp(1j * n * Y)

        # Normalisation
        #norm = np.trapezoid(np.abs(m_j) ** 2 * weight, Y)
        #m_j /= np.sqrt(norm)

        E_lier[jj]=m_j
        j0.append(np.argmax(np.abs(m_j)))


    return E_lier, j0




# Corps du code
# Paramètres
#N = 500                                   # Nombre de modes de Fourrier avent troncage
#M = (2 * N + 1) * 10                      # Nombre de point sur la grille pour la FFT
#n, Y = grille(N, M)
#X = Changement_variables(Y)

#print("Y_min, Y_max:", Y.min(), Y.max())
#print("X_min, X_max:", X.min(), X.max())



# Potentiel
#v1 = 2.4 * 2
#v2 = 1
#deta = 10
# Soliton à l'instant t = 0
#Q = (v1 / 2) * (np.cosh((np.sqrt(v1) / 2) * (X ))) ** (-2) + (v2 / 2) * (np.cosh((np.sqrt(v2) / 2) * (X - deta))) ** (-2)

# idmen, mais numériquement plus stable
#z = np.sqrt(v1)/2 * X

#t1 = np.exp(-2*np.abs(z))

#Q = (v1/2) * (4*t1) / (1 + t1)**2


# Presque soliton a t=0
#Q = (v / 2) * (np.cosh((X - deta))) ** (-2)

# Evaluation des fonctions et des TF avec FFT

#eps = 1e-7

#a_val = a(Y)
#b_val = b(Y)
#Q_val = Q.copy()
#k_a, a_chapeau = fourier_coeffs_shifted(a_val, M)
#k_b, b_chapeau = fourier_coeffs_shifted(b_val, M)
#k_q, Q_chapeau = fourier_coeffs_shifted(Q_val, M)

# Tronquer
#k_restraint, a_chapeau = Fourier_troncage(k_a, a_chapeau, N)
#_, b_chapeau = Fourier_troncage(k_b, b_chapeau, N)
#_, Q_chapeau = Fourier_troncage(k_q, Q_chapeau, N)

#H = Matrice_Hill(N, a_chapeau, b_chapeau, Q_chapeau)
#valeurs_propres, vecteurs_propres = Diag_et_filtre(H)         # Uniquement les états lier
#eta_lier_vecteur = eta_lier(vecteurs_propres, Y)

#print("valeurs_propres:", valeurs_propres)
#print("vecteurs_propres:", vecteurs_propres)

#z_0_im = np.sqrt(-valeurs_propres)
#print("Poles : ",z_0_im)
