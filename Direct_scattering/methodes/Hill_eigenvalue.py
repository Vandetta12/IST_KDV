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
    n = np.arange(-N, N + 1)                           # indices des modes
    Y = 2 * np.pi * (np.arange(M) + 0.5)/M - np.pi     # Grille de points
    return n, Y


def fourier_coeffs_shifted(f_vals, M):
    """
    Fonction calculant les coeficient de Fourier centre sur 0 sur un intervalle [-pi, pi] et avec la convention
    de l'article de B. Deconinck et J. Nathan Kutz : Computing spectra of linear operators using the
    Floquet–Fourier–Hill method
    :param f_vals:Array de valeur de la fonction sur un array de positions
    :param M:nombre de points sur la grille
    :return: Retourne les indices de Fourier et les coefficients de Fourier dans la bonne convention
    """
    F = np.fft.fft(f_vals) / M                # FFT dans les convention de numpy normalisé (bon coef, mais pas centrer)
    k = np.fft.fftfreq(M, d=1.0/M)            # On recupere les frequance associer au mode, mais dans la convention 2ipi

    phase = np.exp(1j*np.pi*k)                # Translate de -pi
    phase = phase * np.exp(-1j*np.pi*k/M)     # Transalte d'un demi point de grille de manière a centre le mode 0
    F = F * phase

    F_sym = np.fft.fftshift(F)                # On place les mode du plus négatif au plus positif (numpy met par défaut
                                              # les positif suivit des négatifs
    k_sym = np.fft.fftshift(k).astype(int)

    return k_sym, F_sym



def Fourier_troncage(ksym, F_chapeau, N):
    """
    Tronque le développement en série de fourier apporter par la FFT
    :param ksym: les frequance des modes
    :param F_chapeau: les cofficients de Fourier
    :param N: Taille a tronquer
    :return: les coeff trnquer ainsi que leur indices
    """
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
    """
    On diagonalise la matrice de Hill
    :param H: Matrice de Hill
    :return: les valeurs prorpies et les vecteurs propre
    """
    # Diagonalisation
    eigvals, eigvecs = np.linalg.eig(H)
    # filtrage
    mask = (np.abs(eigvals.imag) < 1e-10) & (eigvals.real < -1e-10)
    lambda_bonne = eigvals.real[mask]
    eigvecs_bonne = eigvecs[:, mask]

    return lambda_bonne, eigvecs_bonne


