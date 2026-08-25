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
    :return:
    """