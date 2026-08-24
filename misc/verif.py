# Imports
import numpy as np
from scipy.special import gamma


# Fonctions
def coef_a_tild(A,z):
    a_tild = 1/2 - 1j * z + ((A + (1/4)) ** (1/2))
    return a_tild

def coef_b_tild(A,z):
    b_tild = 1/2 - 1j * z - ((A + (1/4)) ** (1/2))
    return b_tild

def coef_c_tild(A,z):
    c_tild = 1 - 1j * z
    return c_tild

def coef_a(A,z):
    num = gamma(coef_a_tild(A,z)) * gamma(coef_b_tild(A,z))
    den = gamma(coef_c_tild(A,z)) * gamma(coef_a_tild(A,z) + coef_b_tild(A,z) - coef_c_tild(A,z))
    return num / den

def rho_verif(A,z):
    """
    Fonction calculant la fonction rho pour un potentielle du type A+cosh ** 2 de manière analytique
    :param A: float
    :param z: compplex
    :return: complex
    """
    num= coef_a(A,z) * gamma(coef_c_tild(A,z)) * gamma(coef_c_tild(A,z) - coef_a_tild(A,z) - coef_b_tild(A,z))
    den = gamma(coef_c_tild(A,z) - coef_a_tild(A,z)) * gamma(coef_c_tild(A,z) - coef_b_tild(A,z))
    return num / den

def verif_cond(A,j):
    return ((A + (1/4)) ** (1/2)) - (j + (1/2))

def Pole_verif(A):
    """
    Fonction calculant la fonction lespôle pour un potentielle du type A+cosh ** 2 de manière analytique
    :param A: float
    :param z: compplex
    :return: complex
    """
    pole = []
    j = 0
    while verif_cond(A,j) > 0:
        print("j, pole", j,  verif_cond(A,j))
        pole.append(1j * verif_cond(A, j))
        j+=1
    return pole