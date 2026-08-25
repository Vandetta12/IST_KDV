# Import
from ftplib import error_perm
import numpy as np
import matplotlib.pyplot as plt
from Direct_scattering.methodes.Collocation_chebychev import Cheb_point

# Fonctions

def Mobius_general(a, b, c, d, z):
    M = (a * z + b) / (c * z + d)
    return M

def Inv_Mobius_general(x,  a, b, c, d):
    z = (x * d - b) / (a - c * x )
    return z

def Inv_Mobius_coef(A, B, C, D):
    a = D
    b = -B
    c = -C
    d = A
    return a, b, c, d

def Mobius_general_der(a, b, c, d, z):
    M_prime = (a * d - b * c) / ((c *z + d ) ** 2)
    return M_prime

def Mobius_inf(a, c):
    if c != 0 :
        m_inf = a / c
    if c == 0:
        m_inf = np.inf
    return m_inf

def Mobius_compositions(Mob2, Mob1):
    """
    Calcule la compostion de deux transformations Mobius M2 rond M1
    :param Mob1: 4-uple des coef de la première transformation
    :param Mob2: 4-uple des coef de la première transformation
    :return:  4 coef de la compostion
    """
    A1, B1, C1, D1 = Mob1
    A2, B2, C2, D2 = Mob2

    A = A2 * A1 + B2 * C1
    B = A2 * B1 + B2 * D1
    C = C2 * A1 + D2 * C1
    D = C2 * B1 + D2 * D1

    return A, B, C, D

def Mobius_interval_ab(a, b):
    A = - 2
    B = a + b
    C = 0
    D = a - b
    return A , B, C, D

def Mobius_strech_ray(a, theta, L):
    q = L * np.exp(1j * theta)
    A = - 1
    B = a + q
    C = -1
    D = a - q
    return A, B, C, D

def Mobius_rivers_strech_ray(a, theta, L):
    q = L * np.exp(1j * theta)
    A = 1
    B = -(a + q)
    C = -1
    D = a - q
    return A, B, C, D

def Mobius_arc(a, r, theta1, theta2):
    z_g = a + r * np.exp(1j * theta1)
    z_m = a + r * np.exp(1j * (theta1 + theta2) / 2)
    z_d = a + r * np.exp(1j * theta2)

    A = 1
    B = - z_m
    C = (z_d + z_g - 2 * z_m) / (z_d - z_g)
    D = z_m - z_g - C * z_g
    return A, B, C, D


# transformation de Jacouwski
def Jacouwski(z):
    z = np.asarray(z, dtype=complex)
    return (1/2) * (z + (1 / z))

def Inv_Jacouwski_plus(x):
    x = np.asarray(x, dtype=complex)
    return x - np.sqrt(x-1) * np.sqrt(1+x)

def Inv_Jacouwski_moin(x):
    x = np.asarray(x, dtype=complex)
    return x + np.sqrt(x-1) * np.sqrt(1+x)

def Inv_Jacouwski_bas(x):
    x = np.asarray(x, dtype=complex)
    return x - 1j * np.sqrt(1 - x) * np.sqrt(1+x)

def Inv_Jacouwski_haut(x):
    x = np.asarray(x, dtype=complex)
    return x + 1j * np.sqrt(1 - x) * np.sqrt(1+x)

