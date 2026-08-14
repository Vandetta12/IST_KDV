# Import
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns

from Collocation_chebychev import Cheb_point
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
# fonctions


def fined_close_to_Fourier_in_cheb(j0,X_cheb,X_fourier):
    """
    Fait exactemetn ce que son nom indique
    :param j0: est un entier car c'est un indice
    :param X_cheb: est la grille de point de chebycheb
    :param X_fourier: est la grille de point pour Fourrier
    :return: l'indice pour la grille de chebychev de l'élément cherché
    """
    dist = []
    for jj in range(len(X_cheb)):
        dist.append(np.abs(X_cheb[jj] - X_fourier[j0]))

    indice = np.argmin(dist)
    return indice



def conv_amp_defa_to_pole_norm(A,delta):
    z_j = np.sqrt(A / 2)
    c_j = 2 * z_j * np.exp(2 * z_j * delta)
    return z_j, c_j


def conv_pole_norm_to_amp_defa(z_j, c_j):
    A = 2 * (z_j ** 2)
    delta = -np.log((2 * z_j) / c_j) * ((2 * z_j) ** (-1))
    return A, delta

def alpha_ij(z_i, z_j):

    quotient = np.abs((z_i - z_j) / (z_i + z_j))
    alpha = 2 * np.log(quotient)

    return alpha

def eta_j(z_j, c_j, x, t):
    eta = -2 * z_j * x + 8 * (z_j ** 3) * t + np.log(c_j/(2 * z_j))
    return eta

def S_mu(mu, z_list, c_list, x, t):

    N = len(z_list)
    Premier_terme = 0
    for jj in range(N):
        Premier_terme += mu[jj] * eta_j(z_list[jj], c_list[jj], x, t)

    Second_terme = 0
    for ii in range(N):
        for jj in np.arange(ii + 1, N):
            Second_terme += mu[ii] * mu[jj] * alpha_ij(z_list[ii], z_list[jj])

    S = Premier_terme + Second_terme

    return S

def gener_mu(N) :
    mu_pos = [[0],[1]]
    for nn in range(N - 1):
        for kk in range(len(mu_pos)):
            cop = mu_pos[kk].copy()
            mu_pos[kk].append(0)
            cop.append(1)
            mu_pos.append(cop)
    return mu_pos



def S_pos(mu_pos, z_list, c_list, x, t):

    S_list = []
    for mu in mu_pos:

        S = S_mu(mu, z_list, c_list, x, t)

        S_list.append(S)

    S_max = max(S_list)


    return S_list, S_max

def p_list(mu_pos, z_list, c_list, x, t):

    S_list, S_max = S_pos(mu_pos, z_list, c_list, x, t)

    P = []
    tau_tild = 0
    for S in S_list:
        tau_tild += np.exp(S - S_max)
        P.append(np.exp(S - S_max))
    P_array = np.array(P)

    return P_array / tau_tild

def Y_mu(mu, z_list):
    sum = 0
    for ii in range(len(z_list)):
        sum += mu[ii] * z_list[ii]

    Y = -2 * sum
    return Y

def multi_soliton_scat_data(z_list , c_list, X, t):
    N = len(z_list)
    mu_pos = gener_mu(N)

    u = np.zeros(len(X), dtype=complex)
    for ii in range(len(X)):
        x = X[ii]
        P_array = p_list(mu_pos, z_list, c_list, x, t)
        terme_1 = 0
        terme_2 = 0
        for jj in range(len(mu_pos)):
            mu = mu_pos[jj]
            p = P_array[jj]
            Y = Y_mu(mu, z_list)
            terme_1 += p * (Y ** 2)
            terme_2 += p * Y

        u[ii] = 2 * (terme_1 - (terme_2 ** 2))

    return u


def multi_soliton_phys(A_list, delta_list, X, t):
    N = len(A_list)
    z_list = np.zeros(N)
    c_list = np.zeros(N)
    for kk in range(N):
        z_list[kk], c_list[kk] = conv_amp_defa_to_pole_norm(A_list[kk], delta_list[kk])
    u = multi_soliton_scat_data(z_list , c_list, X, t)
    return u


def try_pole(A, delta,  A_cal, delta_cal):
    N = len(A)
    A_cal_tri = np.zeros(N)
    delta_cal_tri = np.zeros(N)
    for ii in range(N):
        dist = []
        for jj in range(len(A_cal)):
            a_cal = A_cal[jj]
            d_cal = delta_cal[jj]
            dist.append(np.abs(A[ii] - a_cal) + np.abs(delta[ii] - d_cal))
        indice = np.argmin(dist)
        A_cal_tri[ii] = A_cal[indice]
        delta_cal_tri[ii] = delta_cal[indice]
        A_cal = np.delete(A_cal, indice)
        delta_cal = np.delete(delta_cal, indice)

    return A_cal_tri , delta_cal_tri



def Theta(z, x, t):
    """
    Fonction qui calcule la phase dans les exponentielle du RHP
    :param z: position dans le plan complexe
    :param x: point physique spaciale x sur la droite réelle ou on veux évaluer le potentielle
    :param t: point physique temporelle t sur la droite réelle ou on veux évaluer le potentielle
    :return: theta(z,x,t)
    """
    theta = 2 * 1j * z * x + 8 * 1j * (z ** 3) * t
    return theta

def delta(i,j):
    if i == j:
        sortie = 1
    if i != j:
        sortie =0
    return sortie