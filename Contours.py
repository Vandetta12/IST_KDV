# import
import numpy as np
import matplotlib.pyplot as plt

from Cauchy_transform import Psi_matrice
from Collocation_chebychev import Cheb_point
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa, multi_soliton_scat_data,
                            multi_soliton_phys, try_pole, conv_amp_defa_to_pole_norm,  Theta )


# Fonctions





def Saut_cercle(c_j,z,z_j, signe, x, t):
    V = np.zeros([2,2], dtype=complex)
    arg = (-c_j * np.exp(Theta(z_j,x,t))) / (z - (signe * z_j))
    V[0,0] = 1
    V[1,1] = 1
    if signe == 1:
        V[0,1] = arg
    if signe == -1:
        V[1,0] = arg
    return V

def Dico_courbe(n, Mob, ori):
    """
    Crer un dixionaire de toutes les donnée utile associée à un contour orianté dans le plan complex
    :param n: degré d'interpolation de chebychev
    :param Mob: 4-uple parmètre de la transfor de mobius
    :return: dixionaire
    """
    X_cheb, V = Cheb_point(1, n, oriant = ori, bord=0)
    F = np.linalg.inv(V)

    X_phys = Inv_Mobius_general(X_cheb, Mob[0], Mob[1], Mob[2], Mob[3])

    z_g = Inv_Mobius_general(X_cheb[0], Mob[0], Mob[1], Mob[2], Mob[3])
    z_d = Inv_Mobius_general(X_cheb[-1], Mob[0], Mob[1], Mob[2], Mob[3])


    dico = {"Mob": Mob, "n": n, "X_cheb": X_cheb, "V": V, "F": F, "x_phys": X_phys, "z_g": z_g, "z_d": z_d,}

    return dico

def Add_saut_cercle_to_dico(dico, c_j, z_j, x, t):
    n = dico['n']
    x_phys = dico['x_phys']
    signe = np.sign(z_j)
    G = np.zeros([n, 2, 2], dtype=complex)
    W = np.zeros([n, 2, 2], dtype=complex)
    for ii in range(n):
        G[ii, :, :] = Saut_cercle(c_j, x_phys[ii], np.abs(z_j) * 1j, signe, x, t)
        W[ii, :, :] = G[ii, :, :] - np.eye(2)
    dico['G'] = G
    dico['W'] = G[:]
    dico['signe']= signe
    dico['z_j'] = np.abs(z_j) * 1j
    dico['c_j'] = c_j
    return

def Rayon_cercles(z_j_im):
    dist = []
    for ii in range(len(z_j_im)):
        for jj in range(ii + 1, len(z_j_im)):
            dist.append(np.abs(z_j_im[ii] - z_j_im[jj]))
    return (min(dist) / 2) * 0.8

def visualisation_contour_cercle(dico_liste, z_list):
    taille = 10
    for ii in range(len(dico_liste)):
        plt.plot(np.real(dico_liste[ii]['x_phys']), np.imag(dico_liste[ii]['x_phys']), color='black')
        plt.scatter(np.real(dico_liste[ii]['z_g']), np.imag(dico_liste[ii]['z_g']), s=taille, marker='o', facecolor='none',
                    edgecolor='red', linewidth=1.5, zorder=5)
        plt.scatter(np.real(dico_liste[ii]['z_d']), np.imag(dico_liste[ii]['z_d']),  s=taille * 0.8, marker='^',
                    color='blue', zorder=4)
        plt.scatter(np.real(z_list[ii // 4] * 1j), np.imag(z_list[ii // 4] * 1j), s=taille,  color='green')
        plt.scatter(np.real(-z_list[ii // 4] * 1j), np.imag(-z_list[ii // 4] * 1j), s=taille, color='green')
    plt.xlim(-1, 1)
    plt.ylim(-3, 3)
    plt.axis("equal")
    plt.grid()
    plt.show()

def X_phys_glob_G_glob_W_glob(list_dico, N_interpol, N_interpol_courb):
    X_glob = np.zeros(N_interpol, dtype=complex)
    G_glob = np.zeros([N_interpol, 2, 2], dtype=complex)
    W_glob = np.zeros([N_interpol, 2, 2], dtype=complex)
    for ii in range(len(list_dico)):
        ii_indice = (ii * N_interpol_courb, (ii + 1) * N_interpol_courb)
        X_glob[ii_indice[0] : ii_indice[1]] = list_dico[ii]['x_phys']
        G_glob[ii_indice[0]:ii_indice[1], :, :] = list_dico[ii]['G']
        W_glob[ii_indice[0]:ii_indice[1], :, :] = list_dico[ii]['W']
    return X_glob, G_glob, W_glob

def Decupage_U(u_phys, liste_dico):
    """
    Découpe U par contoure
    :param liste_dico:
    :return: une lise de array
    """
    liste_u = []
    conteur = 0
    for dico in liste_dico:
        n = dico['n']
        liste_u.append(u_phys[n * conteur : n * (conteur + 1), :])
        conteur += 1
    return liste_u

def Evalulation_Cauchy_une_courbe(z, u_phys, dico):
    """
    Evalue la valeur de la transfo de chauchy au point z sur un seul contour
    :param z: point d'éval
    :param u_phys: valeur de la fonction sur la courbe
    :param dico: dictionaire associé à la curbe
    :return: valeur de la TF de cauchy de la fonction au point z
    """
    Mob = dico['Mob']
    F = dico['F']
    n = dico['n']
    A, B, C, D = Mob
    z_prim = Mobius_general(A, B, C, D, z)              # on place z dans les coordonnée de la courbe
    z_prim_cercle = Inv_Jacouwski_plus(z_prim)          # Transfo de jacouwski pour arriver sur le cercle
    C = Psi_matrice(z_prim_cercle, n) @ F

    # correction de déformation
    infinit = Mobius_inf(A, C)
    if not np.isinf(infinit):
        z_inf = Inv_Jacouwski_plus(infinit)
        C = C - Psi_matrice(z_inf, n) @ F

    return C @ u_phys[:,0], C @ u_phys[:,1]

def Evalulation_Cauchy(z, u_phys, liste_dico):
    """
    Evalue la valeur de la transfo de chauchy au point z sur tt les contours
    :param z: point d'éval
    :param u_phys: valeur de la fonction sur la courbe
    :param dico: dictionaire associé à la curbe
    :return: valeur de la TF de cauchy de la fonction au point z
    """
    U_lit = Decupage_U(u_phys, liste_dico)
    Cu1 = 0
    Cu2 = 0
    compteur = 0
    for dico in liste_dico:
        Cu1_compteur, Cu2_compteur = Evalulation_Cauchy_une_courbe(z, U_lit[compteur], dico)
        Cu1 += Cu1_compteur
        Cu2 += Cu2_compteur
        compteur += 1
    return Cu1, Cu2

def Evaluation_cauchy_grid(z_grid, u_phys, liste_dico):
    """
    évalue la TF de cauchy sur une grille de points complexe pour un ensemble de courbes
    :param z_grid: grille de points complexe
    :param u_phys:valeur de la fonction sur les courbes
    :param liste_dico:dictionnaire contenant les info sur les courbes
    :return:
    """
    (n_ii, n_jj) = z_grid.shape()
    Cu1_grid = np.zeros([n_ii,n_jj], dtype=complex)
    Cu2_grid = np.zeros([n_ii,n_jj], dtype=complex)
    for ii in range(n_ii):
        for jj in range(n_jj):
            Cu1_grid[ii, jj], Cu2_grid[ii, jj] = Evalulation_Cauchy(z_grid[ii, jj], u_phys, liste_dico)
    return Cu1_grid, Cu2_grid