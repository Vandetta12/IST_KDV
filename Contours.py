# import
import numpy as np
import matplotlib.pyplot as plt
from Collocation_chebychev import Cheb_point, cheb_int
from Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from Fonction_utile import (fined_close_to_Fourier_in_cheb, conv_pole_norm_to_amp_defa, multi_soliton_scat_data,
                            multi_soliton_phys, try_pole, conv_amp_defa_to_pole_norm,  Theta )


# Fonctions





def Saut_cercle_T(c_j,z,z_j, signe, x, t):
    T = np.zeros([2,2], dtype=complex)

    arg = (-c_j * np.exp(Theta(z_j,x,t))) / (z - (signe * z_j))
    T[0,0] = 1
    T[1,1] = 1
    if signe == -1:
        T[0,1] = arg
    if signe == 1:
        T[1,0] = arg
    return T

def Saut_cercle_S(c_j,z,z_j, signe, x, t):
    S = np.zeros([2, 2], dtype=complex)

    arg = -((z - (signe * z_j)) /c_j) * np.exp(-Theta(z_j, x, t))
    S[0, 0] = 1
    S[1, 1] = 1
    if signe == 1:
        S[0, 1] = arg
    if signe == -1:
        S[1, 0] = arg
    return S



def Dico_courbe(n, Mob, ori):
    """
    Crer un dixionaire de toutes les donnée utile associée à un contour orianté dans le plan complex
    :param n: degré d'interpolation de chebychev
    :param Mob: 4-uple parmètre de la transfor de mobius
    :return: dixionaire
    """
    X_cheb, V = Cheb_point(1, n, oriant = ori, bord=0, compl=True)
    F = np.linalg.inv(V)

    X_phys = Inv_Mobius_general(X_cheb, Mob[0], Mob[1], Mob[2], Mob[3])

    z_g = Inv_Mobius_general(X_cheb[0], Mob[0], Mob[1], Mob[2], Mob[3])
    z_d = Inv_Mobius_general(X_cheb[-1], Mob[0], Mob[1], Mob[2], Mob[3])


    dico = {"Mob": Mob, "n": n, "X_cheb": X_cheb, "V": V, "F": F, "x_phys": X_phys, "z_g": z_g, "z_d": z_d,}

    return dico

def Critere_inv(r,z_j, c_j, x, t, seuille):
    """
    Fonctions qui détermine si un un coutour circulaire associé à un pole doit avoir un suat inverser pour garentire
    la stabilitée numérique.
    :param r: rayon du cercle
    :param z_j: positions du pôle
    :param c_j: constance de normalisation associé au pole
    :param x: positions spaciale (paremetre du RHP)
    :param t: positions temporelle (paremetre du RHP)
    :param seuille: seuille pour garentire la précisions numérique
    :return:
    """
    eta = np.log(np.abs(c_j)) + np.real(Theta(np.abs(z_j) * 1j, x, t))-np.log(r)
    if eta < seuille:
        sortie = False
    else:
        sortie = True
    return sortie

def Q_mat(z,k_ens):
    Q = np.zeros([2,2], dtype=complex)
    Q[0,0] = 1
    Q[1,1] = 1
    for z_j in k_ens:
        Q[0,0] = Q[0,0] * ((z - (np.abs(z_j) * 1j)) / (z + (np.abs(z_j) * 1j)))
        Q[1,1] = Q[1,1] * ((z + (np.abs(z_j) * 1j)) / (z - (np.abs(z_j) * 1j)))
    return Q

def Q_mat_inv(z,k_ens):
    Q_moin = np.zeros([2,2], dtype=complex)
    Q_moin[0,0] = 1
    Q_moin[1,1] = 1
    for z_j in k_ens:
        Q_moin[0,0] = Q_moin[0,0] * ((z + (np.abs(z_j) * 1j)) / (z - (np.abs(z_j) * 1j)))
        Q_moin[1,1] = Q_moin[1,1] * ((z - (np.abs(z_j) * 1j)) / (z + (np.abs(z_j) * 1j)))
    return Q_moin


def K_ensebmle(r, z_j_list, c_j_list, x, t, seuille = 0):
    """

    :param r:
    :param z_j_list:
    :param c_j_list:
    :param x:
    :param t:
    :return:
    """
    k_list = []
    indice_inv_list = []
    for ii in range(len(z_j_list)):
        z_j = z_j_list[ii]
        c_j = c_j_list[ii]
        crit = Critere_inv(r, z_j, c_j, x, t, seuille)
        if crit == True:
            k_list.append(z_j)
            indice_inv_list.append(ii)
    return k_list, indice_inv_list

def Add_saut_cercle_to_dico(dico, c_j, z_j, k_ens, x, t):
    n = dico['n']
    x_phys = dico['x_phys']
    signe = np.sign(z_j)
    G = np.zeros([n, 2, 2], dtype=complex)
    W = np.zeros([n, 2, 2], dtype=complex)
    W_x = np.zeros([n, 2, 2], dtype=complex)
    abs_z_j = np.abs(z_j)
    if abs_z_j in k_ens:
        for ii in range(n):
            Q = Q_mat(x_phys[ii], k_ens)
            Q_inv = Q_mat_inv(x_phys[ii], k_ens)
            G[ii, :, :] = Q_inv @ Saut_cercle_S(c_j, x_phys[ii], abs_z_j * 1j, signe, x, t) @ Q
            W[ii, :, :] = G[ii, :, :] - np.eye(2)
            W_x[ii, :, :] = W[ii,:,:] * 2 * abs_z_j
            inversion = True
    else :
        for ii in range(n):
            Q = Q_mat(x_phys[ii], k_ens)
            Q_inv = Q_mat_inv(x_phys[ii], k_ens)
            G[ii, :, :] = Q_inv @ Saut_cercle_T(c_j, x_phys[ii], abs_z_j * 1j, signe, x, t) @ Q
            W[ii, :, :] = G[ii, :, :] - np.eye(2)
            W_x[ii, :, :] = W[ii,:,:] * -2 * abs_z_j
            inversion = False

    dico['G'] = G
    dico['W'] = W
    dico['signe']= signe
    dico['z_j'] = abs_z_j * 1j
    dico['c_j'] = c_j
    dico['W_x']=W_x
    dico['inversion'] = inversion
    return

def Rayon_cercles(z_j_im):
    dist = []
    N = len(z_j_im)
    if N == 1:
        sortie = np.abs(z_j_im[0]) * 0.8
    if N > 1:
        for ii in range(N):
            for jj in range(ii + 1, len(z_j_im)):
                dist.append(np.abs(z_j_im[ii] - z_j_im[jj]))
        dist_min = (min(dist) / 2)
        norm_min = np.min(np.abs(z_j_im))
        if dist_min < norm_min:
            sortie = dist_min * 0.8
        else:
            sortie = norm_min * 0.8
    return sortie

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

def X_phys_glob_G_glob_W_glob_x_glob(list_dico, N_interpol, N_interpol_courb):
    X_glob = np.zeros(N_interpol, dtype=complex)
    G_glob = np.zeros([N_interpol, 2, 2], dtype=complex)
    W_glob = np.zeros([N_interpol, 2, 2], dtype=complex)
    W_x_glob = np.zeros([N_interpol, 2, 2], dtype=complex)
    for ii in range(len(list_dico)):
        ii_indice = (ii * N_interpol_courb, (ii + 1) * N_interpol_courb)
        X_glob[ii_indice[0] : ii_indice[1]] = list_dico[ii]['x_phys']
        G_glob[ii_indice[0]:ii_indice[1], :, :] = list_dico[ii]['G']
        W_glob[ii_indice[0]:ii_indice[1], :, :] = list_dico[ii]['W']
        W_x_glob[ii_indice[0]:ii_indice[1], :, :] = list_dico[ii]['W_x']
    return X_glob, G_glob, W_glob, W_x_glob

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

def Int_cheb_un_contour(dico,U_x_j, comp):
    X_cheb = dico['X_cheb']
    F = dico['F']
    n = dico['n']
    Mob = dico['Mob']
    A_inv, B_inv, C_inv, D_inv = Inv_Mobius_coef(Mob[0], Mob[1], Mob[2], Mob[3])
    fonc = np.zeros(n, dtype=complex)
    for ii in range(n):
        fonc[ii] = U_x_j[ii, comp] * Mobius_general_der(A_inv, B_inv, C_inv, D_inv, X_cheb[ii])
    int_j = cheb_int(fonc, F, 1, 1)
    return int_j

def Int_cheb_mult_contour(list_dico, U_x_phys, comp):
    integral = 0
    U_x_list = Decupage_U(U_x_phys, list_dico)
    compteur = 0
    for dico in list_dico:
        integral += Int_cheb_un_contour(dico, U_x_list[compteur], comp)
        compteur += 1
    return -integral * (1/(2*1j*np.pi))

