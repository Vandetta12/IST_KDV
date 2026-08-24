# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from Direct_scattering.methodes.Collocation_chebychev import Cheb_point
from misc.Mobius_mpas import (Mobius_interval_ab, Mobius_strech_ray,
                         Mobius_rivers_strech_ray, Mobius_arc, Mobius_general, Inv_Mobius_general,
                         Mobius_general_der, Mobius_inf, Jacouwski, Mobius_compositions, Inv_Mobius_coef,
                         Inv_Jacouwski_bas, Inv_Jacouwski_haut, Inv_Jacouwski_moin, Inv_Jacouwski_plus)
from misc.Fonction_utile import Delta
from Contours import Decupage_U



# C+[II,II]
def P_Coch(N_cheb):
    vect = np.ones(N_cheb)
    vect[1:] = vect[1:] * 2
    A = np.diag(vect)
    B = np.zeros([N_cheb, N_cheb], dtype=complex)
    for ii in range(N_cheb):
        moin = N_cheb
        for jj in range(ii + 1, N_cheb):
            moin -= 1
            if moin % 2 != 0:
                B[ii, jj] = 1 / (N_cheb - moin)
    P = A @ B
    return P



def C_plus_unit_unit(X_cheb, V, V_inv):
    N_cheb = len(X_cheb)
    vect = -2 * np.arctanh(Inv_Jacouwski_bas(X_cheb[1:-1]))
    C = np.diag(vect)
    A = np.zeros([N_cheb, N_cheb], dtype=complex)
    A[0, 0] = (np.log(2) / 2) + (1j * np.pi / 2)
    A[ -1, -1] = -(np.log(2) / 2) + (1j * np.pi / 2)
    A[1:-1 , 1:-1] = C
    P = P_Coch(N_cheb)
    c_plus = (1 / (1j * np.pi)) * (A + V @ P @ V_inv )
    return c_plus


# Pol chebychev

def Mu_k(k,X_cheb):
    born = (k + 1) // 2
    mu_k = np.zeros(len(X_cheb), dtype=complex)
    for ii in range(born):
        mu_k = mu_k + (X_cheb ** ((2 * (ii + 1) ) - 1)) / ((2 * (ii + 1) )- 1)
    return mu_k

def Psi_k_alt(k, X_cheb):
    psi_0 = (2 / (1j * np.pi)) * np.arctanh(X_cheb)
    mu_k = 0
    if k < 0:
        mu_k = Mu_k(-k - 1, X_cheb)
    if k > 0:
        mu_k = Mu_k(k, 1/X_cheb)
    if k == 0:
        mu_k = 0
    psi = (X_cheb ** k) * (psi_0 - (2 / (1j * np.pi)) * mu_k )

    return psi

def Psi_k_alt2(k, X_cheb):

    if k < 0:
        facteur = ((X_cheb ** (1 + 2 * np.floor(-k/2) + k))
                   / ((1 + 2 * np.floor(-k/2))))
        a = np.floor(-k / 2)
        arg = X_cheb ** 2
        H = hyp2f1(1, 0.5 + a, 1.5 + a, arg)

        if np.any(np.isnan(H)):
            bad = np.where(~np.isfinite(H))[0]
            print("hyp2f1 nan/inf pour k =", k)
            print("bad indices:", bad[:10])
            print("bad X:", X_cheb[bad[:10]])
            print("bad arg:", arg[bad[:10]])

        psi = facteur * H
    if k > 0:
        terme = (X_cheb ** k) * (np.arctanh(X_cheb) - np.arctanh(1/X_cheb))
        facteur = ((X_cheb ** (k - 1 - 2 * np.floor((k + 1) / 2)))
                   / (1 + 2 * np.floor((k + 1) / 2)))

        psi = terme + facteur * hyp2f1(1, 0.5 + np.floor((k + 1)/2) ,(3/2) + np.floor((k + 1)/2), (X_cheb ** -2))
    if k == 0:
        psi = np.arctanh(X_cheb)
        if np.any(np.isnan(psi)):
            print("attention, psi = nane pour k = 0")
    psi = psi * (2 / (1j * np.pi))

    return psi

def Psi_k(k, Z, tol=1e-14, max_iter=200000):
    """
    Attention générer par IA place older
    Calcule psi_k(Z) sans récurrence.

    Cas k < 0 :
        queue de série stable.

    Cas k = 0 :
        psi_0 = 2/(i*pi) arctanh(Z).

    Cas k > 0 :
        somme finie exacte issue de mu_k(1/Z).

    Hypothèse pratique :
        Pour k < 0, cette série est efficace si |Z| < 1,
        ce qui est le cas attendu pour Z = T_+^{-1}(x).
    """

    Z = np.asarray(Z, dtype=complex)

    # k = 0
    if k == 0:
        return (2 / (1j * np.pi)) * np.arctanh(Z)

    # k > 0 : somme finie
    if k > 0:
        s = Z**k * np.arctanh(Z)

        Jmax = (k + 1) // 2

        for j in range(1, Jmax + 1):
            s -= Z**(k - 2*j + 1) / (2*j - 1)

        return (2 / (1j * np.pi)) * s

    # k < 0 : queue de série stable
    m = -k
    q = m // 2

    exponent = 2*q + 1 - m
    denom = 2*q + 1

    term = Z**exponent / denom
    s = term.copy()

    Z2 = Z**2

    for _ in range(max_iter):
        old_denom = denom
        denom += 2

        term = term * Z2 * (old_denom / denom)
        s_new = s + term

        scale = np.maximum(1.0, np.abs(s_new))

        if np.max(np.abs(term) / scale) < tol:
            s = s_new
            break

        s = s_new

    return (2 / (1j * np.pi)) * s





def C_plus_unit_T_k_theorique(k, X_cheb):
    sortie = -(1/2) * (Psi_k(k,Inv_Jacouwski_bas(X_cheb)) + Psi_k(-k,Inv_Jacouwski_bas(X_cheb)))
    return sortie

def T_k(k, X_cheb):
    N_cheb = len(X_cheb)
    if k == 0:
        t_k = np.ones(N_cheb)
    if k == 1:
        t_k = X_cheb
    if k > 1 :
        t_k = 2 * X_cheb * T_k(k-1, X_cheb) - T_k(k-2, X_cheb)
    return t_k

# C[II, gamm] pas connecter

def Psi_mat_rec(k):
    if k % 2 !=0:
        sortie =  2 / (1j * np.pi * k)
    if k % 2 ==0:
        sortie = 0
    return sortie

def Psi_matrice(Z, n):
    m = len(Z)
    psi_pre_mat = np.zeros([m, 2 * n - 1], dtype=complex)
    psi_pre_mat[:, 0] = Psi_k(1 - n, Z)
    psi_pre_mat[:, n - 1] = Psi_k(0, Z)

    for ii in range(n-2):
        k = ii + 1
        psi_pre_mat[:, k] = Z * psi_pre_mat[:, k - 1] - Psi_mat_rec(k - n + 1)

    for jj in range(n - 1):
        l = jj + 1
        psi_pre_mat[:, n - 1 + l] = Z * psi_pre_mat[:, n + l - 2] - Psi_mat_rec(l)


    mat = np.zeros([m, n], dtype=complex)
    mat[:, 0] = 2 * psi_pre_mat[:, n - 1]
    for jj in range(n - 1):
        mat[:, jj + 1] = psi_pre_mat[:, n - 1 - (jj + 1)] + psi_pre_mat[:, n - 1 + (jj + 1)]
    return -0.5 * mat

# C[II, gamm] connecter

def Mu_rec(n, cote):
    mu = np.zeros(n, dtype=complex)
    mu[0] = 0
    for k in range(1, n):
        if k % 2 == 0:
            terme = 0
        if k % 2 != 0:
            terme = (cote ** k) / k
        mu[k] = mu[k -1] + terme

    sortie = np.zeros(n, dtype=complex)
    sortie[0] = 0
    for jj in range(1, n):
        sortie[jj] = (mu[jj] + mu[jj - 1]) * (cote ** jj)

    return sortie * (1/(1j * np.pi))

def R_lr_vect(n, cote):
    r = np.zeros(n, dtype=complex)
    for ii in range(n):
        r[ii] = (1/(2 * 1j * np.pi)) * (cote ** (ii + 1))

    return r

def line_finit_part(n, cote, theta):
    r = R_lr_vect(n, cote)
    mu = Mu_rec(n, cote)
    lin = mu + (1j * np.angle(cote * np.exp(1j * theta)) - np.log(2)) * r
    return lin

def C_unit_gamma(con, X_cheb, n, F, Mob):
    A, B, C, D = Mob
    Z = Inv_Jacouwski_plus(Inv_Mobius_general(X_cheb, A, B, C, D))
    m = len(Z)
    Coch = np.zeros([m, n], dtype=complex)
    if (con[0] == 0 and con[1] == 0):
        Coch = Psi_matrice(Z, n) @ F
    if (con[0] != 0 and con[1] == 0):
        if  np.isclose(Mobius_general(A, B, C, D, con[0]), -1):
            theta = -np.angle(Mobius_general_der( A, B, C, D, con[0]))
            line = line_finit_part(n, con[0], theta)
            Coch[0,:] = line
            Coch[1:,:] = Psi_matrice(Z[1:], n)
            Coch = Coch @ F
        if np.isclose(Mobius_general(A, B, C, D, con[0]), 1):
            theta = -np.angle( - Mobius_general_der( A, B, C, D, con[0]))
            line = line_finit_part(n, con[0], theta)
            Coch[-1, :] = line
            Coch[:-1, :] = Psi_matrice(Z[:-1], n)
            Coch = Coch @ F

    if (con[0] == 0 and con[1] != 0):
        if  np.isclose(Mobius_general(A, B, C, D, con[1]), -1):
            theta = -np.angle(Mobius_general_der( A, B, C, D, con[1]))
            line = line_finit_part(n, con[1], theta)
            Coch[0,:] = line
            Coch[1:,:] = Psi_matrice(Z[1:], n)
            Coch = Coch @ F
        if np.isclose(Mobius_general(A, B, C, D, con[1]), 1):
            theta = -np.angle( - Mobius_general_der( A, B, C, D, con[1]))
            line = line_finit_part(n, con[1], theta)
            Coch[-1, :] = line
            Coch[:-1, :] = Psi_matrice(Z[:-1], n)
            Coch = Coch @ F


    if (con[0] !=0 and con[1] != 0):
        if np.isclose(Mobius_general(A, B, C, D, con[0]), -1):
            theta = -np.angle(Mobius_general_der( A, B, C, D, con[0]))
            line = line_finit_part(n,  con[0], theta)
            Coch[0, :] = line
            Coch[1:-1, :] = Psi_matrice(Z[1:-1], n)

        if np.isclose(Mobius_general(A, B, C, D, con[0]), 1):
            theta = -np.angle(-Mobius_general_der(A, B, C, D, con[0]))
            line = line_finit_part(n,  con[0], theta)
            Coch[-1, :] = line
            Coch[1:-1, :] = Psi_matrice(Z[1:-1], n)

        if np.isclose(Mobius_general(A, B, C, D, con[1]), -1):
            theta = -np.angle(Mobius_general_der( A, B, C, D, con[1]))
            line = line_finit_part(n, con[1], theta)
            Coch[0, :] = line

        if np.isclose(Mobius_general(A, B, C, D, con[1]), 1):
            theta = -np.angle(-Mobius_general_der( A, B, C, D, con[1]))
            line = line_finit_part(n, con[1], theta)
            Coch[-1, :] = line

        Coch = Coch @ F

    return Coch

# C+[gamm,gamma]  avec M affine puis M génrale, mais Gamma borné

def C_plus_gamma_gamma_affin(X_cheb, V, V_inv, Mob):
    """
    Definition de C+[gamm,gamma] avec M affine, voir définition 5.6 (de l'articl)
    :param X_cheb: point d'interpollation sur l'ontervalla -1, 1 du second type de chebychev
    :param V: coef -> valeur
    :param V_inv: inv de V
    :param Mob: param de la transfor de mobius (ici affine) gamma -> [-1,1]
    :return: C+[gamm,gamma] de la def 5.6
    """
    n = len(X_cheb)
    A, B, C, D = Mob
    z_g = Inv_Mobius_general(-1, A, B, C, D)
    z_d = Inv_Mobius_general(1, A, B, C, D)
    corr = np.zeros([n,n], dtype=complex)
    corr[0, 0] = -np.log(np.abs(Mobius_general_der(A, B, C, D, z_g)))
    corr[ -1, -1] = np.log(np.abs(Mobius_general_der(A, B, C, D, z_d)))
    C = C_plus_unit_unit(X_cheb, V, V_inv)
    c_plus = C + (1 / (2 * 1j * np.pi)) * corr
    return c_plus

def C_plus_gamma_gamma_borne(X_cheb, V, V_inv, Mob):
    """
    Definition de C+[gamm,gamma] avec gamma borné, voir définition 5.9 (de l'articl)
    :param X_cheb: point d'interpollation sur l'ontervalla -1, 1 du second type de chebychev
    :param V: coef -> valeur
    :param V_inv: inv de V
    :param Mob: param de la transfor de mobius (ici affine) gamma -> [-1,1]
    :return: C+[gamm,gamma] de la def 5.9
    """
    n = len(X_cheb)
    A, B, C, D = Mob
    Minf = Mobius_inf(A , C)
    C = C_plus_gamma_gamma_affin(X_cheb, V, V_inv, Mob)
    if not np.isinf(Minf):
        z = Inv_Jacouwski_plus(Minf)
        psi_vect = Psi_matrice(np.array([z]), n)
        corr = np.ones([n, n], dtype=complex) @ np.diag(psi_vect[0]) @ V_inv
    if np.isinf(Minf):
        corr = 0
    return C - corr

# C[gamma,omega]  avec M affine puis M génrale, mais Gamma borné

def C_gamma_omega_affin(X_cheb_gamma, X_cheb_omega, V_inv, Mob_gamma, Mob_omega):
    n_gamma = len(X_cheb_gamma)
    n_omega = len(X_cheb_omega)
    A_inv_gamma, B_inv_gamma, C_inv_gamma, D_inv_gamma \
        = Inv_Mobius_coef(Mob_gamma[0], Mob_gamma[1], Mob_gamma[2], Mob_gamma[3])
    A_inv_omega, B_inv_omega, C_inv_omega, D_inv_omega \
        = Inv_Mobius_coef(Mob_omega[0], Mob_omega[1], Mob_omega[2], Mob_omega[3])
    Mob_gamma_inv = (A_inv_gamma, B_inv_gamma, C_inv_gamma, D_inv_gamma)
    A, B, C, D = Mobius_compositions(Mob_omega, Mob_gamma_inv)
    Mob = (A, B, C, D)
    z_g = Mobius_general(A_inv_gamma, B_inv_gamma, C_inv_gamma, D_inv_gamma, -1)
    z_d = Mobius_general(A_inv_gamma, B_inv_gamma, C_inv_gamma, D_inv_gamma, 1)
    p_g = Mobius_general(A_inv_omega, B_inv_omega, C_inv_omega, D_inv_omega, -1)
    p_d = Mobius_general(A_inv_omega, B_inv_omega, C_inv_omega, D_inv_omega, 1)
    con = np.zeros(2)
    corr1 = np.zeros([n_omega, n_gamma], dtype=complex)
    corr2 = np.zeros([n_omega, n_gamma], dtype=complex)
    if np.isclose(p_g, z_g):
        con[0] = -1
        corr1[0,0] = -(np.log(np.abs(Mobius_general_der(Mob_gamma[0], Mob_gamma[1], Mob_gamma[2], Mob_gamma[3], z_g)))
                       / (2 * 1j * np.pi))
    if np.isclose(p_d, z_g):
        con[0] = -1
        corr1[-1, 0] = -(np.log(np.abs(Mobius_general_der(Mob_gamma[0], Mob_gamma[1], Mob_gamma[2], Mob_gamma[3], z_g)))
                         / (2 * 1j * np.pi))
    if np.isclose(p_g, z_d):
        con[1] = 1
        corr2[0, -1] = (np.log(np.abs(Mobius_general_der(Mob_gamma[0], Mob_gamma[1], Mob_gamma[2], Mob_gamma[3], z_d)))
                        / (2 * 1j * np.pi))
    if np.isclose(p_d, z_d):
        con[1] = 1
        corr2[-1, -1] = (np.log(np.abs(Mobius_general_der(Mob_gamma[0], Mob_gamma[1], Mob_gamma[2], Mob_gamma[3], z_d)))
                         / (2 * 1j * np.pi))
    #print("z_g, z_d, p_g, p_d :", z_g, z_d, p_g, p_d)
    #print("con :", con)
    C = C_unit_gamma(con, X_cheb_omega, n_gamma, V_inv, Mob) + corr1 + corr2
    return C

def C_gamma_omega_borne(X_cheb_gamma, X_cheb_omega, V_inv, Mob_gamma, Mob_omega):
    n_gamma = len(X_cheb_gamma)
    n_omega = len(X_cheb_omega)
    A_gamma, _, C_gamma, _ = Mob_gamma
    M_inf_gamma = Mobius_inf(A_gamma, C_gamma)
    if not np.isinf(M_inf_gamma):
        z = Inv_Jacouwski_plus(M_inf_gamma)
        psi_vect = Psi_matrice(np.array([z]), n_gamma)
        corr = np.ones([n_omega, n_gamma], dtype=complex) @ np.diag(psi_vect[0]) @ V_inv
    if np.isinf(M_inf_gamma):
        corr = 0
    C = C_gamma_omega_affin(X_cheb_gamma, X_cheb_omega, V_inv, Mob_gamma, Mob_omega) - corr
    return C

# Assamblage borné

def C_plus_assambalge_borne(liste_dico, N_interpol, N_interpol_courb):
    C = np.zeros([N_interpol, N_interpol], dtype=complex)
    N = len(liste_dico)
    #print("N :", N)
    for ii in range(N):
        for jj in range(N):
            if ii == jj:
                X_cheb = liste_dico[ii]['X_cheb']
                V = liste_dico[ii]['V']
                F = liste_dico[ii]['F']
                Mob = liste_dico[ii]['Mob']
                C_ii_ii = C_plus_gamma_gamma_borne(X_cheb, V, F, Mob)
                ii_indice = (ii * N_interpol_courb, (ii + 1) * N_interpol_courb)
                C[ii_indice[0] : ii_indice[1], ii_indice[0] : ii_indice[1]] = C_ii_ii
            if ii != jj:
                X_cheb_gamma = liste_dico[ii]['X_cheb']
                X_cheb_omega = liste_dico[jj]['X_cheb']
                F_gamma = liste_dico[ii]['F']
                Mob_gamma = liste_dico[ii]['Mob']
                Mob_omega = liste_dico[jj]['Mob']
                ii_indice = (ii * N_interpol_courb, (ii + 1) * N_interpol_courb)
                jj_indice = (jj * N_interpol_courb, (jj + 1) * N_interpol_courb)
                C_ii_jj = C_gamma_omega_borne(X_cheb_gamma, X_cheb_omega, F_gamma, Mob_gamma, Mob_omega)
                C[jj_indice[0] : jj_indice[1], ii_indice[0] : ii_indice[1]] = C_ii_jj


    return C

def C_moin_assambalge_borne(liste_dico, N_interpol, N_interpol_courb):
    c_plus = C_plus_assambalge_borne(liste_dico, N_interpol, N_interpol_courb)
    C = c_plus - np.eye(N_interpol, dtype=complex)
    return C

# Assamblage final de l'opérateur discretiser sur les branche du RHP ici, pour des cercle autour des singularitées


def Operateur_bloc_cons(W_global, j,k, c_moin):
    bloc = Delta(j,k) * np.eye(2, dtype=complex) - c_moin[j,k] * W_global[j].T
    return bloc

def Non_homogène(W_global, k):
    e = np.ones(2, dtype=complex)
    b = W_global[k].T @ e
    return b


def Non_homogène_deriv_x(W_x_global, c_moin, N_interpol,U_phys):
    Mat = np.zeros([2 * N_interpol,2 *  N_interpol], dtype=complex)
    terme1 = np.zeros([N_interpol, 2], dtype=complex)
    U_phys = U_phys.reshape(-1)
    for ii in range(N_interpol):
        terme1[ii,:] = Non_homogène(W_x_global, ii)
        for jj in range(N_interpol):
            bloc = c_moin[ii,jj] * W_x_global[ii].T
            Mat[ii * 2 : ii * 2 + 2, jj * 2 : jj * 2 + 2] = bloc
    b = terme1.reshape(-1) + (Mat @ U_phys)
    return b

def Operateur(W_global, c_moin, N_interpol):
    op = np.zeros([2 * N_interpol,2 *  N_interpol], dtype=complex)
    b = np.zeros([N_interpol, 2], dtype=complex)
    for ii in range(N_interpol):
        b_scal = Non_homogène(W_global, ii)
        b[ii, :] = b_scal
        for jj in range(N_interpol):
            bloc = Operateur_bloc_cons(W_global, ii, jj, c_moin)
            op[ii * 2 : ii * 2 + 2, jj * 2 : jj * 2 + 2] = bloc
    return op, b.reshape(-1)


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
    z_prim = Mobius_general(A, B, C, D, np.array([z]))              # on place z dans les coordonnée de la courbe
    z_prim_cercle = np.array([Inv_Jacouwski_plus(z_prim)], dtype=complex)          # Transfo de jacouwski pour arriver sur le cercle
    C_mat = Psi_matrice(z_prim_cercle, n) @ F
    # correction de déformation
    infinit = Mobius_inf(A, C)
    if not np.isinf(infinit):
        z_inf = np.array([Inv_Jacouwski_plus(np.array([infinit]))], dtype=complex)
        C_mat = C_mat - Psi_matrice(z_inf, n) @ F

    # C_mat a la forme (1, n), donc le produit matriciel renvoie un
    # tableau de forme (1,). La grille attend ici un scalaire complexe.
    return (C_mat @ u_phys[:, 0])[0], (C_mat @ u_phys[:, 1])[0]

def Evalulation_Cauchy(z, u_list, liste_dico):
    """
    Evalue la valeur de la transfo de chauchy au point z sur tt les contours
    :param z: point d'éval
    :param u_phys: valeur de la fonction sur la courbe
    :param dico: dictionaire associé à la curbe
    :return: valeur de la TF de cauchy de la fonction au point z
    """

    Cu1 = 0
    Cu2 = 0
    compteur = 0
    for dico in liste_dico:
        Cu1_compteur, Cu2_compteur = Evalulation_Cauchy_une_courbe(z, u_list[compteur], dico)
        Cu1 += Cu1_compteur
        Cu2 += Cu2_compteur
        compteur += 1
    return Cu1, Cu2

def distance_contour(z, list_dico):
    liste_dist = []
    for dico in list_dico:
        n = dico["n"]
        x_phys = dico["x_phys"]
        for ii in range(n):
            dist = np.abs(z - x_phys[ii])
            liste_dist.append(dist)
    minimum = min(liste_dist)
    return minimum


def Evaluation_cauchy_grid(z_grid, u_phys, liste_dico, dist_min_cont):
    """
    évalue la TF de cauchy sur une grille de points complexe pour un ensemble de courbes
    :param z_grid: grille de points complexe
    :param u_phys:valeur de la fonction sur les courbes
    :param liste_dico:dictionnaire contenant les info sur les courbes
    :return:
    """
    u_list = Decupage_U(u_phys, liste_dico)
    (n_ii, n_jj) = z_grid.shape
    Cu1_grid = np.zeros([n_ii,n_jj], dtype=complex)
    Cu2_grid = np.zeros([n_ii,n_jj], dtype=complex)
    for ii in range(n_ii):
        for jj in range(n_jj):
            z = z_grid[ii, jj]
            d = distance_contour(z, liste_dico)
            if d >= dist_min_cont:
                Cu1_grid[ii, jj], Cu2_grid[ii, jj] = Evalulation_Cauchy(z, u_list, liste_dico)
            if d < dist_min_cont:
                Cu1_grid[ii, jj], Cu2_grid[ii, jj] = np.nan, np.nan
                print("trop proche !")
            print(ii,jj)
    return Cu1_grid, Cu2_grid
