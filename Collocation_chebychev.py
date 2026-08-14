 # Import
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander

# Fonctions
def Cheb_point(L, N, oriant = 1, bord = 2):
    """
    Génère les points de collocation et la matrice d'évaluation.

    :param L: rayon de l'intervalle
    :param N: nombre de points
    :return: points de collocation et matrice de Vandermonde de Chebyshev
    """
    N_vect = np.arange(N + bord, dtype=complex)
    Int_tot = -oriant * np.cos(np.pi * (N_vect / (N + bord - 1)))
    V_tot = chebvander(Int_tot, N + bord - 1)
    if bord == 2:
        Int =  Int_tot[1:-1]
        V = V_tot[1:-1, 1:-1]
        #print("min Int , max Int", Int[0], Int[-1])
        #print("taill int :", len(Int))
        #print("forme de V :", np.shape(V))
    if bord == 0 :
        Int =  Int_tot
        V = V_tot
        #print("min Int , max Int", min(Int), max(Int))
        #print("taill int :", len(Int))
        #print("forme de V :", np.shape(V))
    if (bord != 0 and bord != 2) :
        Int = np.zeros(N)
        V = np.zeros([N, N])
        print(" Attention, erreur dans bord, la fnction ne retoune que des 0")
    return L * Int, V


def Diff_cheb_1(N):
    D = np.zeros([N, N])

    for n in range(1, N):

        if n % 2 == 0:
            # n pair : T_n' = 2n (T_{n-1} + T_{n-3} + ... + T_1)
            D[1:n:2, n] = 2 * n
        else:
            # n impair : T_n' = n T_0 + 2n (T_{n-1} + T_{n-3} + ... + T_2)
            D[0, n] = n
            D[2:n:2, n] = 2 * n

    return D

def system(D_1, D_2, U, u, s):
    M1 = D_2 + 2 * 1j * s * D_1 + U
    M2 = D_2 - 2 * 1j * s * D_1 + U

    # Conditions aux bords
    # valeur 0 en -infty

    M1[0,:] = 0.0
    M1[0,0]=1.0
    b_gauche = -u.copy()
    b_gauche[0] = 0

    # valeur 0 en +infty

    M2[-1, :] = 0.0
    M2[-1, -1] = 1.0
    b_droite = -u.copy()
    b_droite[-1] = 0

    # derivée 0 en -infty

    M1[-1, :] = D_1[0, :]
    b_gauche[-1] = 0.0

    # derivée 0 en +infty

    M2[0, :] = D_1[-1, :]
    b_droite[0] = 0.0

    return M1, M2, b_gauche, b_droite

def reconstruction(X_cheb, m1, m2, b_coef, s, j0):


    #j0 = N // 2
    f_p_bar = m1 * np.exp(1j * X_cheb * s) * b_coef
    f_m = m2 * np.exp(-1j * X_cheb * s)

    f_p_bar_d = f_p_bar[j0:]
    f_m_g = f_m[:j0]

    mu = []
    for aa in range(len(X_cheb)):
        if aa < j0 :
            mu.append(f_m_g[aa])
        else :
            mu.append(f_p_bar_d[aa - j0])
    return np.array(mu)

def visu_mu(mu, X_cheb, m1, m2):

    plt.plot(X_cheb, np.real(mu), label="mu Re")
    plt.plot(X_cheb, np.imag(mu), label="mu Im")
    plt.legend()
    plt.grid()
    plt.show()

    indice_max=np.argmax(np.abs(mu))
    X_max = X_cheb[indice_max]
    plt.plot(X_cheb, np.real(mu), label="mu Re")
    plt.plot(X_cheb, np.imag(mu), label="mu Im")
    plt.legend()
    #plt.yscale('symlog', linthresh=1e-10)
    #plt.ylim(-10, 10)
    plt.xlim(X_max - 10, X_max + 10)
    plt.grid()
    plt.show()

    plt.plot(X_cheb, np.real(m1), label="m1 Re")
    plt.plot(X_cheb, np.imag(m1), label="m1 Im")
    plt.legend()
    plt.title("m1")
    plt.show()

    plt.plot(X_cheb, np.real(m2), label="m1 Re")
    plt.plot(X_cheb, np.imag(m2), label="m1 Im")
    plt.legend()
    plt.title("m2")
    plt.show()


    return

def rho(N, D_1, D_2, U, u, s):


    M1, M2, b_g, b_d = system(D_1, D_2, U, u, s)
    M1_m, M2_m, b_g_m, b_d_m = system(D_1, D_2, U, u, -s)

    m1 = np.linalg.solve(M1, b_g) + 1
    m2 = np.linalg.solve(M2, b_d) + 1

    m1m = np.linalg.solve(M1_m, b_g_m) + 1
    m2m = np.linalg.solve(M2_m, b_d_m) + 1

    # Dérivées
    dm1 = D_1 @ m1
    dm2 = D_1 @ m2
    dm1m = D_1 @ m1m
    dm2m = D_1 @ m2m

    # Choix du point x0
    j0 = N // 2
    x0 = 0

    # Extraction des valeurs au point x0
    m1_0 = m1[j0]
    m2_0 = m2[j0]
    m1m_0 = m1m[j0]
    m2m_0 = m2m[j0]

    dm1_0 = dm1[j0]
    dm2_0 = dm2[j0]
    dm1m_0 = dm1m[j0]
    dm2m_0 = dm2m[j0]




    # Coefficients a(s) et b(s)
    b_coef = -(m1m_0 * dm2_0 - dm1m_0 * m2_0)                      # déja simplifier

    a_coef = ((m1m_0 * dm2m_0 - dm1m_0 * m2m_0) + 2 * 1j * s * (m1m_0 * m2m_0))                 # déja symplifier
    if np.abs(a_coef) ==0:
        rho = b_coef * 0
        print(" Warning : a_coef is close to 0. If you want b on a pole of rho ignore this warning")
    else:
        rho = b_coef / a_coef

    return  rho

def Cheb_int_poid(n):
    if n == 0:
        poid = 0
    if n % 2 ==0 :
        poid = 2 /(1- (n ** 2))
    else :
        poid = 0
    return poid


def cheb_int(mu,V_inv, L, n=2):
    N = len(mu)
    cheb_coef = V_inv @ (mu ** n)
    int = 0
    for ii in range(N):
        int += cheb_coef[ii] * Cheb_int_poid(ii)
    return L * int



def residus( D_1, D_2, U, u, s, X_cheb, j0, V_inv, L):
    M1, M2, b_g, b_d = system(D_1, D_2, U, u, s)

    m1 = np.linalg.lstsq(M1, b_g, rcond=None)[0] + 1
    m2 = np.linalg.lstsq(M2, b_d, rcond=None)[0] + 1


    # Extraction des valeurs au point x0
    m1_0 = m1[j0]
    m2_0 = m2[j0]


    b_coef = (m2_0 / m1_0) * np.exp(-2 * 1j * s * X_cheb[j0])
    mu = reconstruction(X_cheb, m1, m2, b_coef, s, j0)
    # Diagnostique
    left =  m2[j0] * np.exp(-1j * s * X_cheb[j0])
    right = b_coef * m1[j0] * np.exp(+1j * s * X_cheb[j0])

    print("Continuité test :", left - right)


    #############
    visu_mu(mu, X_cheb, m1, m2)

    I = cheb_int(mu, V_inv, L)

    #I = np.trapezoid(mu ** 2, X_cheb)

    return b_coef, I




 # Corps du code
#N = 1000                                                # Doit être paire !
#L = 100


# Paramètres
#v = 2.4 * 2
#deta = 0




# Construction du système en nodale
#X, V = Cheb_point(L, N)
#D = Diff_cheb_1(N)

# Soliton à l'instant t = 0
#u = (v / 2) * (np.cosh((np.sqrt(v) / 2) * (X - deta))) ** (-2)

# Presque soliton a t=0
#u = (v / 2) * (np.cosh((X - deta))) ** (-2)




#U = np.diag(u)
#V_inv = np.linalg.solve(V, np.eye(N + 1))

#D_1 = V @ D @ V_inv / L
#D_2 = V @ D @ D @ V_inv / (L ** 2)

#S = np.linspace(-15,15,N//10)
#rho_list = []
#for s in S:
#    print(s)
#    rho_list.append(rho(N, D_1, D_2, U, u, s))
#rho_array = np.array(rho_list)
#plt.plot(S, np.real(rho_array), color='red')
#plt.plot(S, np.imag(rho_array), color='blue')
#plt.show()