# Imports
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander


# Fonctions
def Cheb_point(L, N):
    """
    Génère les points de collocation et la matrice d'évaluation.

    :param L: rayon de l'intervalle
    :param N: nombre de points
    :return: points de collocation et matrice de Vandermonde de Chebyshev
    """
    Int = np.cos(np.pi * np.arange(N + 1) / N)
    V = chebvander(Int, N)
    return L * Int, V


def Diff_cheb_1(N):
    D = np.zeros([N + 1, N + 1])

    for n in range(1, N + 1):
        if n % 2 == 0:
            # n pair : T_n' = 2n (T_{n-1} + T_{n-3} + ... + T_1)
            D[1:n:2, n] = 2 * n
        else:
            # n impair : T_n' = n T_0 + 2n (T_{n-1} + T_{n-3} + ... + T_2)
            D[0, n] = n
            D[2:n:2, n] = 2 * n

    return D


def scalar_1x1(expr):
    """
    Convertit une matrice SymPy 1x1 en scalaire SymPy.

    Les produits du type ligne @ matrice @ colonne retournent une matrice 1x1.
    Pour faire une division symbolique correcte, on extrait l'unique coefficient.
    """
    return sy.Matrix(expr)[0, 0]


# Corps du code
N = 8                                                # Doit être paire !
L = 10

x, s = sy.symbols("x s", real=True)

# Paramètres
v = 1
deta = 1

# Soliton à l'instant t = 0
u = (v / 2) * sy.sech((sy.sqrt(v) / 2) * (x - deta)) ** 2


# Construction du système en nodale
X, V = Cheb_point(L, N)
D = Diff_cheb_1(N)

u_num = sy.lambdify(x, u, modules="numpy")
b_values = u_num(X)

U = np.diag(b_values)
V_inv = np.linalg.solve(V, np.eye(N + 1))

D_1 = V @ D @ V_inv / L
D_2 = V @ D @ D @ V_inv / (L ** 2)

M1 = D_2 + 2 * 1j * s * D_1 - U
M2 = D_2 - 2 * 1j * s * D_1 - U


# Conditions aux bords
M1 = np.vstack([
    M1[1:N, :],
    np.eye(N + 1)[0, :],
    D_1[0, :]
])

M2 = np.vstack([
    M2[1:N, :],
    np.eye(N + 1)[-1, :],
    D_1[-1, :]
])

rhs = np.hstack([
    b_values[1:N],
    0.0,
    0.0
])


# Conversion en matrices SymPy
M1_sym = sy.Matrix(M1)
M2_sym = sy.Matrix(M2)
b_sym = sy.Matrix(rhs)

D_1_sym = sy.Matrix(D_1)


# Résolution symbolique
m1 = M1_sym.LUsolve(b_sym) + sy.ones(N + 1, 1)
m2 = M2_sym.LUsolve(b_sym) + sy.ones(N + 1, 1)
print("hello")


# Versions évaluées en -s
m1_minus = m1.subs(s, -s)
m2_minus = m2.subs(s, -s)

# Dérivées nodales
dm1 = D_1_sym * m1
dm2 = D_1_sym * m2

dm1_minus = D_1_sym * m1_minus
dm2_minus = D_1_sym * m2_minus

# Choix du point x0
j0 = N // 2
x0 = 0

# Extraction des valeurs au point x0
m1_0 = m1[j0, 0]
m2_0 = m2[j0, 0]
m1m_0 = m1_minus[j0, 0]
m2m_0 = m2_minus[j0, 0]

dm1_0 = dm1[j0, 0]
dm2_0 = dm2[j0, 0]
dm1m_0 = dm1_minus[j0, 0]
dm2m_0 = dm2_minus[j0, 0]
print("I'm still a live")
# Coefficients a(s) et b(s)
b_coef = -(m1m_0 * dm2_0 - dm1m_0 * m2_0)

a_coef = m1m_0 * m2m_0 + (m1m_0 * dm2m_0 - dm1m_0 * m2m_0)

# Coefficient de réflexion
#print("see you after the big SYMPLIFICATIONALISATIOn")
#b_coef_symp = sy.cancel(b_coef)
#print("almost there")
#a_coef_symp = sy.cancel(a_coef)
#print("hi")
#rho = sy.sympify(b_coef_symp/a_coef_symp)
rho= b_coef / a_coef
print('hi ih')
# plot
rho_num = sy.lambdify(s, rho, modules="numpy")
print("hi hi")
S = np.linspace(-15,15,100*N)
plt.plot(S, np.real(rho_num(S)), color='red')
plt.plot(S, np.imag(rho_num(S)), color='blue')
plt.show()
