# Import
import numpy as np
import matplotlib.pyplot as plt
from Main_multi_soliton_invers_scat import Scatt_invers_multi_soliton
from Fonction_utile import multi_soliton_phys
# import des donnée de scattering
A = [2.4, 1]
delta = [10, 500]

v1 = A[0] * 2
v2 = A[1] * 2

t_x = (delta[1] - delta[0]) / (v1 - v2)
x_t = delta[0] + v1 * t_x
#v3 = (x_t - delta[2]) / t_x
#A.append((v3 / 2))

scattering_data = np.load("Scattering_data_trois_soliton_A_[2.4, 1]_Delt_[10, 5]N_cheb_1001_N_rho_10_[-5, 5].npz", allow_pickle=True)

# param
N_interpol_courb = 50
Nx = 100
Nt = 100
x_lareur = 7
t_lareur = 7
epsilon_t = 1
epsilon_x = 5
x_values = np.linspace(x_t - x_lareur, x_t + x_lareur, Nx)

t_values = np.linspace(t_x - t_lareur, t_x + t_lareur, Nt)

t_moins = t_x - epsilon_t
t_plus = t_x + epsilon_t
q_x_col_moin = np.zeros(Nx, dtype=complex)
q_x_col_centre = np.zeros(Nx, dtype=complex)
q_x_col_plus = np.zeros(Nx, dtype=complex)

x_moins = x_t - epsilon_x
x_plus = x_t + epsilon_x
q_t_col_moin = np.zeros(Nt, dtype=complex)
q_t_col_centre = np.zeros(Nt, dtype=complex)
q_t_col_plus = np.zeros(Nt, dtype=complex)


for ii in range(Nx):
    print(ii)
    q_x_col_moin[ii], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A, delta, x_values[ii], t_moins, scattering_data)
    q_x_col_centre[ii], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A, delta, x_values[ii], t_x, scattering_data)
    q_x_col_plus[ii], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A, delta, x_values[ii], t_plus, scattering_data)

for jj in range(Nt):
    print(jj)
    q_t_col_moin[jj], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A, delta, x_moins, t_values[jj],  scattering_data)
    q_t_col_centre[jj], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A,delta, x_t, t_values[jj],  scattering_data)
    q_t_col_plus[jj], _, _ = Scatt_invers_multi_soliton(N_interpol_courb, A,delta, x_plus, t_values[jj],  scattering_data)



q_x_num = np.vstack([q_x_col_moin,q_x_col_centre,q_x_col_plus])

q_t_num = np.vstack([q_t_col_moin,q_t_col_centre,q_t_col_plus])

t_fixes = np.array([t_moins, t_x, t_plus])
x_fixes = np.array([x_moins, x_t, x_plus])

q_x_theorique = np.zeros((3, Nx), dtype=complex)
q_t_theorique = np.zeros((3, Nt), dtype=complex)

for kk in range(3):
    q_x_theorique[kk, :] = np.asarray(multi_soliton_phys(A, delta, x_values, t_fixes[kk])).reshape(-1)

for kk in range(3):
    for jj in range(Nt):
        q_t_theorique[kk, jj] = np.asarray(multi_soliton_phys(A,delta,np.array([x_fixes[kk]]),t_values[jj])).reshape(-1)[0]

erreur_abs_x = np.abs(q_x_num - q_x_theorique)
erreur_abs_t = np.abs(q_t_num - q_t_theorique)

seuil_relatif = 1e-14

erreur_rel_x = erreur_abs_x / np.maximum(np.abs(q_x_theorique),seuil_relatif)

erreur_rel_t = erreur_abs_t / np.maximum(np.abs(q_t_theorique),seuil_relatif)

print("Erreur absolue max x:", np.max(erreur_abs_x))
print("Erreur absolue max t:", np.max(erreur_abs_t))
print("Erreur relatif max x:", np.max(erreur_rel_x),"%")
print("Erreur relatif max t:", np.max(erreur_rel_t),"%")

plancher = np.finfo(float).eps

fig, axes = plt.subplots(3,1,figsize=(10, 11),sharex=True,constrained_layout=True)

for kk, ax in enumerate(axes):
    ax.plot(x_values,np.real(q_x_num[kk]),color="tab:blue",linewidth=1.6,label="RHP numérique")

    ax.plot(x_values,np.real(q_x_theorique[kk]),color="black",linestyle="--",linewidth=1.4,label="Solution théorique")


    ax.set_ylabel(r"$q(x,t)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$x$")
plt.show()


fig, axes = plt.subplots(3,1,figsize=(10, 11),sharex=True,constrained_layout=True)

for kk, ax in enumerate(axes):
    ax.plot(t_values, np.real(q_t_num[kk]), color="tab:blue", linewidth=1.6,label="RHP")

    ax.plot(
        t_values,
        np.real(q_t_theorique[kk]),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Solution théorique"
    )



    ax.set_ylabel(r"$q(x,t)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$t$")
plt.show()


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True,
    constrained_layout=True
)

for kk, ax in enumerate(axes):
    ax.semilogy(
        x_values,
        np.maximum(erreur_abs_x[kk], plancher),
        color="tab:red",
        linewidth=1.5,
        label=rf"$t={t_fixes[kk]:.4g}$")



    ax.set_ylabel("Erreur absolue")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$x$")
plt.show()


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True,
    constrained_layout=True
)

for kk, ax in enumerate(axes):
    ax.semilogy(
        t_values,
        np.maximum(erreur_abs_t[kk], plancher),
        color="tab:red",
        linewidth=1.5,
        label=rf"$x={x_fixes[kk]:.4g}$")



    ax.set_ylabel("Erreur absolue")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$t$")
plt.show()


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True,
    constrained_layout=True
)

for kk, ax in enumerate(axes):
    ax.semilogy(
        x_values,
        np.maximum(100 * erreur_rel_x[kk], plancher),
        color="tab:purple",
        linewidth=1.5,
        label=rf"$t={t_fixes[kk]:.4g}$")


    ax.set_title(rf"$t={t_fixes[kk]:.4g}$")
    ax.set_ylabel("Erreur relative (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$x$")
plt.show()


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True,
    constrained_layout=True
)

for kk, ax in enumerate(axes):
    ax.semilogy(
        t_values,
        np.maximum(100 * erreur_rel_t[kk], plancher),
        color="tab:purple",
        linewidth=1.5,
        label=rf"$x={x_fixes[kk]:.4g}$")



    ax.set_ylabel("Erreur relative (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"$t$")
plt.show()
