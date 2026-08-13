for j in range(len(mu_list)):

    kap = float(z_0[j])  # kap = kappa_j
    mu = np.array(mu_list[j], dtype=complex).copy()

    # --- fenêtre de queue dépendante de kap ---
    x_cap_j = min(float(X.max()), np.log(1.0 / eps) / kap)
    x_tail_j = min(float(x_queu), 0.8 * x_cap_j)  # garantit x_tail < x_cap

    right = (X > x_tail_j) & (X < x_cap_j)
    left = (X < -x_tail_j) & (X > -x_cap_j)

    if right.sum() == 0 or left.sum() == 0:
        raise ValueError(
            f"Fenêtre vide pour j={j}: kap={kap}, x_tail={x_tail_j}, x_cap={x_cap_j}"
        )

    # --- A_plus stable (scalé) ---
    # A+ ~ mean( mu(x) * exp(kap*x) )  (car mu ~ A+ e^{-kap x})
    # On scale pour éviter exp(kap*x) énorme:
    x0 = np.max(X[right])
    A_plus_scaled = np.mean(mu[right] * np.exp(kap * (X[right] - x0)))  # exp<=1

    if (not np.isfinite(A_plus_scaled)) or abs(A_plus_scaled) < 1e-300:
        raise ValueError(
            f"A_plus_scaled invalide pour j={j} (trop loin dans la queue). "
            f"Essaye de diminuer x_queu ou d'augmenter eps (ex eps=1e-10)."
        )

    # Fixe la phase pour rendre A_plus_scaled réel positif
    mu /= (A_plus_scaled / abs(A_plus_scaled))
    A_plus_scaled = np.mean(mu[right] * np.exp(kap * (X[right] - x0)))

    # Normalisation Jost à droite: A+ = 1
    # A_plus = A_plus_scaled * exp(kap*x0)
    muJ = mu * np.exp(-kap * x0) / A_plus_scaled

    # (optionnel mais souvent utile) rendre muJ quasi réel si potentiel réel

    idx = np.argmax(np.abs(muJ))
    muJ = muJ * np.exp(-1j * np.angle(muJ[idx]))  # rend muJ[idx] réel positif
    # muJ = np.real(muJ)  # potentiel réel -> eigenfonction réelle

    # --- b(kappa) fourni en entrée ---
    bk = b_z_0[j]

    if abs(bk) < 1e-300:
        raise ValueError(f"b(kappa) ~ 0 pour j={j}. Impossible de calculer a'.")

    # --- intégrale I = ∫ mu^2 dx ---
    I = np.trapezoid((muJ ** 2) * w, Y)

    print("I", I)

    # --- a'(kappa) et résidu ---
    a_p = I / (1j * bk)
    print("a_p", a_p)
    C_j = bk / a_p  # équivalent à 1j * bk**2 / I

    a_prime.append(a_p)
    C.append(C_j)

return np.array(C), np.array(a_prime)