#!/usr/bin/env python3
r"""
floquet_solver.py - Floquet Monodromy Stability Engine for Precessing Domain Walls

Implements:
  1. Rigid wall background limit-cycle integration for \psi_1(\tau) over \tau \in [0, 2\pi].
  2. Simultaneous integration of the decoupled linear perturbation system for wavenumber \kappa:
       dQ_\kappa/d\tau  = (1/\Omega) * [ -\alpha * (\kappa^2/2) * Q_\kappa + (\kappa^2/2 + \cos\psi_1) * \Phi_\kappa ]
       d\Phi_\kappa/d\tau = (1/\Omega) * [ -(\kappa^2/2) * Q_\kappa - \alpha * (\kappa^2/2 + \cos\psi_1) * \Phi_\kappa ]
  3. Construction of the 2x2 Monodromy Matrix M_\kappa over one drive period \tau \in [0, 2\pi].
  4. Exact analytical eigenvalue calculation to obtain the Floquet multipliers \mu_{1,2}:
       |\mu_\kappa| = \max(|\mu_1|, |\mu_2|)
  5. Continuous growth rate:
       \lambda_\kappa = (\Omega / 2\pi) * \ln|\mu_\kappa|
  6. Maximum singular value (transient growth):
       \sigma_\kappa = \sqrt{\lambda_{\max}(M_\kappa^T M_\kappa)}
"""

import numpy as np


def solve_background_cycle(h0, ha, alpha=0.27, omega=1.0,
                           trans_cycles=200, check_cycles=100, steps_per_cycle=500, tol=1e-5):
    r"""
    Integrates the 0D rigid wall ODE to find the 2\pi-periodic background state \psi_1(\tau).
    Returns (converged, psi_cycle) where psi_cycle is an array of length steps_per_cycle.
    """
    dt = 2.0 * np.pi / steps_per_cycle
    A = (alpha * h0) / omega
    B = alpha / omega
    C = (alpha * ha) / omega

    tau_0 = np.arange(steps_per_cycle) * dt
    tau_half = (np.arange(steps_per_cycle) + 0.5) * dt
    cos_0 = np.cos(tau_0)
    cos_half = np.cos(tau_half)

    psi = 0.0

    # Transient relaxation
    for _ in range(trans_cycles):
        for s in range(steps_per_cycle):
            inext = (s + 1) % steps_per_cycle
            k1 = dt * (A - B * np.sin(psi) + C * cos_0[s])
            k2 = dt * (A - B * np.sin(psi + 0.5 * k1) + C * cos_half[s])
            k3 = dt * (A - B * np.sin(psi + 0.5 * k2) + C * cos_half[s])
            k4 = dt * (A - B * np.sin(psi + k3) + C * cos_0[inext])
            psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # Convergence check
    prev_strobe = psi % (2.0 * np.pi)
    converged = false = False

    for _ in range(check_cycles):
        for s in range(steps_per_cycle):
            inext = (s + 1) % steps_per_cycle
            k1 = dt * (A - B * np.sin(psi) + C * cos_0[s])
            k2 = dt * (A - B * np.sin(psi + 0.5 * k1) + C * cos_half[s])
            k3 = dt * (A - B * np.sin(psi + 0.5 * k2) + C * cos_half[s])
            k4 = dt * (A - B * np.sin(psi + k3) + C * cos_0[inext])
            psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        curr_strobe = psi % (2.0 * np.pi)
        diff = abs(curr_strobe - prev_strobe) % (2.0 * np.pi)
        circ_dist = 2.0 * np.pi - diff if diff > np.pi else diff
        if circ_dist < tol:
            converged = True
            break
        prev_strobe = curr_strobe

    if not converged:
        return False, None

    # Record one complete period
    psi_cycle = np.zeros(steps_per_cycle, dtype=np.float64)
    for s in range(steps_per_cycle):
        psi_cycle[s] = psi
        inext = (s + 1) % steps_per_cycle
        k1 = dt * (A - B * np.sin(psi) + C * cos_0[s])
        k2 = dt * (A - B * np.sin(psi + 0.5 * k1) + C * cos_half[s])
        k3 = dt * (A - B * np.sin(psi + 0.5 * k2) + C * cos_half[s])
        k4 = dt * (A - B * np.sin(psi + k3) + C * cos_0[inext])
        psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return True, psi_cycle


def compute_floquet_multiplier(psi_cycle, kappa, alpha=0.27, omega=1.0, steps_per_cycle=500):
    r"""
    Integrates the linearized 2x2 system over one period starting from (1,0) and (0,1).
    Returns largest Floquet multiplier modulus |\mu_\kappa|.
    """
    dt = 2.0 * np.pi / steps_per_cycle
    kappa2 = 0.5 * kappa * kappa

    # Column 1 initial condition: (1, 0)^T
    q1, p1 = 1.0, 0.0
    # Column 2 initial condition: (0, 1)^T
    q2, p2 = 0.0, 1.0

    for s in range(steps_per_cycle):
        inext = (s + 1) % steps_per_cycle
        th1 = psi_cycle[s]
        th2 = 0.5 * (psi_cycle[s] + psi_cycle[inext])  # Midpoint approximation
        th4 = psi_cycle[inext]

        cos1 = np.cos(th1)
        term1 = kappa2 + cos1
        cos2 = np.cos(th2)
        term2 = kappa2 + cos2
        cos4 = np.cos(th4)
        term4 = kappa2 + cos4

        # RK4 for columns 1 and 2
        # k1
        k1_q1 = dt * (-alpha * kappa2 * q1 + term1 * p1) / omega
        k1_p1 = dt * (-kappa2 * q1 - alpha * term1 * p1) / omega
        k1_q2 = dt * (-alpha * kappa2 * q2 + term1 * p2) / omega
        k1_p2 = dt * (-kappa2 * q2 - alpha * term1 * p2) / omega

        # k2
        q1_2, p1_2 = q1 + 0.5 * k1_q1, p1 + 0.5 * k1_p1
        q2_2, p2_2 = q2 + 0.5 * k1_q2, p2 + 0.5 * k1_p2
        k2_q1 = dt * (-alpha * kappa2 * q1_2 + term2 * p1_2) / omega
        k2_p1 = dt * (-kappa2 * q1_2 - alpha * term2 * p1_2) / omega
        k2_q2 = dt * (-alpha * kappa2 * q2_2 + term2 * p2_2) / omega
        k2_p2 = dt * (-kappa2 * q2_2 - alpha * term2 * p2_2) / omega

        # k3
        q1_3, p1_3 = q1 + 0.5 * k2_q1, p1 + 0.5 * k2_p1
        q2_3, p2_3 = q2 + 0.5 * k2_q2, p2 + 0.5 * k2_p2
        k3_q1 = dt * (-alpha * kappa2 * q1_3 + term2 * p1_3) / omega
        k3_p1 = dt * (-kappa2 * q1_3 - alpha * term2 * p1_3) / omega
        k3_q2 = dt * (-alpha * kappa2 * q2_3 + term2 * p2_3) / omega
        k3_p2 = dt * (-kappa2 * q2_3 - alpha * term2 * p2_3) / omega

        # k4
        q1_4, p1_4 = q1 + k3_q1, p1 + k3_p1
        q2_4, p2_4 = q2 + k3_q2, p2 + k3_p2
        k4_q1 = dt * (-alpha * kappa2 * q1_4 + term4 * p1_4) / omega
        k4_p1 = dt * (-kappa2 * q1_4 - alpha * term4 * p1_4) / omega
        k4_q2 = dt * (-alpha * kappa2 * q2_4 + term4 * p2_4) / omega
        k4_p2 = dt * (-kappa2 * q2_4 - alpha * term4 * p2_4) / omega

        q1 += (k1_q1 + 2.0 * k2_q1 + 2.0 * k3_q1 + k4_q1) / 6.0
        p1 += (k1_p1 + 2.0 * k2_p1 + 2.0 * k3_p1 + k4_p1) / 6.0
        q2 += (k1_q2 + 2.0 * k2_q2 + 2.0 * k3_q2 + k4_q2) / 6.0
        p2 += (k1_p2 + 2.0 * k2_p2 + 2.0 * k3_p2 + k4_p2) / 6.0

    # Monodromy Matrix M = [[q1, q2], [p1, p2]]
    # Eigenvalues via trace and determinant
    tr = q1 + p2
    det = q1 * p2 - q2 * p1
    disc = tr * tr - 4.0 * det

    if disc >= 0.0:
        root = np.sqrt(disc)
        e1 = abs((tr + root) * 0.5)
        e2 = abs((tr - root) * 0.5)
        return max(e1, e2)
    else:
        # Complex conjugate roots: |\mu| = \sqrt{det}
        return np.sqrt(max(det, 0.0))


def compute_floquet_stability_grid(h0_vals, kappa_vals, ha, alpha=0.27, omega=1.0,
                                   steps_per_cycle=400, verbose=False):
    r"""
    Computes a 2D matrix of Floquet multipliers |\mu_\kappa| across (kappa, h0) at fixed ha.
    Unsynchronized points are set to np.nan.
    Returns: mu_matrix of shape (len(kappa_vals), len(h0_vals))
    """
    n_k = len(kappa_vals)
    n_h = len(h0_vals)
    mu_grid = np.full((n_k, n_h), np.nan, dtype=np.float64)

    for j, h0 in enumerate(h0_vals):
        conv, psi_cycle = solve_background_cycle(
            h0, ha, alpha=alpha, omega=omega, steps_per_cycle=steps_per_cycle
        )
        if not conv:
            continue

        for i, k in enumerate(kappa_vals):
            mu_grid[i, j] = compute_floquet_multiplier(
                psi_cycle, k, alpha=alpha, omega=omega, steps_per_cycle=steps_per_cycle
            )

        if verbose and (j + 1) % 50 == 0:
            print(f"  h0 [{j+1}/{n_h}] processed (h0={h0:.2f})")

    return mu_grid

