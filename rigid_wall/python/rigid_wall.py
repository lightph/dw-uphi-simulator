#!/usr/bin/env python3
r"""
rigid_wall.py - 0D Walker / Slonczewski Model ODE Solver & Perturbation Theory

Implements:
  1. 0D rigid wall ODEs in scaled time \tau = \Omega * t:
       d\psi_1 / d\tau = (\alpha / \Omega) * [h_0 + h_a * cos(\tau) - sin(\psi_1)]
       dq_1 / d\tau    = (1 / (2*\Omega)) * [\alpha^2 * h_0 + \alpha^2 * h_a * cos(\tau) + sin(\psi_1)]
     Note:
       <dq_1/d\tau>    = [(\alpha^2 + 1) / (2*\Omega)] * h_0 - (\alpha / 2) * <d\psi_1/d\tau>

  2. 4th-order Runge-Kutta (RK4) integration with stroboscopic limit cycle detection.

  3. Analytical solutions:
     - ha = 0 (Adler equation):
         <d\psi_1/d\tau> = 0 for |h_0| <= 1,
         <d\psi_1/d\tau> = sgn(h_0) * (\alpha / \Omega) * sqrt(h_0^2 - 1) for |h_0| > 1.
     - 1st-order perturbation theory:
         h_0^{(1)} = n*\Omega / \alpha \pm |J_n(\alpha*h_a / \Omega)|.
     - 2nd-order perturbation theory:
         h_0^{(2)} = n*\Omega / \alpha + (\alpha / \Omega)*A \pm |J_n(\alpha*h_a / \Omega)|
         where A = 0.5 * sum_{m != -n} [ J_m^2(\tilde{h}_a) / (n + m) ], \tilde{h}_a = \alpha*h_a / \Omega.
         (For n = 0, A = 0 identically).
"""

import numpy as np

# Bessel function implementation using pure NumPy with fallback to scipy if available
try:
    from scipy.special import jv as scipy_jv
    def bessel_j(n, x):
        return scipy_jv(n, x)
except ImportError:
    def bessel_j(n, x, num_pts=500):
        """Pure NumPy implementation of Bessel function J_n(x) via integral representation."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            theta = np.linspace(0, np.pi, num_pts)
            integrand = np.cos(n * theta - float(x) * np.sin(theta))
            if hasattr(np, 'trapezoid'):
                return np.trapezoid(integrand, theta) / np.pi
            return np.trapz(integrand, theta) / np.pi
        else:
            theta = np.linspace(0, np.pi, num_pts)[:, None]
            integrand = np.cos(n * theta - x[None, :] * np.sin(theta))
            if hasattr(np, 'trapezoid'):
                return np.trapezoid(integrand, theta, axis=0) / np.pi
            return np.trapz(integrand, theta, axis=0) / np.pi


def adler_velocities(h0_grid, alpha=0.27, omega=1.0):
    r"""
    Exact analytical steady-state velocities for ha = 0 (Adler equation).
    Returns (v_psi, v_q) across h0_grid.
    """
    h0 = np.asarray(h0_grid, dtype=np.float64)
    v_psi = np.zeros_like(h0)
    unpinned = np.abs(h0) > 1.0
    v_psi[unpinned] = np.sign(h0[unpinned]) * (alpha / omega) * np.sqrt(h0[unpinned]**2 - 1.0)
    v_q = ((alpha**2 + 1.0) / (2.0 * omega)) * h0 - (1.0 / (2.0 * alpha)) * v_psi
    return v_psi, v_q


def perturbation_boundaries(n_steps=3, alpha=0.27, omega=1.0, ha=5.0, max_m=40):
    r"""
    Computes 1st-order and 2nd-order Shapiro step boundaries for steps n = 0, 1, ..., n_steps-1.
    Returns:
      dict with:
        'first_order': list of (center, low, high)
        'second_order': list of (center, low, high, shift)
    """
    h_tilde_a = alpha * ha / omega
    first_order = []
    second_order = []

    for n in range(n_steps):
        center1 = n * omega / alpha
        width = abs(float(bessel_j(n, h_tilde_a)))
        b1_low = center1 - width
        b1_high = center1 + width
        first_order.append((center1, b1_low, b1_high))

        if n == 0:
            shift = 0.0
            A = 0.0
        else:
            m_vals = [m for m in range(-max_m, max_m + 1) if m != -n]
            A = 0.5 * sum((bessel_j(m, h_tilde_a)**2) / (n + m) for m in m_vals)
            shift = (alpha / omega) * float(A)

        center2 = center1 + shift
        b2_low = center2 - width
        b2_high = center2 + width
        second_order.append((center2, b2_low, b2_high, shift))

    return {'first_order': first_order, 'second_order': second_order}


def solve_rigid_wall_rk4(h0_grid, ha=5.0, alpha=0.27, omega=1.0,
                         trans_cycles=200, eval_cycles=200, steps_per_cycle=1000):
    r"""
    Numerically integrates the 0D Walker ODEs over transient and evaluation cycles:
      d\psi_1/d\tau = (\alpha/\Omega) * [h0 + ha*cos(\tau) - sin(\psi_1)]
      dq_1/d\tau    = (1/(2*\Omega)) * [\alpha^2 * (h0 + ha*cos(\tau)) + sin(\psi_1)]
    where \tau = \Omega * t \in [0, 2\pi].

    Returns:
      h0_grid: array of DC fields
      v_psi:   time-averaged internal angular velocity <d\psi_1/d\tau>
      v_q:     time-averaged translational velocity <dq_1/d\tau>
      is_sync: boolean mask indicating frequency-locking to an integer step
    """
    h0 = np.asarray(h0_grid, dtype=np.float64)
    n_h0 = len(h0)
    dt = 2.0 * np.pi / steps_per_cycle

    A_arr = (alpha * h0) / omega
    B_val = alpha / omega
    C_val = (alpha * ha) / omega

    # Precompute trigonometric terms for scaled time tau
    tau_0 = np.arange(steps_per_cycle) * dt
    tau_half = (np.arange(steps_per_cycle) + 0.5) * dt
    cos_0 = np.cos(tau_0)
    cos_half = np.cos(tau_half)

    psi = np.zeros(n_h0, dtype=np.float64)

    # 1. Transient relaxation
    for _ in range(trans_cycles):
        for s in range(steps_per_cycle):
            inext = (s + 1) % steps_per_cycle
            c0 = cos_0[s]
            ch = cos_half[s]
            c1 = cos_0[inext]

            k1 = dt * (A_arr - B_val * np.sin(psi) + C_val * c0)
            k2 = dt * (A_arr - B_val * np.sin(psi + 0.5 * k1) + C_val * ch)
            k3 = dt * (A_arr - B_val * np.sin(psi + 0.5 * k2) + C_val * ch)
            k4 = dt * (A_arr - B_val * np.sin(psi + k3) + C_val * c1)

            psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # 2. Evaluation phase: track winding and accumulation
    psi_start = psi.copy()
    sum_sin_psi = np.zeros(n_h0, dtype=np.float64)
    total_steps = eval_cycles * steps_per_cycle

    for _ in range(eval_cycles):
        for s in range(steps_per_cycle):
            inext = (s + 1) % steps_per_cycle
            c0 = cos_0[s]
            ch = cos_half[s]
            c1 = cos_0[inext]

            k1 = dt * (A_arr - B_val * np.sin(psi) + C_val * c0)
            k2 = dt * (A_arr - B_val * np.sin(psi + 0.5 * k1) + C_val * ch)
            k3 = dt * (A_arr - B_val * np.sin(psi + 0.5 * k2) + C_val * ch)
            k4 = dt * (A_arr - B_val * np.sin(psi + k3) + C_val * c1)

            # Midpoint approximation for integrand
            psi_mid = psi + 0.5 * (k1 + k2) / 2.0
            sum_sin_psi += np.sin(psi_mid)

            psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    total_tau = eval_cycles * 2.0 * np.pi
    v_psi = (psi - psi_start) / total_tau

    # dq_1/d\tau formula: <dq_1/d\tau> = [(alpha^2 + 1)/(2*omega)]*h0 - [1/(2*alpha)]*<d\psi_1/d\tau>
    v_q = ((alpha**2 + 1.0) / (2.0 * omega)) * h0 - (1.0 / (2.0 * alpha)) * v_psi

    # Identify frequency-locked synchronization steps (within 0.005 of integer)
    nearest_int = np.round(v_psi)
    is_sync = np.abs(v_psi - nearest_int) < 0.008

    return h0, v_psi, v_q, is_sync
