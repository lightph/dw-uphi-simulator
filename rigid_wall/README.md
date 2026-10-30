# Rigid Wall Analysis (0D Walker / Slonczewski Model)

This module provides analytical perturbation theory, CPU and GPU numerical solvers, and visualization tools for the zero-dimensional rigid domain wall (DW) dynamics described in Section III and Appendix C of the paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Physical Review B (2026)

---

## 1. Physical Equations of Motion

In the rigid wall approximation (neglecting spatial curvature $\partial_x^2 \to 0$), the domain wall internal angle $\psi_1(t) = 2\phi(t)$ and position $q_1(t) = u(t)$ satisfy the coupled ODEs under magnetic field $h(t) = h_0 + h_a \cos(\tau)$:

$$\dot{\psi}_1 = \frac{\alpha}{\Omega} \left[ h_0 + h_a \cos(\tau) - \sin(\psi_1) \right]$$

$$\dot{q}_1 = \frac{\alpha^2 + 1}{2\Omega} \left[ h_0 + h_a \cos(\tau) \right] - \frac{1}{2\alpha} \dot{\psi}_1$$

where $\tau = \Omega t$ is rescaled time, $\alpha = 0.27$ is the Gilbert damping, and $\Omega = 1.0$ is the driving frequency.

### Average Velocities
Averaged over steady-state limit cycles:
$$\langle \dot{q}_1 \rangle_\tau = \frac{\alpha^2 + 1}{2\Omega} h_0 - \frac{1}{2\alpha} \langle \dot{\psi}_1 \rangle_\tau$$

Within the $n$-th Shapiro synchronization step, frequency-locking occurs:
$$\langle \dot{\psi}_1 \rangle_\tau = n \quad (n = 0, 1, 2, \dots)$$

---

## 2. Analytical Perturbation Theory

- **First-Order Perturbation Theory (Jacobi-Anger Expansion)**:
  The $n$-th Shapiro step is centered at $h_{0, \text{center}}^{(1)} = n\Omega / \alpha$ with full width $\Delta h_0^{(1)} = 2 |J_n(\alpha h_a / \Omega)|$.
- **Second-Order Secular Frequency Shift**:
  Non-linear coupling induces a net DC torque, causing a rigid shift of the step center:
  $$h_{0, \text{center}}^{(2)} = \frac{n\Omega}{\alpha} + \frac{\alpha}{2\Omega} \sum_{m \neq -n} \frac{J_m^2(\alpha h_a / \Omega)}{n + m}$$

---

## 3. Quickstart & Usage

### Reproduce Paper Figure 2
Recreates PRB **Figure 2** (`figure2.pdf`), comparing RK4 numerical integration with 1st- and 2nd-order perturbation theory:
```bash
python3 reproduction/plot_figure2_rigid_wall.py --export-paper
```

### Running C++ Parameter Sweeps (CPU OpenMP)
Sweep $h_0$ at fixed $h_a = 5.0$ with 500 points:
```bash
./build_local/bin/rigid_wall_sweep_cpu --h0-min 0.0 --h0-max 10.0 --num-h0 500 --ha 5.0 -o output/rigid_ha5.txt
```

Options:
- `--ha <float>`: Fixed AC amplitude (runs 1D sweep across $h_0$).
- `--h0-min <float>`, `--h0-max <float>`, `--num-h0 <int>`: DC field range and grid points.
- `--ha-min <float>`, `--ha-max <float>`, `--num-ha <int>`: AC amplitude range for 2D sweeps.
- `--alpha <float>`: Damping (default: 0.27).
- `--omega <float>`: Driving frequency (default: 1.0).
- `-o`, `--output <filename>`: Output file path.

### Interactive Python Prototyping
Integrate the ODE and evaluate perturbation boundaries directly in Python without compiling C++:
```bash
python3 rigid_wall/python/rigid_wall.py
```
