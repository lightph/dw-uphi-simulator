# Rigid Wall Analysis (0D Walker / Slonczewski Model)

This module provides analytical perturbation theory, CPU and GPU numerical solvers, and visualization tools for the zero-dimensional rigid domain wall (DW) dynamics described in Section III and Appendix C of the paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Physical Review B

---

## 1. Physical Equations of Motion

In the zero-dimensional rigid wall approximation (neglecting spatial flexural variations $\partial_x^2 \to 0$), the domain wall internal magnetization angle $\psi_1(t) = 2\phi(t)$ and translational position $q_1(t) = u(t)$ satisfy the coupled ODEs under perpendicular DC ($h_0$) and AC ($h_a\cos\Omega t$) magnetic fields:

$$\dot{\psi}_1 = \frac{\alpha}{\Omega} \left[ h_0 + h_a \cos(\tau) - \sin(\psi_1) \right]$$

$$\dot{q}_1 = \frac{\alpha^2 + 1}{2\Omega} \left[ h_0 + h_a \cos(\tau) \right] - \frac{1}{2\alpha} \dot{\psi}_1$$

where:
- $\tau = \Omega t$ is dimensionless scaled time.
- $\alpha = 0.27$ is the Gilbert damping parameter.
- $\Omega = 1.0$ is the applied driving frequency.

### Average Velocities
Averaged over steady-state limit cycles:
$$\langle \dot{q}_1 \rangle_\tau = \frac{\alpha^2 + 1}{2\Omega} h_0 - \frac{1}{2\alpha} \langle \dot{\psi}_1 \rangle_\tau$$

Within the $n$-th Shapiro synchronization step, frequency-locking occurs:
$$\langle \dot{\psi}_1 \rangle_\tau = n \quad (n = 0, 1, 2, \dots)$$

---

## 2. Analytical Perturbation Theory

### First-Order Perturbation Theory (Jacobi-Anger Expansion)
Treating $\epsilon = \alpha / \Omega$ as small parameter, the $n$-th Shapiro step is centered at:
$$h_{0, \text{center}}^{(1)} = \frac{n\Omega}{\alpha}$$
and has full width:
$$\Delta h_0^{(1)} = 2 \left| J_n\left(\frac{\alpha}{\Omega} h_a\right) \right|$$

Boundaries (vertical red dotted lines in Fig. 2):
$$h_0 \in \left[ \frac{n\Omega}{\alpha} - \left| J_n\left(\frac{\alpha}{\Omega} h_a\right) \right|, \, \frac{n\Omega}{\alpha} + \left| J_n\left(\frac{\alpha}{\Omega} h_a\right) \right| \right]$$

### Second-Order Perturbation Theory (Secular Frequency Shift)
Non-linear coupling between the first-order precessional oscillations and the AC drive induces a net DC torque, causing a rigid shift in the step center without altering its width:
$$h_{0, \text{center}}^{(2)} = \frac{n\Omega}{\alpha} + \frac{\alpha}{\Omega} A$$
where:
$$A = \frac{1}{2} \sum_{m \neq -n} \frac{J_m^2(\alpha h_a / \Omega)}{n + m}$$
For $n = 0$, $A = 0$ identically (the $n=0$ step experiences no second-order shift).

Boundaries (vertical green dashed lines in Fig. 2):
$$h_0 \in \left[ h_{0, \text{center}}^{(2)} - \left| J_n\left(\frac{\alpha}{\Omega} h_a\right) \right|, \, h_{0, \text{center}}^{(2)} + \left| J_n\left(\frac{\alpha}{\Omega} h_a\right) \right| \right]$$

---

## 3. Module Tour & Quickstart

```text
rigid_wall/
├── include/
│   └── rigid_wall.hpp              # Header-only C++ ODE stepper, Bessel quadrature & perturbation formulas
├── src/
│   ├── rigid_wall_sweep_cpu.cpp    # CPU OpenMP 1D/2D parameter sweep executable
│   └── rigid_wall_sweep_gpu.cu     # CUDA GPU 2D parameter sweep executable
├── python/
│   ├── rigid_wall.py               # Pure-NumPy ODE integrator & analytical formulas
│   ├── plot_figure2.py             # Reproduces publication Figure 2 (figure2.pdf)
│   └── plot_rigid_sync.py          # Visualizer for 2D synchrony maps
└── README.md
```

### Reproducing Paper Figure 2
To recreate Physical Review B **Figure 2** (`figure2.pdf`):
```bash
python3 rigid_wall/python/plot_figure2.py --out-dir figures/ --export-paper
```
Outputs:
- `figures/figure2.pdf` (and `figure2.png`)
- `paper/figuras/figure2.pdf`

### Running the CPU OpenMP Parameter Sweep
```bash
./build_local/bin/rigid_wall_sweep_cpu --h0-min 0 --h0-max 10 --num-h0 500 --ha 5.0 -o rigid_wall_ha5.txt
```

### Running the GPU Parameter Sweep (CUDA)
```bash
./build/bin/rigid_wall_sweep_gpu --h0-min 0 --h0-max 10 --num-h0 1000 --ha-min 0 --ha-max 20 --num-ha 1000 -o sync_results.bin
```

