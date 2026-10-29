# Floquet Linear Stability Analysis

This module implements linear Floquet stability theory for flexural spatial modes $\kappa$ around synchronized flat domain walls, reproducing Section IV and Figure 3 of the paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Physical Review B

---

## 1. Mathematical Formulation

Linearizing the full $u-\phi$ PDE around the flat synchronized rigid-wall solution $\mathbf{v}_1(\tau) = (q_1(\tau), \psi_1(\tau)/2)^T$ gives:

$$\partial_\tau \boldsymbol{\eta} = \frac{1}{2\Omega} \mathcal{M}(\tau) \boldsymbol{\eta}$$

where $\boldsymbol{\eta}(x, \tau) = (Q(x,\tau), \Phi(x,\tau))^T$. In spatial Fourier space, distinct spatial modes $\kappa$ completely decouple:

$$\partial_\tau \boldsymbol{\eta}_\kappa = \frac{1}{2\Omega} \mathcal{M}_\kappa(\tau) \boldsymbol{\eta}_\kappa$$

$$\mathcal{M}_\kappa(\tau) = -\kappa^2 \begin{pmatrix} \alpha & -1 \\ 1 & \alpha \end{pmatrix} + 2\cos\psi_1(\tau) \begin{pmatrix} 0 & 1 \\ 0 & -\alpha \end{pmatrix}$$

Expanding in components ($Q_\kappa, \Phi_\kappa$):
$$\dot{Q}_\kappa = \frac{1}{\Omega} \left[ -\alpha \frac{\kappa^2}{2} Q_\kappa + \left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right) \Phi_\kappa \right]$$
$$\dot{\Phi}_\kappa = \frac{1}{\Omega} \left[ -\frac{\kappa^2}{2} Q_\kappa - \alpha \left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right) \Phi_\kappa \right]$$

### The Monodromy Matrix & Floquet Multipliers
By Floquet's theorem, the fundamental solution matrix after one period $\tau \in [0, 2\pi]$ defines the $2\times 2$ real **Monodromy Matrix** $M_\kappa$:
- Column 1: integrated with initial condition $\boldsymbol{\eta}_\kappa(0) = (1, 0)^T$
- Column 2: integrated with initial condition $\boldsymbol{\eta}_\kappa(0) = (0, 1)^T$

Eigenvalues $\mu_{1,2}$ satisfy:
$$\lambda^2 - \operatorname{tr}(M_\kappa)\lambda + \det(M_\kappa) = 0$$

- **Stability Criterion**: The mode $\kappa$ is exponentially unstable if and only if $|\mu_\kappa| = \max(|\mu_1|, |\mu_2|) > 1$.
- **Growth Rate**: Continuous exponential growth rate $\lambda_\kappa = \frac{\Omega}{2\pi} \ln|\mu_\kappa|$.
- **Dominant Instability Mode**: $\kappa^* = \operatorname{argmax}_\kappa |\mu_\kappa| \approx 0.833$.

---

## 2. Module Directory Tour

```text
floquet/
├── include/
│   └── floquet.hpp                 # Header-only C++ Floquet monodromy integrator & eigenvalue solver
├── src/
│   ├── floquet_sweep_cpu.cpp       # CPU OpenMP tool: (h0, ha) sweeps and (h0, kappa) diagrams
│   └── floquet_sweep_gpu.cu        # CUDA GPU tool: High-throughput sweeps across millions of points
├── python/
│   ├── floquet_solver.py           # Python Floquet engine
│   ├── plot_figure3.py             # Reproduces publication Figure 3 (figure3.pdf)
│   └── plot_floquet_map.py         # Visualizer for 2D Floquet multiplier maps
└── README.md
```

---

## 3. Quickstart & Usage

### Reproducing Paper Figure 3
To regenerate Figure 3 (linear stability diagrams for $h_a=2$ and $h_a=3$):
```bash
python3 floquet/python/plot_figure3.py --out-dir figures/ --export-paper
```
Outputs:
- `figures/figure3.pdf` (and `figure3.png`)
- `paper/figuras/figure3.pdf`

### Running Fast CPU Mode Stability Slices
Generate $(h_0, \kappa)$ stability diagrams in sub-second execution on CPU:
```bash
# Slice at ha = 3.0
./build_local/bin/floquet_sweep_cpu --ha 3.0 --h0-min 0 --h0-max 8 --num-h0 300 --k-min 0.7 --k-max 1.2 --num-k 150 -o slice_ha3.txt

# Slice at ha = 2.0
./build_local/bin/floquet_sweep_cpu --ha 2.0 --h0-min 0 --h0-max 8 --num-h0 300 --k-min 0.7 --k-max 1.2 --num-k 150 -o slice_ha2.txt
```

### Running 2D Floquet Parameter Sweeps (CPU & GPU)
```bash
# CPU OpenMP sweep across (h0, ha) for system size L = 1024:
./build_local/bin/floquet_sweep_cpu -L 1024 --h0-min 0 --h0-max 10 --num-h0 1000 --ha-min 0 --ha-max 20 --num-ha 1000 -o output/floquet_L1024.bin

# GPU CUDA sweep (on GPU-enabled machine or cluster node):
./build/bin/floquet_sweep_gpu -L 1024 --h0-min 0 --h0-max 10 --num-h0 1000 --ha-min 0 --ha-max 20 --num-ha 1000 -o output/floquet_L1024.bin
```

