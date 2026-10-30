# Floquet Linear Stability Analysis

This module implements linear Floquet stability theory for flexural spatial modes $\kappa$ around synchronized flat domain walls, reproducing Section IV and Figure 3 of the paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Physical Review B (2026)

---

## 1. Mathematical Formulation

Linearizing the full $u-\phi$ PDE around the flat synchronized rigid-wall solution $\mathbf{v}_1(\tau) = (q_1(\tau), \psi_1(\tau)/2)^T$ gives decoupled equations for each spatial Fourier mode $\kappa$:

$$\dot{Q}_\kappa = \frac{1}{\Omega} \left[ -\alpha \frac{\kappa^2}{2} Q_\kappa + \left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right) \Phi_\kappa \right]$$

$$\dot{\Phi}_\kappa = \frac{1}{\Omega} \left[ -\frac{\kappa^2}{2} Q_\kappa - \alpha \left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right) \Phi_\kappa \right]$$

### The Monodromy Matrix & Multipliers
Integrating the system over one driving period $\tau \in [0, 2\pi]$ yields the $2\times 2$ real **Monodromy Matrix** $M_\kappa$.
- **Stability Criterion**: Mode $\kappa$ is exponentially unstable if its largest Floquet multiplier modulus satisfies:
  $$|\mu_\kappa| = \max(|\mu_1|, |\mu_2|) > 1$$
- **Most Unstable Mode**: $\kappa^* = \operatorname{argmax}_\kappa |\mu_\kappa| \approx 0.833$.

---

## 2. Quickstart & Usage

### Reproduce Paper Figure 3
Recreates PRB **Figure 3** (`figure3.pdf`), showing $(h_0, \kappa)$ stability diagrams for $h_a = 2.0$ and $h_a = 3.0$:
```bash
python3 reproduction/plot_figure3_floquet.py --export-paper
```
To recompute the stability grids on the fly:
```bash
python3 reproduction/plot_figure3_floquet.py --recompute --h0-pts 300 --kappa-pts 200
```

### Running C++ Stability Sweeps (CPU OpenMP)

#### 1. 1D Stability Slice across $(h_0, \kappa)$ at Fixed $h_a$ (Figure 3 Mode)
```bash
# Stability slice at ha = 3.0:
./build_local/bin/floquet_sweep_cpu --ha 3.0 --h0-min 0.0 --h0-max 8.0 --num-h0 300 \
                                   --k-min 0.7 --k-max 1.2 --num-k 150 \
                                   -o output/floquet_slice_ha3.txt

# Stability slice at ha = 2.0:
./build_local/bin/floquet_sweep_cpu --ha 2.0 --h0-min 0.0 --h0-max 8.0 --num-h0 300 \
                                   --k-min 0.7 --k-max 1.2 --num-k 150 \
                                   -o output/floquet_slice_ha2.txt
```

#### 2. Full 2D Parameter Sweep across $(h_0, h_a)$ (Figure 5 & 9 Phase Diagrams)
```bash
./build_local/bin/floquet_sweep_cpu -L 256 --h0-min 0.0 --h0-max 10.0 --num-h0 500 \
                                   --ha-min 0.0 --ha-max 20.0 --num-ha 500 \
                                   -o output/floquet_L256.bin
```

Options:
- `--ha <float>`: Fixes AC amplitude and evaluates mode stability across $(h_0, \kappa)$.
- `--h0-min <float>`, `--h0-max <float>`, `--num-h0 <int>`: DC field range and grid points.
- `--ha-min <float>`, `--ha-max <float>`, `--num-ha <int>`: AC amplitude range for 2D sweeps.
- `--k-min <float>`, `--k-max <float>`, `--num-k <int>`: Wavenumber range and modes to test.
- `-L <float>`: System length (determines discrete mode spacing $2\pi/L$).
- `-o`, `--output <filename>`: Output file path.

### Interactive Python Prototyping
Solve the monodromy matrix and test mode stability directly in Python:
```bash
python3 floquet/python/floquet_solver.py
```
