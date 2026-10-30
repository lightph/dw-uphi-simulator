# 1D Pseudospectral PDE Simulation Engine

This module contains the batched, pseudospectral Crank-Nicolson simulation framework for the 1D Slonczewski ($u-\phi$) partial differential equations describing driven magnetic domain walls in PMA thin films.

---

## 1. Physical Equations & Numerical Integration

In dimensionless units, the coupled equations for domain wall position $u(x,t)$ and magnetization angle $\phi(x,t)$ under magnetic field $h(t) = h_0 + h_a \cos(\Omega t)$ are:

$$\dot{u} = \frac{1}{2} \left[ \alpha^2 h(t) + \sin(2\phi) + \alpha \partial_x^2 u - \partial_x^2 \phi \right]$$

$$\dot{\phi} = \frac{1}{2} \left[ \alpha h(t) - \alpha \sin(2\phi) + \partial_x^2 u + \alpha \partial_x^2 \phi \right]$$

Introducing the complex coordinate $z(x,t) = u(x,t) - i\phi(x,t)$:

$$\dot{z} = \frac{\alpha - i}{2} \left[ \alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right) + \partial_x^2 z \right]$$

- **Linear Term**: $\mathcal{L}(z) = \frac{\alpha - i}{2}\partial_x^2 z$ is integrated in Fourier space via the unconditionally stable **Crank-Nicolson** method.
- **Non-Linear Term**: Stepped explicitly in real space using double precision.
- **Backends**: Runs seamlessly on **CPU** (OpenMP + FFTW3) or **GPU** (CUDA + cuFFT).

---

## 2. Compilation

Executables are compiled from the repository root:
```bash
# CPU build (OpenMP + FFTW3):
./compile_local.sh
# Binaries placed in build_local/bin/

# GPU build (CUDA + cuFFT):
./compile_cluster.sh
# Binaries placed in build/bin/
```

---

## 3. Parameter Reference

All PDE binaries share standard physical and numerical arguments:

| Argument | Description | Paper Standard | Notes |
|:---|:---|:---:|:---|
| `<size>` / `N` | Number of grid points | $2 \times L$ | Grid spacing $\delta x = L/N = 0.5$ |
| `<L>` | Domain length | `8.0`, `256.0`, `1024.0` | In units of $L_0 \approx 58\text{ nm}$ |
| `<alpha>` | Gilbert damping | `0.27` | Typical for Co/Pt multilayers |
| `<h0>` | DC magnetic field | `0.0 – 10.0` | In units of Walker breakdown field $H_W$ |
| `<ha>` | AC magnetic field amplitude | `0.0 – 20.0` | In units of Walker breakdown field $H_W$ |
| `<omega>` | AC driving frequency | `1.0` | $\approx 480\text{ MHz}$ |
| `<dt>` | Time step | `0.019635` | $2\pi / (320\,\Omega)$ (320 steps/period) |
| `[eps]` | Initial noise amplitude | `0.0005` | Random perturbation on flat initial state |
| `[seed]` | Random seed | `42` | For reproducible initial conditions |
| `[backend]` | Compute backend | `cpu` or `gpu` | Selects CPU (OpenMP) or GPU (CUDA) |

---

## 4. Executables & Usage Examples

### 1. `main` (Time Evolution & Observables)
Calculates steady-state velocities $\langle\dot u\rangle$, $\langle\dot\phi\rangle$, interface roughness $w^2(t)$, and spatial metrics.

```bash
./build_local/bin/main <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                       <transient_time> <averaging_time> <window_time> <subsample_rate> \
                       <output_prefix> [eps=0.0005] [seed=42] [backend=cpu]
```

*Example*:
```bash
# Simulate L=256 on CPU with h0=4.0, ha=3.0:
./build_local/bin/main 512 1 256.0 0.27 4.0 3.0 1.0 0.019635 \
                       2000.0 1000.0 100.0 10 \
                       output/main_test 0.0005 42 cpu
```
**Outputs**:
- `<output_prefix>_transient_metrics.txt`: Time, roughness, $\dot u$, $\dot\phi$, and entropy vs time.
- `<output_prefix>_time_series.txt`: Detailed time-series measurements.

---

### 2. `sweep` (Parameter Sweeps & Phase Order Parameters)
Sweeps across AC fields $h_a$ at fixed $h_0$ to measure steady-state roughness $\langle w^2\rangle_t$ and stroboscopic variance (used for Figures 4, 5, 9).

```bash
./build_local/bin/sweep <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> \
                        <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> \
                        [backend=cpu] [eps=0.0005] [seed=42]
```

*Example*:
```bash
# Sweep ha in [0, 6] with 61 steps at h0 = 4.0:
./build_local/bin/sweep 512 256.0 0.27 1.0 0.019635 2000.0 1000.0 \
                        4.0 0.0 6.0 61 \
                        output/sweep_h0_4.txt cpu 0.0005 42
```
**Output**: Tabulated columns of $h_0$, $h_a$, $\langle\dot u\rangle$, $\langle\dot\phi\rangle$, $\langle w^2\rangle_t$, and stroboscopic variances.

---

### 3. `time_evo` (Logarithmic Surface Roughening)
Evaluates dynamic scaling and structure factor $S(k,t)$ at logarithmic times $t = 2^p \delta t$ (used for Figure 6, Edwards-Wilkinson collapse).

```bash
./build_local/bin/time_evo <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> \
                           <output_prefix> [eps=0.0005] [seed=42] [backend=cpu]
```

*Example*:
```bash
# Sample powers of 2 up to 2^24 steps for L=1024:
./build_local/bin/time_evo 2048 1024.0 0.27 5.0 3.0 1.0 0.019635 24 \
                           output/time_evo_test 0.0005 42 cpu
```

---

### 4. `snapshots` (Real-Space Configurations)
Records full spatial profiles $u(x,t)$ and $\phi(x,t)$ across driving periods (used for Figure 7 cascade).

```bash
./build_local/bin/snapshots <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                            <transient_time> <snapshot_time> <window_time> <snapshot_rate> \
                            <output_prefix> [eps=0.0005] [seed=42] [backend=cpu]
```

---

### 5. Other Utilities
- `moments`: Computes central moments ($\mu_2, \mu_3, \mu_4$), skewness, and kurtosis.
- `movies`: Streams binary profile frames to disk for high-framerate animations.
- `sat_check`: Tests crossover time until roughness reaches finite-size saturation.
