# Full PDE Simulation Engine (1D Slonczewski Model)

This module contains the core high-performance batched pseudospectral simulation framework for the 1D Slonczewski ($u-\phi$) partial differential equations describing driven magnetic domain walls in perpendicular magnetic anisotropy (PMA) thin films.

---

## 1. Physical Equations & Numerical Method

In dimensionless units, the coupled equations for interface position $u(x,t)$ and internal magnetization angle $\phi(x,t)$ are:

$$\dot{u} = \frac{1}{2} \left[ \alpha^2 h(t) + \sin(2\phi) + \alpha \partial_x^2 u - \partial_x^2 \phi \right]$$

$$\dot{\phi} = \frac{1}{2} \left[ \alpha h(t) - \alpha \sin(2\phi) + \partial_x^2 u + \alpha \partial_x^2 \phi \right]$$

where:
- $\alpha = 0.27$ is the Gilbert damping.
- $h(t) = h_0 + h_a \cos(\Omega t)$ is the applied external magnetic field.

Introducing the complex coordinate $z(x,t) = u(x,t) - i\phi(x,t)$, the system maps compactly to:

$$\dot{z} = \frac{\alpha - i}{2} \left[ \alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right) + \partial_x^2 z \right]$$

### Pseudospectral Crank-Nicolson Integration
- **Linear Term**: $\mathcal{L}(z) = \frac{\alpha - i}{2}\partial_x^2 z$ is integrated in Fourier space via the unconditionally stable **Crank-Nicolson** scheme.
- **Non-Linear Term**: $\mathcal{N}(z) = \frac{\alpha - i}{2} \left[\alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right)\right]$ is stepped explicitly in real space.
- All FFT transformations are batched and executed in double precision using **cuFFT** (GPU) or **FFTW3** (CPU).

---

## 2. Dual-Backend Architecture (GPU & CPU)

Every executable supports both execution backends via a trailing `[backend]` parameter (`gpu` or `cpu`):

1. **GPU Backend (`GpuBackend`)**:
   - CUDA C++ with custom device kernels.
   - Paged-locked pinned memory transfers and asynchronous CUDA streams.
   - Massive batching of concurrent domain walls across SMs.
2. **CPU Backend (`CpuBackend`)**:
   - Standard C++17 with OpenMP multithreading.
   - FFTW3 plan caching and SIMD aligned memory allocations.
   - Zero CUDA dependencies required when building with `-DUSE_CUDA=OFF`.

---

## 3. Simulation Executables Reference

All binaries are located in `build/bin/` (or `build_local/bin/`).

### 1. `main` (Standard Time Evolution & Observables)
Simulates time evolution and writes transient metrics, time series, and spatial height histograms.
```bash
./bin/main <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
           <transient_time> <averaging_time> <window_time> <subsample_rate> \
           <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 2. `time_evo` (Logarithmic Sampling & Surface Growth)
Samples system properties at powers-of-two steps ($t = 2^p \times \delta t$), tracking Edwards-Wilkinson surface growth across decades (used for Figure 6).
```bash
./bin/time_evo <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> \
               <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 3. `sweep` (1D / 2D Parameter Sweeps & Order Parameters)
Performs sweeps over $h_a$ at fixed $h_0$, computing steady-state velocities $\langle\dot{u}\rangle$, $\langle\dot{\phi}\rangle$, roughness $\langle w^2 \rangle_t$, and stroboscopic variance $\langle \sigma^2 \rangle_x$ (used for Figures 4, 5, 9).
```bash
./bin/sweep <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> \
             <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend=gpu]
```

### 4. `snapshots` (Real-Space Profile Snapshots)
Outputs real-space configurations $u(x,t)$ and $\phi(x,t)$ at specified intervals (used for Figure 7).
```bash
./bin/snapshots <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                <transient_time> <snapshot_time> <window_time> <snapshot_rate> \
                <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 5. `moments` (Higher-Order Moments & Cumulants)
Computes central moments ($\mu_2, \mu_3, \mu_4$), skewness, and kurtosis of interface fluctuations.
```bash
./bin/moments <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <total_time> \
              <start_time> <num_checkpoints> <num_sq_checkpoints> <output_prefix> \
              [eps=0.05] [seed=42] [backend=gpu]
```

### 6. `movies` (High-Framerate Binary Animations)
Streams binary state snapshots to disk for video animation.
```bash
./bin/movies <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <t_max> \
             <window_time> <movie_start> <movie_end> <movie_rate> \
             <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 7. `sat_check` (Roughness Saturation Utility)
Tests interface growth until roughness reaches steady-state saturation.
```bash
./bin/sat_check <size> <L> <alpha> <dt> [backend=gpu]
```

