#pragma once

#include "FftwHandler.h"
#include <vector>
#include <algorithm>
#include <cmath>

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
class cpu_pde_pseudospectral_stepper
{
private:
  using COMPLEX = FFTW_COMPLEX_STD;
  using REAL = FFTW_REAL;

public:
  cpu_pde_pseudospectral_stepper(size_t n_modes, REAL total_length, REAL dtt,
                                 const NonlinearPart &nonlinear_part,
                                 const InitialCondition &initial_condition,
                                 const LinearPart &linear_part);

  void set_nonlinear_part(const NonlinearPart &new_nonlin)
  {
    nonlinear_part = new_nonlin;
  }

  void set_linear_part(const LinearPart &new_lin)
  {
    linear_part = new_lin;
    update_precomputations();
  }

  void step();
  void run_block(size_t steps);

  const AlignedComplexVector &get_z() const;
  const AlignedComplexVector &get_zhat() const;

  void update_precomputations();
  void set_dt(REAL dtt);

  REAL get_time() const { return t; }
  void set_time(REAL new_t)
  {
    t = new_t;
  }

  void initialize();
  void reset(InitialCondition new_ic);

  // =========================================================================
  // HIGH-EFFICIENCY LOGGING & SPECTRAL ACCUMULATION
  // =========================================================================
  void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1);
  std::vector<REAL> get_and_clear_fine_phi_dot_history();
  std::vector<REAL> get_and_clear_fine_rugosity_history();

  void accumulate_u_power_spectrum();
  std::vector<REAL> get_averaged_u_power_spectrum() const;
  void reset_spectrum_accumulator();

private:
  REAL t = static_cast<REAL>(0.0);
  NonlinearPart nonlinear_part;
  InitialCondition initial_condition;
  LinearPart linear_part;

  size_t n_modes;
  REAL total_length;
  REAL dt;

  std::vector<REAL> k_vals;
  AlignedComplexVector denom_precomp;
  AlignedComplexVector factor_precomp;
  AlignedComplexVector z_buf;
  AlignedComplexVector N_buf;
  AlignedComplexVector Nhat_buf;
  AlignedComplexVector zhat_buf;

  FftwHandlerC2COut fftw_handler;

  // Tracking and Spectral Accumulation State
  bool fine_tracking_enabled = false;
  size_t tracking_sample_rate = 1;
  unsigned int internal_step_counter = 0;

  std::vector<REAL> d_phi_dot_history;
  std::vector<REAL> d_rugosity_history;

  std::vector<REAL> d_u_power_spectrum;
  unsigned int spectrum_accumulations = 0;
};

// ==========================================================
// Implementation
// ==========================================================

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition, LinearPart>::
    cpu_pde_pseudospectral_stepper(size_t modes, REAL length, REAL dtt,
                                   const NonlinearPart &nonlin,
                                   const InitialCondition &init_cond,
                                   const LinearPart &lin)
    : nonlinear_part(nonlin), initial_condition(init_cond), linear_part(lin),
      n_modes(modes), total_length(length), dt(dtt), z_buf(modes), N_buf(modes),
      Nhat_buf(modes), zhat_buf(modes)
{
  k_vals.resize(n_modes);
  denom_precomp.resize(n_modes);
  factor_precomp.resize(n_modes);
  fftw_handler.prepare(n_modes);

  REAL delta_k = static_cast<REAL>(2.0 * M_PI) / total_length;

  COMPLEX one(static_cast<REAL>(1.0), static_cast<REAL>(0.0));
  COMPLEX half(static_cast<REAL>(0.5), static_cast<REAL>(0.0));

  for (size_t i = 0; i < n_modes; ++i)
  {
    k_vals[i] =
        (i <= n_modes / 2)
            ? static_cast<REAL>(i) * delta_k
            : (static_cast<REAL>(i) - static_cast<REAL>(n_modes)) * delta_k;

    denom_precomp[i] = one / (one - half * dt * linear_part(k_vals[i]));
    factor_precomp[i] = one + half * dt * linear_part(k_vals[i]);
  }
  initialize();
}

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::initialize()
{
  COMPLEX *z_ptr = z_buf.data();
  REAL dx = total_length / static_cast<REAL>(n_modes);
  for (size_t i = 0; i < n_modes; ++i)
  {
    REAL x = static_cast<REAL>(i) * dx;
    z_ptr[i] = initial_condition(x);
  }

  // Unnormalized Forward FFT (Factor of 1)
  fftw_handler.do_fft(z_buf, zhat_buf);
}

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::reset(I new_ic)
{
  initial_condition = new_ic;
  t = static_cast<REAL>(0.0);
  initialize();
}

template <typename NonlinearPart, typename InitialCondition, typename LinearPart>
void cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition, LinearPart>::step()
{
  COMPLEX *z_ptr = z_buf.data();
  COMPLEX *N_ptr = N_buf.data();

#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    // 1. Evaluate nonlinear term on the correctly scaled physical z
    N_ptr[i] = nonlinear_part(z_ptr[i], t);
  }

  // 2. Unnormalized Forward FFT (Factor of 1)
  fftw_handler.do_fft(N_buf, Nhat_buf);

  COMPLEX *zhat_ptr = zhat_buf.data();
  const COMPLEX *Nhat_ptr = Nhat_buf.data();
  const COMPLEX *d_ptr = denom_precomp.data();
  const COMPLEX *f_ptr = factor_precomp.data();

#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    // 3. Fused frequency step update
    zhat_ptr[i] = d_ptr[i] * (f_ptr[i] * zhat_ptr[i] + dt * Nhat_ptr[i]);
  }

  // 4. Inverse FFT (Introduces a scaling factor of N)
  fftw_handler.do_ifft(zhat_buf, z_buf);

  // 5. Scale back to physical units to match GPU logic
  REAL inv_N = static_cast<REAL>(1.0) / static_cast<REAL>(n_modes);
#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    z_ptr[i] *= inv_N;
  }

  t += dt;

  // --- Metrics Recording Hook ---
  if (fine_tracking_enabled && (internal_step_counter % tracking_sample_rate == 0))
  {
    // Compute mean_phidot and rugosity here and push them to d_phi_dot_history and d_rugosity_history.
    // e.g. d_phi_dot_history.push_back(calculated_mean_phidot);
    // e.g. d_rugosity_history.push_back(calculated_rugosity);
  }
  internal_step_counter++;
}

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
const AlignedComplexVector &
cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition, LinearPart>::get_z()
    const
{
  return z_buf;
}

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
const AlignedComplexVector &cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition,
                                                           LinearPart>::get_zhat() const
{
  return zhat_buf;
}

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
void cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition,
                                    LinearPart>::update_precomputations()
{
  COMPLEX one(static_cast<REAL>(1.0), static_cast<REAL>(0.0));
  COMPLEX half(static_cast<REAL>(0.5), static_cast<REAL>(0.0));

  for (size_t i = 0; i < n_modes; ++i)
  {
    denom_precomp[i] = one / (one - half * dt * linear_part(k_vals[i]));
    factor_precomp[i] = one + half * dt * linear_part(k_vals[i]);
  }
}

template <typename NonlinearPart, typename InitialCondition,
          typename LinearPart>
void cpu_pde_pseudospectral_stepper<NonlinearPart, InitialCondition,
                                    LinearPart>::set_dt(REAL dtt)
{
  dt = dtt;
  update_precomputations();
}

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::run_block(size_t steps)
{
  for (size_t i = 0; i < steps; ++i)
  {
    step();
  }
}

// =========================================================================
// HIGH-EFFICIENCY LOGGING & SPECTRAL ACCUMULATION IMPLEMENTATIONS
// =========================================================================

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::set_fine_tracking(bool enable, size_t sample_every_n_steps)
{
  fine_tracking_enabled = enable;
  tracking_sample_rate = sample_every_n_steps;
  if (!enable)
  {
    d_phi_dot_history.clear();
    d_rugosity_history.clear();
  }
}

template <typename N, typename I, typename L>
std::vector<FFTW_REAL> cpu_pde_pseudospectral_stepper<N, I, L>::get_and_clear_fine_phi_dot_history()
{
  std::vector<REAL> h = std::move(d_phi_dot_history);
  d_phi_dot_history.clear();
  return h;
}

template <typename N, typename I, typename L>
std::vector<FFTW_REAL> cpu_pde_pseudospectral_stepper<N, I, L>::get_and_clear_fine_rugosity_history()
{
  std::vector<REAL> h = std::move(d_rugosity_history);
  d_rugosity_history.clear();
  return h;
}

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::accumulate_u_power_spectrum()
{
  if (d_u_power_spectrum.size() != n_modes)
  {
    d_u_power_spectrum.assign(n_modes, 0.0);
    spectrum_accumulations = 0;
  }

  const COMPLEX *zhat_ptr = zhat_buf.data();

#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    COMPLEX val = zhat_ptr[i];
    d_u_power_spectrum[i] += (val.real() * val.real() + val.imag() * val.imag());
  }

  spectrum_accumulations++;
}

template <typename N, typename I, typename L>
std::vector<FFTW_REAL> cpu_pde_pseudospectral_stepper<N, I, L>::get_averaged_u_power_spectrum() const
{
  std::vector<REAL> host_spec = d_u_power_spectrum;
  if (spectrum_accumulations > 0)
  {
    for (auto &val : host_spec)
    {
      val /= static_cast<REAL>(spectrum_accumulations);
    }
  }
  return host_spec;
}

template <typename N, typename I, typename L>
void cpu_pde_pseudospectral_stepper<N, I, L>::reset_spectrum_accumulator()
{
  std::fill(d_u_power_spectrum.begin(), d_u_power_spectrum.end(), static_cast<REAL>(0.0));
  spectrum_accumulations = 0;
}