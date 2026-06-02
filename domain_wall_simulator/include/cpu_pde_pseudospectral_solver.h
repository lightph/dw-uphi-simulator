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
};

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
    N_ptr[i] = nonlinear_part(z_ptr[i], t);
  }

  fftw_handler.do_fft(N_buf, Nhat_buf);

  COMPLEX *zhat_ptr = zhat_buf.data();
  const COMPLEX *Nhat_ptr = Nhat_buf.data();
  const COMPLEX *d_ptr = denom_precomp.data();
  const COMPLEX *f_ptr = factor_precomp.data();

#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    zhat_ptr[i] = d_ptr[i] * (f_ptr[i] * zhat_ptr[i] + dt * Nhat_ptr[i]);
  }

  fftw_handler.do_ifft(zhat_buf, z_buf);

  REAL inv_N = static_cast<REAL>(1.0) / static_cast<REAL>(n_modes);
#pragma omp parallel for simd if (n_modes > 10000)
  for (size_t i = 0; i < n_modes; ++i)
  {
    z_ptr[i] *= inv_N;
  }

  t += dt;
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
