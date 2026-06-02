#pragma once
#include "FftwHandler.h"
#include "cpu_pde_pseudospectral_solver.h"
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <algorithm>

class cpu_domain_wall_base
{
public:
  virtual ~cpu_domain_wall_base() = default;

  virtual void step() = 0;

  virtual void create_step_graph(size_t steps) = 0;
  virtual void run_block(size_t steps) = 0;

  virtual FFTW_REAL get_time() const = 0;
  virtual FFTW_REAL get_dt() const = 0;
  virtual FFTW_REAL get_Omega() const = 0;
  virtual const AlignedComplexVector &get_z() const = 0;
  virtual const AlignedComplexVector &get_zhat() const = 0;

  virtual FFTW_REAL get_h() const = 0;
  virtual FFTW_REAL get_h0() const = 0;
  virtual FFTW_REAL get_alpha() const = 0;
  virtual FFTW_REAL get_system_size() const = 0;
  virtual size_t get_n_samples() const = 0;

  virtual void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) = 0;
  virtual std::vector<FFTW_REAL> get_and_clear_fine_phi_dot_history() = 0;
  virtual std::vector<FFTW_REAL> get_and_clear_fine_rugosity_history() = 0;
  virtual void accumulate_u_power_spectrum() = 0;
  virtual std::vector<FFTW_REAL> get_averaged_u_power_spectrum() const = 0;
  virtual void reset_spectrum_accumulator() = 0;
};

struct RandomInitialCondition
{
  FFTW_REAL eps;
  mutable std::mt19937 gen;
  mutable std::uniform_real_distribution<FFTW_REAL> dist;

  RandomInitialCondition(FFTW_REAL eps_val, int seed_offset)
      : eps(eps_val), gen(std::random_device{}() + seed_offset),
        dist(static_cast<FFTW_REAL>(0.0), eps_val) {}

  FFTW_COMPLEX_STD operator()(FFTW_REAL /*x*/) const
  {
    return FFTW_COMPLEX_STD(dist(gen), -dist(gen));
  }
};

template <typename InitialCondition>
class cpu_domain_wall : public cpu_domain_wall_base
{
public:
  void reset(const InitialCondition &new_ic)
  {
    initial_condition = new_ic;
    stepper.reset(new_ic);
  }

  struct NonLinear
  {
    FFTW_REAL h;
    FFTW_REAL h0;
    FFTW_REAL Omega;
    FFTW_REAL alpha;
    FFTW_COMPLEX_STD prefactor;

    NonLinear(FFTW_REAL h_, FFTW_REAL h0_, FFTW_REAL Omega_, FFTW_REAL alpha_)
        : h(h_), h0(h0_), Omega(Omega_), alpha(alpha_),
          prefactor(alpha_ / static_cast<FFTW_REAL>(2.0), static_cast<FFTW_REAL>(-0.5)) {}

    FFTW_COMPLEX_STD operator()(FFTW_COMPLEX_STD z, FFTW_REAL t) const
    {
      FFTW_REAL current_h = h + h0 * std::cos(Omega * t);
      return prefactor * FFTW_COMPLEX_STD(alpha * current_h, std::sin(static_cast<FFTW_REAL>(2.0) * z.imag()));
    }
  };

  struct Linear
  {
    FFTW_REAL alpha;
    FFTW_COMPLEX_STD prefactor;

    Linear(FFTW_REAL alpha_)
        : alpha(alpha_), prefactor(static_cast<FFTW_REAL>(-0.5) * alpha_, static_cast<FFTW_REAL>(0.5)) {}

    FFTW_COMPLEX_STD operator()(FFTW_REAL k) const
    {
      return prefactor * (k * k);
    }
  };

  cpu_domain_wall(FFTW_REAL alpha_, FFTW_REAL h_, FFTW_REAL h0_, FFTW_REAL Omega_,
                  FFTW_REAL dt_, FFTW_REAL system_size_, FFTW_REAL min_resolution_,
                  const InitialCondition &initial_condition_)
      : alpha(alpha_), h(h_), h0(h0_), Omega(Omega_), dt(dt_),
        system_size(system_size_),
        n_samples(FftwUtils::getOptimalSize(
            static_cast<size_t>(std::ceil(system_size_ / min_resolution_)))),
        nonlinear(h_, h0_, Omega_, alpha_),
        linear(alpha_),
        initial_condition(initial_condition_),
        stepper(n_samples, system_size_, dt_, nonlinear, initial_condition_, linear)
  {
  }

  void set_time(FFTW_REAL t_)
  {
    stepper.set_time(t_);
  }

  FFTW_REAL get_time() const override { return stepper.get_time(); }
  FFTW_REAL get_dt() const override { return dt; }
  FFTW_REAL get_Omega() const override { return Omega; }

  void set_h(FFTW_REAL h_)
  {
    h = h_;
    nonlinear.h = h_;
    stepper.set_nonlinear_part(nonlinear);
  }

  void set_alpha(FFTW_REAL alpha_)
  {
    alpha = alpha_;
    nonlinear.alpha = alpha_;
    nonlinear.prefactor = FFTW_COMPLEX_STD(alpha_ / static_cast<FFTW_REAL>(2.0), static_cast<FFTW_REAL>(-0.5));
    linear.alpha = alpha_;
    linear.prefactor = FFTW_COMPLEX_STD(static_cast<FFTW_REAL>(-0.5) * alpha_, static_cast<FFTW_REAL>(0.5));
    stepper.set_nonlinear_part(nonlinear);
    stepper.set_linear_part(linear);
  }

  void set_h0(FFTW_REAL h0_)
  {
    h0 = h0_;
    nonlinear.h0 = h0_;
    stepper.set_nonlinear_part(nonlinear);
  }

  void set_Omega(FFTW_REAL Omega_)
  {
    Omega = Omega_;
    nonlinear.Omega = Omega_;
    stepper.set_nonlinear_part(nonlinear);
  }

  void set_dt(FFTW_REAL dt_)
  {
    dt = dt_;
    stepper.set_dt(dt_);
  }

  const AlignedComplexVector &get_z() const override { return stepper.get_z(); }
  const AlignedComplexVector &get_zhat() const override { return stepper.get_zhat(); }

  void step() override
  {
    stepper.step();
  }

  void create_step_graph(size_t /*steps*/) override
  {
  }

  void run_block(size_t steps) override
  {
    stepper.run_block(steps);
  }

  FFTW_REAL get_h() const override { return h; }
  FFTW_REAL get_h0() const override { return h0; }
  FFTW_REAL get_alpha() const override { return alpha; }
  FFTW_REAL get_system_size() const override { return system_size; }
  size_t get_n_samples() const override { return n_samples; }
  void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) override
  {
    stepper.set_fine_tracking(enable, sample_every_n_steps);
  }

  std::vector<FFTW_REAL> get_and_clear_fine_phi_dot_history() override
  {
    return stepper.get_and_clear_fine_phi_dot_history();
  }

  std::vector<FFTW_REAL> get_and_clear_fine_rugosity_history() override
  {
    return stepper.get_and_clear_fine_rugosity_history();
  }

  void accumulate_u_power_spectrum() override
  {
    stepper.accumulate_u_power_spectrum();
  }

  std::vector<FFTW_REAL> get_averaged_u_power_spectrum() const override
  {
    return stepper.get_averaged_u_power_spectrum();
  }

  void reset_spectrum_accumulator() override
  {
    stepper.reset_spectrum_accumulator();
  }

private:
  FFTW_REAL alpha, h, h0, Omega;
  FFTW_REAL dt, system_size;
  size_t n_samples;

  NonLinear nonlinear;
  Linear linear;
  InitialCondition initial_condition;

  cpu_pde_pseudospectral_stepper<NonLinear, InitialCondition, Linear> stepper;
};