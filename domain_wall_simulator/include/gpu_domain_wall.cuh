#pragma once
#include "gpu_pde_pseudospectral_solver.cuh"
#include "types.cuh"
#include <thrust/functional.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <thrust/random.h>

class gpu_domain_wall_base
{
public:
  virtual ~gpu_domain_wall_base() = default;

  virtual void step() = 0;

  virtual void create_step_graph(size_t steps) = 0;
  virtual void run_block(size_t steps) = 0;

  virtual cuda_math::REAL get_time() const = 0;
  virtual cuda_math::REAL get_dt() const = 0;
  virtual cuda_math::REAL get_Omega() const = 0;
  virtual const cuda_math::COMPLEX_DEVICE_VECTOR &get_z() const = 0;
  virtual const cuda_math::COMPLEX_DEVICE_VECTOR &get_zhat() const = 0;

  virtual cuda_math::REAL get_h() const = 0;
  virtual cuda_math::REAL get_h0() const = 0;
  virtual cuda_math::REAL get_alpha() const = 0;
  virtual cuda_math::REAL get_system_size() const = 0;
  virtual size_t get_n_samples() const = 0;

  virtual void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) = 0;
  virtual std::vector<cuda_math::REAL> get_and_clear_fine_phi_dot_history() = 0;
  virtual std::vector<cuda_math::REAL> get_and_clear_fine_rugosity_history() = 0;
  virtual void accumulate_u_power_spectrum() = 0;
  virtual std::vector<cuda_math::REAL> get_averaged_u_power_spectrum() const = 0;
  virtual void reset_spectrum_accumulator() = 0;
};

struct RandomInitialCondition
{
  cuda_math::REAL amplitude;
  unsigned int master_seed;

  __host__
  RandomInitialCondition(cuda_math::REAL amp, int seed_offset)
      : amplitude(amp)
  {
    std::random_device rd;
    master_seed = static_cast<unsigned int>(rd() + seed_offset);
  }

  __device__
      cuda_math::COMPLEX
      operator()(cuda_math::REAL x) const
  {
    unsigned int unique_id = static_cast<unsigned int>((x + static_cast<cuda_math::REAL>(10000.0)) * static_cast<cuda_math::REAL>(100000.0));
    unsigned int local_seed = master_seed ^ (unique_id * 2654435761U);

    thrust::default_random_engine rng(local_seed);
    thrust::uniform_real_distribution<cuda_math::REAL> dist(static_cast<cuda_math::REAL>(0.0), amplitude);

    return cuda_math::COMPLEX(dist(rng), -dist(rng));
  }
};

template <typename InitialCondition>
class gpu_domain_wall : public gpu_domain_wall_base
{
public:
  void reset(const InitialCondition &new_ic)
  {
    initial_condition = new_ic;
    stepper.reset(new_ic);
    graph_needs_update = true;
  }

  struct NonLinear
  {
    cuda_math::REAL h;
    cuda_math::REAL h0;
    cuda_math::REAL Omega;
    cuda_math::REAL alpha;
    cuda_math::COMPLEX prefactor;

    NonLinear(cuda_math::REAL h_, cuda_math::REAL h0_, cuda_math::REAL Omega_, cuda_math::REAL alpha_)
        : h(h_), h0(h0_), Omega(Omega_), alpha(alpha_),
          prefactor(alpha_ / static_cast<cuda_math::REAL>(2.0), static_cast<cuda_math::REAL>(-0.5)) {} // Precompute this division and construction

    __device__ __host__
        cuda_math::COMPLEX
        operator()(cuda_math::COMPLEX z, cuda_math::REAL t) const
    {
      cuda_math::REAL current_h = h + h0 * cos(Omega * t);
      return prefactor * cuda_math::COMPLEX(alpha * current_h, sin(static_cast<cuda_math::REAL>(2.0) * z.imag()));
    }
  };

  struct Linear
  {
    cuda_math::REAL alpha;
    cuda_math::COMPLEX prefactor;

    Linear(cuda_math::REAL alpha_)
        : alpha(alpha_),
          prefactor(static_cast<cuda_math::REAL>(-0.5) * alpha_, static_cast<cuda_math::REAL>(0.5)) {} // -0.5 * (alpha, -1.0) = (-0.5*alpha, 0.5)

    __device__ __host__
        cuda_math::COMPLEX
        operator()(cuda_math::REAL k) const
    {
      return prefactor * (k * k);
    }
  };

  gpu_domain_wall(cuda_math::REAL alpha_, cuda_math::REAL h_, cuda_math::REAL h0_, cuda_math::REAL Omega_,
                  cuda_math::REAL dt_, cuda_math::REAL system_size_, cuda_math::REAL min_resolution_,
                  const InitialCondition &initial_condition_)
      : alpha(alpha_), h(h_), h0(h0_), Omega(Omega_), dt(dt_),
        system_size(system_size_),
        n_samples(CufftUtils::getOptimalSize(
            static_cast<size_t>(std::ceil(system_size_ / min_resolution_)))),
        nonlinear(h_, h0_, Omega_, alpha_),
        linear(alpha_),
        initial_condition(initial_condition_),
        stepper(n_samples, system_size_, dt_, nonlinear, initial_condition_, linear)
  {
  }

  void set_time(cuda_math::REAL t_)
  {
    stepper.set_time(t_);
  }

  cuda_math::REAL get_time() const override { return stepper.get_time(); }
  cuda_math::REAL get_dt() const override { return dt; }
  cuda_math::REAL get_Omega() const override { return Omega; }

  void set_h(cuda_math::REAL h_)
  {
    h = h_;
    nonlinear.h = h_;
    stepper.set_nonlinear_part(nonlinear);
    graph_needs_update = true;
  }

  void set_alpha(cuda_math::REAL alpha_)
  {
    alpha = alpha_;
    nonlinear.alpha = alpha_;
    nonlinear.prefactor = cuda_math::COMPLEX(alpha_ / static_cast<cuda_math::REAL>(2.0), static_cast<cuda_math::REAL>(-0.5));
    linear.alpha = alpha_;
    linear.prefactor = cuda_math::COMPLEX(static_cast<cuda_math::REAL>(-0.5) * alpha_, static_cast<cuda_math::REAL>(0.5));
    stepper.set_nonlinear_part(nonlinear);
    stepper.set_linear_part(linear);
    graph_needs_update = true;
  }

  void set_h0(cuda_math::REAL h0_)
  {
    h0 = h0_;
    nonlinear.h0 = h0_;
    stepper.set_nonlinear_part(nonlinear);
    graph_needs_update = true;
  }

  void set_Omega(cuda_math::REAL Omega_)
  {
    Omega = Omega_;
    nonlinear.Omega = Omega_;
    stepper.set_nonlinear_part(nonlinear);
    graph_needs_update = true;
  }

  void set_dt(cuda_math::REAL dt_)
  {
    dt = dt_;
    stepper.set_dt(dt_);
    graph_needs_update = true;
  }

  const cuda_math::COMPLEX_DEVICE_VECTOR &get_z() const override { return stepper.get_z(); }
  const cuda_math::COMPLEX_DEVICE_VECTOR &get_zhat() const override { return stepper.get_zhat(); }

  void step() override
  {
    stepper.step();
  }

  void create_step_graph(size_t steps) override
  {
    stepper.create_step_graph(steps);
  }

  void run_block(size_t steps) override
  {
    if (graph_needs_update)
    {
      stepper.create_step_graph(steps);
      graph_needs_update = false;
    }
    stepper.run_block(steps);
  }

  cuda_math::REAL get_h() const override { return h; }
  cuda_math::REAL get_h0() const override { return h0; }
  cuda_math::REAL get_alpha() const override { return alpha; }
  cuda_math::REAL get_system_size() const override { return system_size; }
  size_t get_n_samples() const override { return n_samples; }

  void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) override
  {
    stepper.set_fine_tracking(enable, sample_every_n_steps);
  }

  std::vector<cuda_math::REAL> get_and_clear_fine_phi_dot_history() override
  {
    return stepper.get_and_clear_fine_phi_dot_history();
  }

  std::vector<cuda_math::REAL> get_and_clear_fine_rugosity_history() override
  {
    return stepper.get_and_clear_fine_rugosity_history();
  }

  void accumulate_u_power_spectrum() override
  {
    stepper.accumulate_u_power_spectrum();
  }

  std::vector<cuda_math::REAL> get_averaged_u_power_spectrum() const override
  {
    return stepper.get_averaged_u_power_spectrum();
  }

  void reset_spectrum_accumulator() override
  {
    stepper.reset_spectrum_accumulator();
  }

private:
  cuda_math::REAL alpha,
      h,
      h0,
      Omega;
  cuda_math::REAL dt, system_size;
  size_t n_samples;

  NonLinear nonlinear;
  Linear linear;
  InitialCondition initial_condition;

  bool graph_needs_update = true;
  gpu_pde_pseudospectral_stepper<NonLinear, InitialCondition, Linear> stepper;
};
