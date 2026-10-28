#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

#include "state.cuh"
#include "stepper.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dw {

template <typename Backend>
class DomainWall {
   public:
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    Real eps_;
    unsigned int seed_;

    Real get_alpha() const { return alpha_; }
    Real get_omega() const { return omega_; }
    const typename Backend::template Vector<Real>& get_h0_batch() const { return h0_; }
    const typename Backend::template Vector<Real>& get_ha_batch() const { return ha_; }
    Real get_L() const { return L_; }

    DomainWall(std::size_t size, double L, double alpha, const std::vector<double>& h0_batch,
               const std::vector<double>& ha_batch, double omega, double dt, double eps,
               unsigned int seed)
        : size_(size),
          batch_(h0_batch.size()),
          L_(static_cast<Real>(L)),
          alpha_(static_cast<Real>(alpha)),
          omega_(static_cast<Real>(omega)),
          dt_(static_cast<Real>(dt)),
          eps_(static_cast<Real>(eps)),
          seed_(seed),
          h0_(h0_batch.size()),
          ha_(ha_batch.size()),
          state_(size, h0_batch.size()),
          stepper_(size, h0_batch.size()),
          prop_z_(size),
          prop_nl_(size) {
        for (std::size_t b = 0; b < batch_; ++b) {
            h0_[b] = static_cast<Real>(h0_batch[b]);
            ha_[b] = static_cast<Real>(ha_batch[b]);
        }

        initialize_propagators();
        initialize_state();
#ifdef HAS_CUDA
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaMemLocation loc = {};
        loc.type =
            (deviceId == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
        loc.id = (deviceId == cudaCpuDeviceId) ? 0 : deviceId;
        cudaMemPrefetchAsync(h0_.data(), batch_ * sizeof(Real), loc, 0);
        cudaMemPrefetchAsync(ha_.data(), batch_ * sizeof(Real), loc, 0);
#endif
    }

    void step(Real& current_time, cudaStream_t stream = nullptr) {
        NonlinearEvaluator nonlin{alpha_,     omega_, current_time, h0_.data(),
                                  ha_.data(), size_,  batch_};
        LinearStepper lin{prop_z_.data(), prop_nl_.data()};

        stepper_.step(state_, dt_, lin, nonlin, stream);
        current_time += dt_;
    }

    const State<Backend>& get_state() const { return state_; }
    State<Backend>& get_state() { return state_; }
    std::size_t get_size() const { return size_; }
    std::size_t get_batch() const { return batch_; }
    Real get_dt() const { return dt_; }

   private:
    std::size_t size_;
    std::size_t batch_;
    Real L_, alpha_, omega_, dt_;
    typename Backend::template Vector<Real> h0_;
    typename Backend::template Vector<Real> ha_;
    State<Backend> state_;
    PseudospectralStepper<Backend> stepper_;
    ComplexVector prop_z_;
    ComplexVector prop_nl_;

    void initialize_propagators() {
        Real dk = static_cast<Real>(2.0 * M_PI / L_);
        Complex dt_c(dt_, Real(0.0));
        Complex one(Real(1.0), Real(0.0));
        Complex half(Real(0.5), Real(0.0));

        // Calculate the absolute maximum physical wavenumber the grid supports
        Real k_max = static_cast<Real>(size_ / 2) * dk;

        for (std::size_t i = 0; i < size_; ++i) {
            long long k =
                (i <= size_ / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size_);
            Real k_phys = static_cast<Real>(k) * dk;

            // Apply the 2/3 De-Aliasing Rule
            if (std::abs(k_phys) > k_max * (2.0 / 3.0)) {
                prop_z_[i] = Complex(0.0, 0.0);
                prop_nl_[i] = Complex(0.0, 0.0);
                continue;
            }

            Real k2 = k_phys * k_phys;
            Complex L_k(-alpha_ * k2 / Real(2.0), k2 / Real(2.0));
            Complex half_dt_L_k = half * dt_c * L_k;

            Complex numerator = one + half_dt_L_k;
            Complex denominator = one - half_dt_L_k;

            prop_z_[i] = numerator / denominator;
            prop_nl_[i] = dt_c / denominator;
        }
        Backend::synchronize();
    }

    void initialize_state() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<Real> dist(-eps_, eps_);
        std::size_t total_elements = size_ * batch_;

        // Populate the entire contiguous batched memory block
        for (std::size_t i = 0; i < total_elements; ++i) {
            Real u = dist(gen);
            Real phi = dist(gen);
            state_.u[i] = Complex(u, phi);
        }
#ifdef HAS_CUDA
        int deviceId;
        cudaGetDevice(&deviceId);

        cudaMemLocation loc = {};
        loc.type =
            (deviceId == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
        loc.id = (deviceId == cudaCpuDeviceId) ? 0 : deviceId;

        cudaMemPrefetchAsync(prop_z_.data(), size_ * sizeof(Complex), loc, 0);
        cudaMemPrefetchAsync(prop_nl_.data(), size_ * sizeof(Complex), loc, 0);
        cudaMemPrefetchAsync(state_.u.data(), total_elements * sizeof(Complex), loc, 0);
        cudaMemPrefetchAsync(state_.u_hat.data(), total_elements * sizeof(Complex), loc, 0);
        cudaMemPrefetchAsync(state_.nl_eval.data(), total_elements * sizeof(Complex), loc, 0);
        cudaMemPrefetchAsync(state_.nl_hat.data(), total_elements * sizeof(Complex), loc, 0);
#endif
    }

    struct NonlinearEvaluator {
        Real alpha, omega, t;
        const Real* h0;
        const Real* ha;
        std::size_t size, batch;

        void operator()(const Complex* z, Complex* nl_out, std::size_t /*total_elements*/,
                        cudaStream_t stream = nullptr) const {
            Backend::evaluate_nonlinear(z, nl_out, size, batch, alpha, h0, ha, omega, t, stream);
        }
    };

    struct LinearStepper {
        const Complex* prop_z;
        const Complex* prop_nl;

        void operator()(Complex* z_hat, const Complex* nl_hat, double /*dt_in*/,
                        std::size_t grid_size, std::size_t total_elements,
                        cudaStream_t stream = nullptr) const {
            Backend::step_linear(z_hat, nl_hat, prop_z, prop_nl, grid_size, total_elements, stream);
        }
    };
};

}  // namespace dw
