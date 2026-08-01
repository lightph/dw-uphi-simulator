#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>

#include "state.hpp"
#include "stepper.hpp"

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

    DomainWall(std::size_t size, double L, double alpha, double h0, double ha, double omega,
               double dt)
        : size_(size),
          L_(static_cast<Real>(L)),
          alpha_(static_cast<Real>(alpha)),
          h0_(static_cast<Real>(h0)),
          ha_(static_cast<Real>(ha)),
          omega_(static_cast<Real>(omega)),
          dt_(static_cast<Real>(dt)),
          state_(size),
          stepper_(size),
          prop_z_(size),
          prop_nl_(size) {
        initialize_propagators();
        initialize_state();
    }

    void run(const std::string& output_file, unsigned long long max_steps = 1ULL << 20) {
        std::ofstream out(output_file);
        out << std::setprecision(std::numeric_limits<Real>::max_digits10);

        Real current_time = 0.0;
        unsigned long long step_count = 1;
        unsigned long long target_step = 2;
        int n = 1;

        const Complex* d_prop_z = prop_z_.data();
        const Complex* d_prop_nl = prop_nl_.data();

        while (step_count <= max_steps) {
            NonlinearEvaluator nonlin{alpha_, h0_, ha_, omega_, current_time};
            LinearStepper lin{d_prop_z, d_prop_nl};

            stepper_.step(state_, dt_, lin, nonlin);
            current_time += dt_;

            if (step_count == target_step) {
                Backend::synchronize();

                out << "n=" << n << " step=" << step_count << " t=" << current_time << "\n";
                for (std::size_t i = 0; i < size_; ++i) {
                    out << state_.u[i].real() << " " << -state_.u[i].imag() << "\n";
                }

                n++;
                target_step = 1ULL << n;
            }
            step_count++;
        }
        out.close();
    }

   private:
    std::size_t size_;
    Real L_, alpha_, h0_, ha_, omega_, dt_;
    State<Backend> state_;
    PseudospectralStepper<Backend> stepper_;
    ComplexVector prop_z_;
    ComplexVector prop_nl_;

    void initialize_propagators() {
        Real dk = static_cast<Real>(2.0 * M_PI / L_);
        Complex dt_c(dt_, Real(0.0));
        Complex one(Real(1.0), Real(0.0));
        Complex half(Real(0.5), Real(0.0));

        for (std::size_t i = 0; i < size_; ++i) {
            long long k =
                (i <= size_ / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size_);
            Real k_phys = static_cast<Real>(k) * dk;
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
        std::mt19937 gen(42);
        std::uniform_real_distribution<Real> dist(-0.05, 0.05);
        for (std::size_t i = 0; i < size_; ++i) {
            Real u = dist(gen);
            Real phi = dist(gen);
            state_.u[i] = Complex(u, -phi);
        }
        Backend::synchronize();
    }

    struct NonlinearEvaluator {
        Real alpha, h0, ha, omega, t;

        void operator()(const ComplexVector& z, ComplexVector& nl_out) const {
            Real h_val = h0 + ha * std::cos(omega * t);
            Backend::evaluate_nonlinear(z, nl_out, alpha, h_val);
        }
    };

    struct LinearStepper {
        const Complex* prop_z;
        const Complex* prop_nl;

        void operator()(ComplexVector& z_hat, const ComplexVector& nl_hat, double /*dt_in*/) const {
            Backend::step_linear(z_hat, nl_hat, prop_z, prop_nl);
        }
    };
};

}  // namespace dw