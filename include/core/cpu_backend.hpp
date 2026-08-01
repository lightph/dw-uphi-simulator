#pragma once

#include <cmath>
#include <vector>

#include "aligned_memory.hpp"
#include "fftw_handlers.hpp"

namespace dw {

template <typename Precision>
struct CpuBackend {
    using PrecisionType = Precision;

    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;

    using R2C = FftwHandlerR2C<Precision>;
    using C2R = FftwHandlerC2R<Precision>;
    using C2CIn = FftwHandlerC2CIn<Precision>;
    using C2COut = FftwHandlerC2COut<Precision>;

    template <typename T>
    using Vector = AlignedVector<T>;

    using ComplexVector = AlignedVector<Complex>;

    static void compute_difference(const typename Precision::Complex* u,
                                   const typename Precision::Complex* u_prev,
                                   typename Precision::Complex* delta, std::size_t size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < size; ++i) {
            delta[i] = u[i] - u_prev[i];
        }
    }
    static void synchronize() { return; }
    static void evaluate_nonlinear(const ComplexVector& z, ComplexVector& nl_out, Real alpha,
                                   Real h_val) {
        Complex prefactor(alpha / Real(2.0), Real(-1.0) / Real(2.0));

#pragma omp parallel for
        for (std::size_t j = 0; j < z.size(); ++j) {
            Real phi = -z[j].imag();
            Complex bracket(alpha * h_val, std::sin(Real(2.0) * phi));
            nl_out[j] = prefactor * bracket;
        }
    }

    static void step_linear(ComplexVector& z_hat, const ComplexVector& nl_hat,
                            const Complex* prop_z, const Complex* prop_nl) {
        Real norm_factor = Real(1.0) / static_cast<Real>(z_hat.size());

#pragma omp parallel for
        for (std::size_t j = 0; j < z_hat.size(); ++j) {
            z_hat[j] = (prop_z[j] * z_hat[j] + prop_nl[j] * nl_hat[j]) * norm_factor;
        }
    }

    static void copy(const ComplexVector& src, ComplexVector& dst) { dst = src; }

    static Real compute_diff_sq(const ComplexVector& u1, const ComplexVector& u2) {
        Real sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
        for (std::size_t i = 0; i < u1.size(); ++i) {
            Real dr = u1[i].real() - u2[i].real();
            Real di = u1[i].imag() - u2[i].imag();
            sum += dr * dr + di * di;
        }
        return sum;
    }

    struct ObsTuple {
        Real sin2phi, u, u2;
    };

    static ObsTuple compute_observables(const ComplexVector& z) {
        Real s_sin = 0.0, s_u = 0.0, s_u2 = 0.0;

#pragma omp parallel for reduction(+ : s_sin, s_u, s_u2)
        for (std::size_t i = 0; i < z.size(); ++i) {
            Real u = z[i].real();
            Real phi = -z[i].imag();
            s_sin += std::sin(Real(2.0) * phi);
            s_u += u;
            s_u2 += u * u;
        }
        return {s_sin, s_u, s_u2};
    }
};

}  // namespace dw