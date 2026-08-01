#pragma once

#include "aligned_memory.hpp"
#include "fftw_handlers.hpp"

namespace dw {

template <typename Precision>
struct CpuBackend {
    using PrecisionType = Precision;

    template <typename T>
    using Vector = AlignedVector<T>;

    using R2C = FftwHandlerR2C<Precision>;
    using C2R = FftwHandlerC2R<Precision>;
    using C2CIn = FftwHandlerC2CIn<Precision>;
    using C2COut = FftwHandlerC2COut<Precision>;

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
};

}  // namespace dw