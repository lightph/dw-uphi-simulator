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
};

}  // namespace dw