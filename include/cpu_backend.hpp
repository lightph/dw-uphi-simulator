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
};

}  // namespace dw