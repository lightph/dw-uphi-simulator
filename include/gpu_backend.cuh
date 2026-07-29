#pragma once

#include "cuda_memory.cuh"
#include "cufft_handlers.cuh"

namespace dw {

template <typename Precision>
struct GpuBackend {
    using PrecisionType = Precision;

    template <typename T>
    using Vector = CudaVector<T>;

    using R2C = CufftHandlerR2C<Precision>;
    using C2R = CufftHandlerC2R<Precision>;
    using C2CIn = CufftHandlerC2CIn<Precision>;
    using C2COut = CufftHandlerC2COut<Precision>;
};

}  // namespace dw