#pragma once

#include "cuda_memory.cuh"
#include "cufft_handlers.cuh"

template <typename Complex>
__global__ void compute_difference_kernel(const Complex* u, const Complex* u_prev, Complex* delta,
                                          std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        delta[i] = u[i] - u_prev[i];
    }
}

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

    static void compute_difference(const typename Precision::Complex* u,
                                   const typename Precision::Complex* u_prev,
                                   typename Precision::Complex* delta, std::size_t size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        compute_difference_kernel<<<numBlocks, blockSize>>>(u, u_prev, delta, size);
    }
    static void synchronize() { cudaDeviceSynchronize(); }
};

}  // namespace dw