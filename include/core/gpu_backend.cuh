#pragma once

#include "cuda_memory.cuh"
#include "cufft_handlers.cuh"

namespace dw {

template <typename Complex>
__global__ void compute_difference_kernel(const Complex* u, const Complex* u_prev, Complex* delta,
                                          std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        delta[i] = u[i] - u_prev[i];
    }
}

template <typename Real, typename Complex>
__global__ void nonlinear_kernel(const Complex* z, Complex* nl_out, Real alpha, Real h_val,
                                 std::size_t size) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
        Real phi = -z[j].imag();
        Complex prefactor(alpha / Real(2.0), Real(-1.0) / Real(2.0));
        Complex bracket(alpha * h_val, std::sin(Real(2.0) * phi));
        nl_out[j] = prefactor * bracket;
    }
}

template <typename Real, typename Complex>
__global__ void linear_kernel(Complex* z_hat, const Complex* nl_hat, const Complex* prop_z,
                              const Complex* prop_nl, std::size_t size) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
        Real norm_factor = Real(1.0) / static_cast<Real>(size);
        z_hat[j] = (prop_z[j] * z_hat[j] + prop_nl[j] * nl_hat[j]) * norm_factor;
    }
}

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
    static void evaluate_nonlinear(const ComplexVector& z, ComplexVector& nl_out, Real alpha,
                                   Real h_val) {
        int blockSize = 256;
        int numBlocks = (z.size() + blockSize - 1) / blockSize;
        nonlinear_kernel<<<numBlocks, blockSize>>>(z.data(), nl_out.data(), alpha, h_val, z.size());
    }

    static void step_linear(ComplexVector& z_hat, const ComplexVector& nl_hat,
                            const Complex* prop_z, const Complex* prop_nl) {
        int blockSize = 256;
        int numBlocks = (z_hat.size() + blockSize - 1) / blockSize;
        linear_kernel<<<numBlocks, blockSize>>>(z_hat.data(), nl_hat.data(), prop_z, prop_nl,
                                                z_hat.size());
    }
};

}  // namespace dw