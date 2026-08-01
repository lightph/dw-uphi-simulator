#pragma once

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

#include <cmath>

#include "cuda_memory.cuh"
#include "cufft_handlers.cuh"

namespace dw {

template <typename Real>
struct ObsTuple {
    Real sin2phi, u, u2;
};

template <typename Real, typename Complex>
struct ObsFunctor {
    __host__ __device__ ObsTuple<Real> operator()(const Complex& z) const {
        Real u = z.real();
        Real phi = -z.imag();
        return {std::sin(Real(2.0) * phi), u, u * u};
    }
};

template <typename Real>
struct ObsReduce {
    __host__ __device__ ObsTuple<Real> operator()(const ObsTuple<Real>& a,
                                                  const ObsTuple<Real>& b) const {
        return {a.sin2phi + b.sin2phi, a.u + b.u, a.u2 + b.u2};
    }
};

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

template <typename Real, typename Complex>
struct DiffSqFunctor {
    template <typename Tuple>
    __host__ __device__ Real operator()(const Tuple& t) const {
        Complex a = thrust::get<0>(t);
        Complex b = thrust::get<1>(t);
        Real dr = a.real() - b.real();
        Real di = a.imag() - b.imag();
        return dr * dr + di * di;
    }
};

template <typename Precision>
struct GpuBackend {
    using PrecisionType = Precision;

    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;

    template <typename T>
    using Vector = CudaVector<T>;

    using R2C = CufftHandlerR2C<Precision>;
    using C2R = CufftHandlerC2R<Precision>;
    using C2CIn = CufftHandlerC2CIn<Precision>;
    using C2COut = CufftHandlerC2COut<Precision>;
    using ComplexVector = CudaVector<Complex>;

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

        nonlinear_kernel<Real, Complex>
            <<<numBlocks, blockSize>>>(z.data(), nl_out.data(), alpha, h_val, z.size());
    }

    static void step_linear(ComplexVector& z_hat, const ComplexVector& nl_hat,
                            const Complex* prop_z, const Complex* prop_nl) {
        int blockSize = 256;
        int numBlocks = (z_hat.size() + blockSize - 1) / blockSize;

        linear_kernel<Real, Complex>
            <<<numBlocks, blockSize>>>(z_hat.data(), nl_hat.data(), prop_z, prop_nl, z_hat.size());
    }

    static void copy(const ComplexVector& src, ComplexVector& dst) {
        cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(Complex), cudaMemcpyDeviceToDevice);
    }

    static Real compute_diff_sq(const ComplexVector& u1, const ComplexVector& u2) {
        thrust::device_ptr<const Complex> ptr1(u1.data());
        thrust::device_ptr<const Complex> ptr2(u2.data());

        auto start = thrust::make_zip_iterator(thrust::make_tuple(ptr1, ptr2));
        auto end =
            thrust::make_zip_iterator(thrust::make_tuple(ptr1 + u1.size(), ptr2 + u1.size()));

        DiffSqFunctor<Real, Complex> functor;
        return thrust::transform_reduce(start, end, functor, Real(0.0), thrust::plus<Real>());
    }

    static ObsTuple<Real> compute_observables(const ComplexVector& z) {
        thrust::device_ptr<const Complex> ptr(z.data());
        ObsFunctor<Real, Complex> transform_op;
        ObsReduce<Real> reduce_op;
        ObsTuple<Real> init = {Real(0.0), Real(0.0), Real(0.0)};

        return thrust::transform_reduce(ptr, ptr + z.size(), transform_op, init, reduce_op);
    }
};

}  // namespace dw