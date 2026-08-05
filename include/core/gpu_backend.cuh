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

template <typename Real, typename Complex>
struct PowerSpectrumOp {
    __host__ __device__ Real operator()(const Complex& c) const {
        Real r = c.real();
        Real im = c.imag();
        return r * r + im * im;
    }
};

template <typename Real, typename Complex>
struct EntropyOp {
    Real sum_S;
    __host__ __device__ Real operator()(const Complex& c) const {
        Real r = c.real();
        Real im = c.imag();
        Real S = r * r + im * im;
        if (S > 0.0) {
            Real p = S / sum_S;
            return -p * log(p);  // cuFFT uses math.h log
        }
        return 0.0;
    }
};

template <typename Real, typename Complex>
__global__ void accumulate_ps_kernel(const Complex* u_hat, Real* ps_accum, std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        Real r = u_hat[i].real();
        Real im = u_hat[i].imag();
        ps_accum[i] += r * r + im * im;
    }
}

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

    static Real compute_spectral_entropy(const ComplexVector& u_hat) {
        thrust::device_ptr<const Complex> ptr(u_hat.data());

        PowerSpectrumOp<Real, Complex> ps_op;
        Real sum_S = thrust::transform_reduce(ptr, ptr + u_hat.size(), ps_op, Real(0.0),
                                              thrust::plus<Real>());

        if (sum_S <= 0.0) return Real(0.0);

        EntropyOp<Real, Complex> ent_op{sum_S};
        Real entropy = thrust::transform_reduce(ptr, ptr + u_hat.size(), ent_op, Real(0.0),
                                                thrust::plus<Real>());

        return entropy / std::log(static_cast<Real>(u_hat.size()));
    }

    static void fill_zero(CudaVector<Real>& vec) {
        thrust::device_ptr<Real> ptr(vec.data());
        thrust::fill(ptr, ptr + vec.size(), Real(0.0));
    }

    static void accumulate_power_spectrum(const ComplexVector& u_hat, CudaVector<Real>& ps_accum) {
        int blockSize = 256;
        int numBlocks = (u_hat.size() + blockSize - 1) / blockSize;
        accumulate_ps_kernel<Real, Complex>
            <<<numBlocks, blockSize>>>(u_hat.data(), ps_accum.data(), u_hat.size());
    }

    static std::vector<Real> download_array(const CudaVector<Real>& vec) {
        std::vector<Real> host_vec(vec.size());
        cudaMemcpy(host_vec.data(), vec.data(), vec.size() * sizeof(Real), cudaMemcpyDeviceToHost);
        return host_vec;
    }
};

}  // namespace dw