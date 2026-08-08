#pragma once

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
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
struct UPowerSpectrumOp {
    const Complex* z_hat;
    std::size_t N;

    __host__ __device__ Real operator()(std::size_t i) const {
        std::size_t i_neg = (N - i) % N;
        Real A = z_hat[i].real();
        Real B = z_hat[i].imag();
        Real C = z_hat[i_neg].real();
        Real D = z_hat[i_neg].imag();

        Real norm = Real(1.0) / static_cast<Real>(N);
        Real real_u = Real(0.5) * (A + C) * norm;
        Real imag_u = Real(0.5) * (B - D) * norm;
        return real_u * real_u + imag_u * imag_u;
    }
};

template <typename Real, typename Complex>
struct UEntropyOp {
    const Complex* z_hat;
    std::size_t N;
    Real sum_S;

    __host__ __device__ Real operator()(std::size_t i) const {
        std::size_t i_neg = (N - i) % N;
        Real A = z_hat[i].real();
        Real B = z_hat[i].imag();
        Real C = z_hat[i_neg].real();
        Real D = z_hat[i_neg].imag();

        Real norm = Real(1.0) / static_cast<Real>(N);
        Real real_u = Real(0.5) * (A + C) * norm;
        Real imag_u = Real(0.5) * (B - D) * norm;
        Real S = real_u * real_u + imag_u * imag_u;

        if (S > Real(0.0)) {
            Real p = S / sum_S;
            return -p * log(p);
        }
        return Real(0.0);
    }
};

template <typename Real, typename Complex>
__global__ void accumulate_ps_u_kernel(const Complex* z_hat, Real* ps_accum, std::size_t size) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        std::size_t i_neg = (size - i) % size;
        Real A = z_hat[i].real();
        Real B = z_hat[i].imag();
        Real C = z_hat[i_neg].real();
        Real D = z_hat[i_neg].imag();

        Real norm = Real(1.0) / static_cast<Real>(size);
        Real real_u = Real(0.5) * (A + C) * norm;
        Real imag_u = Real(0.5) * (B - D) * norm;
        ps_accum[i] += real_u * real_u + imag_u * imag_u;
    }
}

template <typename Real, typename Complex>
__global__ void accumulate_hist_kernel(const Complex* z, unsigned long long* hist, std::size_t size,
                                       std::size_t num_bins, Real mean_u, Real sigma_u,
                                       Real min_val, Real max_val) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
        Real norm_u = (z[j].real() - mean_u) / sigma_u;
        if (norm_u >= min_val && norm_u < max_val) {
            Real bin_width = (max_val - min_val) / static_cast<Real>(num_bins);
            int bin = static_cast<int>((norm_u - min_val) / bin_width);
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&hist[bin], 1ULL);
            }
        }
    }
}

template <typename Real, typename Complex>
__global__ void record_observables_kernel(const Complex* z, std::size_t size, Real* d_mean_u,
                                          Real* d_mean_u2, Real* d_mean_sin2phi,
                                          std::size_t step_idx) {
    extern __shared__ unsigned char shared_mem[];
    Real* s_u = reinterpret_cast<Real*>(shared_mem);
    Real* s_u2 = s_u + blockDim.x;
    Real* s_sin = s_u2 + blockDim.x;

    Real sum_u = Real(0.0);
    Real sum_u2 = Real(0.0);
    Real sum_sin = Real(0.0);

    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;

    for (std::size_t i = tid; i < size; i += stride) {
        Real u = z[i].real();
        Real phi = -z[i].imag();
        sum_u += u;
        sum_u2 += u * u;
        sum_sin += sin(Real(2.0) * phi);
    }

    s_u[threadIdx.x] = sum_u;
    s_u2[threadIdx.x] = sum_u2;
    s_sin[threadIdx.x] = sum_sin;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_u[threadIdx.x] += s_u[threadIdx.x + s];
            s_u2[threadIdx.x] += s_u2[threadIdx.x + s];
            s_sin[threadIdx.x] += s_sin[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        Real N = static_cast<Real>(size);
        atomicAdd(&d_mean_u[step_idx], s_u[0] / N);
        atomicAdd(&d_mean_u2[step_idx], s_u2[0] / N);
        atomicAdd(&d_mean_sin2phi[step_idx], s_sin[0] / N);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

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

    static void accumulate_power_spectrum(const ComplexVector& u_hat, CudaVector<Real>& ps_accum) {
        int blockSize = 256;
        int numBlocks = (u_hat.size() + blockSize - 1) / blockSize;
        accumulate_ps_u_kernel<Real, Complex>
            <<<numBlocks, blockSize>>>(u_hat.data(), ps_accum.data(), u_hat.size());
    }

    static Real compute_spectral_entropy(const ComplexVector& u_hat) {
        std::size_t N = u_hat.size();
        auto count_it = thrust::make_counting_iterator<std::size_t>(0);

        UPowerSpectrumOp<Real, Complex> ps_op{u_hat.data(), N};

        // Start transform_reduce from index 1 (count_it + 1) to ignore k=0
        Real sum_S = thrust::transform_reduce(count_it + 1, count_it + N, ps_op, Real(0.0),
                                              thrust::plus<Real>());

        if (sum_S <= Real(0.0)) return Real(0.0);

        UEntropyOp<Real, Complex> ent_op{u_hat.data(), N, sum_S};
        Real entropy = thrust::transform_reduce(count_it + 1, count_it + N, ent_op, Real(0.0),
                                                thrust::plus<Real>());

        return entropy / std::log(static_cast<Real>(N - 1));
    }

    static void fill_zero(CudaVector<Real>& vec) {
        thrust::device_ptr<Real> ptr(vec.data());
        thrust::fill(ptr, ptr + vec.size(), Real(0.0));
    }

    static std::vector<Real> download_array(const CudaVector<Real>& vec) {
        std::vector<Real> host_vec(vec.size());
        cudaMemcpy(host_vec.data(), vec.data(), vec.size() * sizeof(Real), cudaMemcpyDeviceToHost);
        return host_vec;
    }

    static std::vector<Complex> download_array(const ComplexVector& vec) {
        std::vector<Complex> host_vec(vec.size());
        cudaMemcpy(host_vec.data(), vec.data(), vec.size() * sizeof(Complex),
                   cudaMemcpyDeviceToHost);
        return host_vec;
    }

    static void accumulate_height_histogram(const ComplexVector& z, Real mean_u, Real sigma_u,
                                            CudaVector<unsigned long long>& hist, Real min_val,
                                            Real max_val) {
        int blockSize = 256;
        int numBlocks = (z.size() + blockSize - 1) / blockSize;
        accumulate_hist_kernel<Real, Complex><<<numBlocks, blockSize>>>(
            z.data(), hist.data(), z.size(), hist.size(), mean_u, sigma_u, min_val, max_val);
    }

    static void fill_zero_ull(CudaVector<unsigned long long>& vec) {
        cudaMemset(vec.data(), 0, vec.size() * sizeof(unsigned long long));
    }

    static std::vector<unsigned long long> download_array_ull(
        const CudaVector<unsigned long long>& vec) {
        std::vector<unsigned long long> host_vec(vec.size());
        cudaMemcpy(host_vec.data(), vec.data(), vec.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        return host_vec;
    }

    static void record_observables_async(const ComplexVector& z, CudaVector<Real>& d_mean_u,
                                         CudaVector<Real>& d_mean_u2,
                                         CudaVector<Real>& d_mean_sin2phi, std::size_t step_idx) {
        int blockSize = 256;
        int numBlocks = std::min(1024, static_cast<int>((z.size() + blockSize - 1) / blockSize));
        std::size_t sharedMemSize = 3 * blockSize * sizeof(Real);

        record_observables_kernel<Real, Complex><<<numBlocks, blockSize, sharedMemSize>>>(
            z.data(), z.size(), d_mean_u.data(), d_mean_u2.data(), d_mean_sin2phi.data(), step_idx);
    }
};

}  // namespace dw