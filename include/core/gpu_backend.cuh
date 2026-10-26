#pragma once

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cmath>
#include <vector>

#include "cuda_memory.cuh"
#include "cufft_handlers.cuh"

namespace dw {

template <typename Real>
struct ObsTuple {
    Real sin2phi, u, var_sum;
};

template <typename Real>
struct Pass1Tuple {
    Real sin2phi, u;
};

template <typename Real, typename Complex>
struct Pass1Functor {
    __host__ __device__ Pass1Tuple<Real> operator()(const Complex& z) const {
        Real u = z.real();
        Real phi = -z.imag();
        return {std::sin(Real(2.0) * phi), u};
    }
};

template <typename Real>
struct Pass1Reduce {
    __host__ __device__ Pass1Tuple<Real> operator()(const Pass1Tuple<Real>& a,
                                                    const Pass1Tuple<Real>& b) const {
        return {a.sin2phi + b.sin2phi, a.u + b.u};
    }
};

template <typename Real, typename Complex>
struct Pass2Functor {
    Real mean_u;
    explicit Pass2Functor(Real mu) : mean_u(mu) {}

    __host__ __device__ Real operator()(const Complex& z) const {
        Real diff = z.real() - mean_u;
        return diff * diff;
    }
};

template <typename Real, typename Complex>
__global__ void nonlinear_kernel(const Complex* z, Complex* nl_out, std::size_t size,
                                 std::size_t batch, Real alpha, const Real* h0, const Real* ha,
                                 Real omega, Real t) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size * batch) {
        std::size_t b = j / size;
        Real h_val = h0[b] + ha[b] * std::cos(omega * t);
        Complex val = z[j];
        Real phi = -val.imag();
        Real sin_val = std::sin(Real(2.0) * phi);
        Real nl_real = Real(0.5) * (alpha * alpha * h_val + sin_val);
        Real nl_imag = Real(0.5) * (alpha * sin_val - alpha * h_val);
        nl_out[j] = Complex(nl_real, nl_imag);
    }
}

template <typename Real, typename Complex>
__global__ void linear_kernel(Complex* z_hat, const Complex* nl_hat, const Complex* prop_z,
                              const Complex* prop_nl, std::size_t grid_size,
                              std::size_t total_elements) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < total_elements) {
        std::size_t local_idx = j % grid_size;
        Real norm_factor = Real(1.0) / static_cast<Real>(grid_size);
        z_hat[j] = (prop_z[local_idx] * z_hat[j] + prop_nl[local_idx] * nl_hat[j]) * norm_factor;
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

        Real real_u = Real(0.5) * (A + C);
        Real imag_u = Real(0.5) * (B - D);

        // Multiply by k^2 to get the power spectrum of the spatial derivative
        Real k = static_cast<Real>((i <= N / 2) ? i : N - i);
        return (real_u * real_u + imag_u * imag_u) * (k * k);
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

        Real real_u = Real(0.5) * (A + C);
        Real imag_u = Real(0.5) * (B - D);

        // Multiply by k^2 to get the power spectrum of the spatial derivative
        Real k = static_cast<Real>((i <= N / 2) ? i : N - i);
        Real S = (real_u * real_u + imag_u * imag_u) * (k * k);

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

        Real real_u = Real(0.5) * (A + C);
        Real imag_u = Real(0.5) * (B - D);
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

__device__ __inline__ float safeAtomicAdd(float* address, float val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, val);
#else
    return 0.0f;
#endif
}

__device__ __inline__ double safeAtomicAdd(double* address, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#elif defined(__CUDA_ARCH__)
    return atomicAdd(address, val);
#else
    return 0.0;
#endif
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
        sum_sin += std::sin(Real(2.0) * phi);
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
        safeAtomicAdd(&d_mean_u[step_idx], s_u[0] / N);
        safeAtomicAdd(&d_mean_u2[step_idx], s_u2[0] / N);
        safeAtomicAdd(&d_mean_sin2phi[step_idx], s_sin[0] / N);
    }
}

inline unsigned long long binomial_coeff(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    unsigned long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

template <typename Real, int MaxN>
struct MomentTuple {
    Real vals[MaxN];

    __host__ __device__ MomentTuple() {
        for (int i = 0; i < MaxN; ++i) {
            vals[i] = Real(0.0);
        }
    }

    __host__ __device__ MomentTuple operator+(const MomentTuple& other) const {
        MomentTuple res;
        for (int i = 0; i < MaxN; ++i) {
            res.vals[i] = vals[i] + other.vals[i];
        }
        return res;
    }
};

template <typename Real, typename Complex, int NumMoments>
struct CentralMomentsFunctor {
    Real mean_u;
    explicit CentralMomentsFunctor(Real mu) : mean_u(mu) {}

    __host__ __device__ MomentTuple<Real, NumMoments> operator()(const Complex& z) const {
        MomentTuple<Real, NumMoments> res;
        Real diff = z.real() - mean_u;
        Real current = diff * diff;

#pragma unroll
        for (int k = 0; k < NumMoments; ++k) {
            res.vals[k] = current;
            current *= diff;
        }
        return res;
    }
};

__device__ inline float wrap_phase(float dphi) {
    float w = fmodf(dphi + (float)M_PI, 2.0f * (float)M_PI);
    if (w < 0.0f) w += 2.0f * (float)M_PI;
    return w - (float)M_PI;
}

__device__ inline double wrap_phase(double dphi) {
    double w = fmod(dphi + M_PI, 2.0 * M_PI);
    if (w < 0.0) w += 2.0 * M_PI;
    return w - M_PI;
}

template <typename Real, typename Complex>
__global__ void sweep_metrics_kernel(const Complex* u, const Complex* u_prev, Real* acc_vu,
                                     Real* acc_vphi, Real* acc_rugosity, std::size_t size,
                                     Real dt) {
    // Dedicate one block per batch dimension
    std::size_t b = blockIdx.x;
    std::size_t offset = b * size;

    extern __shared__ unsigned char shared_mem[];
    Real* s_u = reinterpret_cast<Real*>(shared_mem);
    Real* s_u2 = s_u + blockDim.x;
    Real* s_du = s_u2 + blockDim.x;
    Real* s_dphi = s_du + blockDim.x;

    Real sum_u = Real(0.0), sum_u2 = Real(0.0), sum_du = Real(0.0), sum_dphi = Real(0.0);

    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        Complex curr = u[offset + i];
        Complex prev = u_prev[offset + i];

        Real c_u = curr.real();
        Real du = c_u - prev.real();

        Real c_phi = -curr.imag();
        Real p_phi = -prev.imag();
        Real dphi = c_phi - p_phi;

        Real w_dphi = wrap_phase(dphi);

        sum_u += c_u;
        sum_u2 += c_u * c_u;
        sum_du += du;
        sum_dphi += w_dphi;
    }

    s_u[threadIdx.x] = sum_u;
    s_u2[threadIdx.x] = sum_u2;
    s_du[threadIdx.x] = sum_du;
    s_dphi[threadIdx.x] = sum_dphi;
    __syncthreads();

    // Shared memory reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_u[threadIdx.x] += s_u[threadIdx.x + s];
            s_u2[threadIdx.x] += s_u2[threadIdx.x + s];
            s_du[threadIdx.x] += s_du[threadIdx.x + s];
            s_dphi[threadIdx.x] += s_dphi[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        Real inv_size = Real(1.0) / static_cast<Real>(size);
        Real mean_u = s_u[0] * inv_size;
        Real mean_u2 = s_u2[0] * inv_size;

        // Calculate variance exclusively for the current spatial step
        Real rugosity = mean_u2 - (mean_u * mean_u);
        if (rugosity < Real(0.0)) rugosity = Real(0.0);

        safeAtomicAdd(&acc_rugosity[b], rugosity);
        safeAtomicAdd(&acc_vu[b], (s_du[0] * inv_size) / dt);
        safeAtomicAdd(&acc_vphi[b], (s_dphi[0] * inv_size) / dt);
    }
}

template <typename Real, typename Complex>
__global__ void strobo_step_accum_kernel(const Complex* u, const Complex* u_strobo_prev,
                                         Real* acc_step, Real* acc_step2, std::size_t size,
                                         std::size_t batch) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size * batch) {
        Real du_step = u[j].real() - u_strobo_prev[j].real();
        safeAtomicAdd(&acc_step[j], du_step);
        safeAtomicAdd(&acc_step2[j], du_step * du_step);
    }
}

template <typename Real, typename Complex>
__global__ void batched_sum_kernel(const Complex* u, Real* batch_sums, std::size_t size,
                                   std::size_t batch) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size * batch) {
        std::size_t b = j / size;
        safeAtomicAdd(&batch_sums[b], u[j].real());
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

    static void synchronize(cudaStream_t stream = nullptr) {
        if (stream)
            cudaStreamSynchronize(stream);
        else
            cudaDeviceSynchronize();
    }

    static void evaluate_nonlinear(const Complex* z, Complex* nl_out, std::size_t size,
                                   std::size_t batch, Real alpha, const Real* h0, const Real* ha,
                                   Real omega, Real t, cudaStream_t stream = nullptr) {
        std::size_t total_elements = size * batch;
        if (total_elements == 0) return;
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        nonlinear_kernel<Real, Complex>
            <<<numBlocks, blockSize, 0, stream>>>(z, nl_out, size, batch, alpha, h0, ha, omega, t);
    }

    static void accumulate_sweep_metrics(const Complex* u, const Complex* u_prev, Real* acc_vu,
                                         Real* acc_vphi, Real* acc_rugosity, std::size_t size,
                                         std::size_t batch, Real dt,
                                         cudaStream_t stream = nullptr) {
        if (size == 0 || batch == 0) return;
        int threads = 256;
        int blocks = batch;  // Exactly one block per batch
        std::size_t smem = 4 * threads * sizeof(Real);
        sweep_metrics_kernel<Real, Complex><<<blocks, threads, smem, stream>>>(
            u, u_prev, acc_vu, acc_vphi, acc_rugosity, size, dt);
    }

    static void accumulate_strobo_metrics(const Complex* u, const Complex* u_strobo_prev,
                                          Real* acc_step, Real* acc_step2, std::size_t size,
                                          std::size_t batch, cudaStream_t stream = nullptr) {
        std::size_t total = size * batch;
        if (total == 0) return;
        int blocks = (total + 255) / 256;
        strobo_step_accum_kernel<Real, Complex>
            <<<blocks, 256, 0, stream>>>(u, u_strobo_prev, acc_step, acc_step2, size, batch);
    }

    static void step_linear(Complex* z_hat, const Complex* nl_hat, const Complex* prop_z,
                            const Complex* prop_nl, std::size_t grid_size,
                            std::size_t total_elements, cudaStream_t stream = nullptr) {
        if (total_elements == 0) return;
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;

        linear_kernel<Real, Complex><<<numBlocks, blockSize, 0, stream>>>(
            z_hat, nl_hat, prop_z, prop_nl, grid_size, total_elements);
    }

    template <typename T>
    static void copy(const T* src, T* dst, std::size_t size,
                     cudaStream_t stream = nullptr) {
        cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    static Real compute_diff_sq(const Complex* u1, const Complex* u2, std::size_t size,
                                cudaStream_t stream = nullptr) {
        thrust::device_ptr<const Complex> ptr1(u1);
        thrust::device_ptr<const Complex> ptr2(u2);

        auto start = thrust::make_zip_iterator(thrust::make_tuple(ptr1, ptr2));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(ptr1 + size, ptr2 + size));

        DiffSqFunctor<Real, Complex> functor;
        return thrust::transform_reduce(thrust::cuda::par.on(stream), start, end, functor,
                                        Real(0.0), thrust::plus<Real>());
    }

    static ObsTuple<Real> compute_observables(const Complex* z, std::size_t size,
                                              cudaStream_t stream = nullptr) {
        thrust::device_ptr<const Complex> ptr(z);

        Pass1Functor<Real, Complex> p1_op;
        Pass1Reduce<Real> p1_red;
        Pass1Tuple<Real> p1_init = {Real(0.0), Real(0.0)};

        Pass1Tuple<Real> p1_res = thrust::transform_reduce(thrust::cuda::par.on(stream), ptr,
                                                           ptr + size, p1_op, p1_init, p1_red);

        Real mean_u = p1_res.u / static_cast<Real>(size);
        Pass2Functor<Real, Complex> p2_op(mean_u);

        Real var_sum = thrust::transform_reduce(thrust::cuda::par.on(stream), ptr, ptr + size,
                                                p2_op, Real(0.0), thrust::plus<Real>());

        return {p1_res.sin2phi, p1_res.u, var_sum};
    }

    static void accumulate_power_spectrum(const Complex* u_hat, Real* ps_accum, std::size_t size,
                                          cudaStream_t stream = nullptr) {
        if (size == 0) return;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        accumulate_ps_u_kernel<Real, Complex>
            <<<numBlocks, blockSize, 0, stream>>>(u_hat, ps_accum, size);
    }

    static Real compute_spectral_entropy(const Complex* u_hat, std::size_t size,
                                         cudaStream_t stream = nullptr) {
        auto count_it = thrust::make_counting_iterator<std::size_t>(0);

        UPowerSpectrumOp<Real, Complex> ps_op{u_hat, size};

        Real sum_S =
            thrust::transform_reduce(thrust::cuda::par.on(stream), count_it + 1, count_it + size,
                                     ps_op, Real(0.0), thrust::plus<Real>());

        if (sum_S <= Real(0.0)) return Real(0.0);

        UEntropyOp<Real, Complex> ent_op{u_hat, size, sum_S};
        Real entropy =
            thrust::transform_reduce(thrust::cuda::par.on(stream), count_it + 1, count_it + size,
                                     ent_op, Real(0.0), thrust::plus<Real>());

        return entropy / std::log(static_cast<Real>(size - 1));
    }

    static void fill_zero(Real* vec, std::size_t size, cudaStream_t stream = nullptr) {
        thrust::device_ptr<Real> ptr(vec);
        thrust::fill(thrust::cuda::par.on(stream), ptr, ptr + size, Real(0.0));
    }

    static std::vector<Real> download_array(const Real* vec, std::size_t size,
                                            cudaStream_t stream = nullptr) {
        std::vector<Real> host_vec(size);
        cudaMemcpyAsync(host_vec.data(), vec, size * sizeof(Real), cudaMemcpyDeviceToHost, stream);
        if (stream) cudaStreamSynchronize(stream);
        return host_vec;
    }

    static std::vector<Complex> download_array(const Complex* vec, std::size_t size,
                                               cudaStream_t stream = nullptr) {
        std::vector<Complex> host_vec(size);
        cudaMemcpyAsync(host_vec.data(), vec, size * sizeof(Complex), cudaMemcpyDeviceToHost,
                        stream);
        if (stream) cudaStreamSynchronize(stream);
        return host_vec;
    }

    static void accumulate_height_histogram(const Complex* z, std::size_t z_size, Real mean_u,
                                            Real sigma_u, unsigned long long* hist,
                                            std::size_t hist_size, Real min_val, Real max_val,
                                            cudaStream_t stream = nullptr) {
        if (z_size == 0) return;
        int blockSize = 256;
        int numBlocks = (z_size + blockSize - 1) / blockSize;
        accumulate_hist_kernel<Real, Complex><<<numBlocks, blockSize, 0, stream>>>(
            z, hist, z_size, hist_size, mean_u, sigma_u, min_val, max_val);
    }

    static void fill_zero_ull(unsigned long long* vec, std::size_t size,
                              cudaStream_t stream = nullptr) {
        cudaMemsetAsync(vec, 0, size * sizeof(unsigned long long), stream);
    }

    static std::vector<unsigned long long> download_array_ull(const unsigned long long* vec,
                                                              std::size_t size,
                                                              cudaStream_t stream = nullptr) {
        std::vector<unsigned long long> host_vec(size);
        cudaMemcpyAsync(host_vec.data(), vec, size * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, stream);
        if (stream) cudaStreamSynchronize(stream);
        return host_vec;
    }

    static void record_observables_async(const Complex* z, std::size_t size, Real* d_mean_u,
                                         Real* d_mean_u2, Real* d_mean_sin2phi,
                                         std::size_t step_idx, cudaStream_t stream = nullptr) {
        if (size == 0) return;
        int blockSize = 256;
        int numBlocks = std::min(1024, static_cast<int>((size + blockSize - 1) / blockSize));
        std::size_t sharedMemSize = 3 * blockSize * sizeof(Real);

        record_observables_kernel<Real, Complex><<<numBlocks, blockSize, sharedMemSize, stream>>>(
            z, size, d_mean_u, d_mean_u2, d_mean_sin2phi, step_idx);
    }

    template <int K_MAX>
    static std::vector<Real> compute_central_moments_impl(const Complex* z, std::size_t size,
                                                          Real mean_u,
                                                          cudaStream_t stream = nullptr) {
        constexpr int NUM_MOMENTS = K_MAX - 1;
        thrust::device_ptr<const Complex> ptr(z);

        CentralMomentsFunctor<Real, Complex, NUM_MOMENTS> op(mean_u);
        MomentTuple<Real, NUM_MOMENTS> init;

        MomentTuple<Real, NUM_MOMENTS> res =
            thrust::transform_reduce(thrust::cuda::par.on(stream), ptr, ptr + size, op, init,
                                     thrust::plus<MomentTuple<Real, NUM_MOMENTS>>());

        std::vector<Real> moments(K_MAX + 1, Real(0.0));
        moments[0] = Real(1.0);
        moments[1] = Real(0.0);

        Real invN = Real(1.0) / static_cast<Real>(size);
        for (int k = 2; k <= K_MAX; ++k) {
            moments[k] = res.vals[k - 2] * invN;
        }
        return moments;
    }

    static std::vector<Real> compute_central_moments(const Complex* z, std::size_t size, int n_max,
                                                     cudaStream_t stream = nullptr) {
        ObsTuple<Real> obs = compute_observables(z, size, stream);
        Real mean_u = obs.u / static_cast<Real>(size);

        if (n_max <= 1)
            return {Real(1.0), Real(0.0)};
        else if (n_max == 2)
            return compute_central_moments_impl<2>(z, size, mean_u, stream);
        else if (n_max == 3)
            return compute_central_moments_impl<3>(z, size, mean_u, stream);
        else if (n_max == 4)
            return compute_central_moments_impl<4>(z, size, mean_u, stream);
        else if (n_max == 5)
            return compute_central_moments_impl<5>(z, size, mean_u, stream);
        else if (n_max == 6)
            return compute_central_moments_impl<6>(z, size, mean_u, stream);
        else if (n_max == 7)
            return compute_central_moments_impl<7>(z, size, mean_u, stream);
        else
            return compute_central_moments_impl<8>(z, size, mean_u, stream);
    }

    static std::vector<Real> compute_cumulants(Real mean_u,
                                               const std::vector<Real>& central_moments) {
        int n_max = static_cast<int>(central_moments.size()) - 1;
        std::vector<Real> cumulants(n_max + 1, Real(0.0));

        if (n_max >= 1) cumulants[1] = mean_u;

        for (int n = 2; n <= n_max; ++n) {
            Real sum = Real(0.0);
            for (int k = 1; k <= n - 1; ++k) {
                Real kc_k = (k == 1) ? Real(0.0) : cumulants[k];
                sum +=
                    static_cast<Real>(binomial_coeff(n - 1, k - 1)) * kc_k * central_moments[n - k];
            }
            cumulants[n] = central_moments[n] - sum;
        }

        return cumulants;
    }
};

}  // namespace dw
