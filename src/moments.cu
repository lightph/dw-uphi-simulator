#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cpu_backend.hpp"
#include "fftw_precision.hpp"
#ifdef HAS_CUDA
#include "cuda_precision.cuh"
#include "gpu_backend.cuh"
#endif
#include "domain_wall.cuh"
#include "sim_utils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

#ifdef HAS_CUDA
// Batched reduction kernels to evaluate moments for all batches simultaneously
template <typename Real, typename Complex>
__global__ void batched_mean_kernel(const Complex* u, Real* batch_sums, std::size_t size) {
    extern __shared__ unsigned char shared_mem[];
    Real* s_sum = reinterpret_cast<Real*>(shared_mem);

    std::size_t b = blockIdx.y;
    std::size_t offset = b * size;

    Real sum = Real(0.0);
    std::size_t tid = threadIdx.x;
    for (std::size_t i = blockIdx.x * blockDim.x + tid; i < size; i += blockDim.x * gridDim.x) {
        sum += u[offset + i].real();
    }
    s_sum[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dw::safeAtomicAdd(&batch_sums[b], s_sum[0]);
    }
}

template <typename Real>
__global__ void normalize_mean_kernel(Real* batch_sums, std::size_t size, std::size_t batch) {
    std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        batch_sums[b] /= static_cast<Real>(size);
    }
}

template <typename Real, typename Complex>
__global__ void batched_moments_kernel(const Complex* u, const Real* batch_means, Real* mu2,
                                       Real* mu3, Real* mu4, Real* mu5, std::size_t size) {
    extern __shared__ unsigned char shared_mem[];
    Real* s_mu2 = reinterpret_cast<Real*>(shared_mem);
    Real* s_mu3 = s_mu2 + blockDim.x;
    Real* s_mu4 = s_mu3 + blockDim.x;
    Real* s_mu5 = s_mu4 + blockDim.x;

    std::size_t b = blockIdx.y;
    std::size_t offset = b * size;
    Real mean = batch_means[b];

    Real m2 = Real(0.0), m3 = Real(0.0), m4 = Real(0.0), m5 = Real(0.0);
    std::size_t tid = threadIdx.x;

    for (std::size_t i = blockIdx.x * blockDim.x + tid; i < size; i += blockDim.x * gridDim.x) {
        Real diff = u[offset + i].real() - mean;
        Real d2 = diff * diff;
        Real d3 = d2 * diff;
        m2 += d2;
        m3 += d3;
        m4 += d2 * d2;
        m5 += d3 * d2;
    }
    s_mu2[tid] = m2;
    s_mu3[tid] = m3;
    s_mu4[tid] = m4;
    s_mu5[tid] = m5;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mu2[tid] += s_mu2[tid + s];
            s_mu3[tid] += s_mu3[tid + s];
            s_mu4[tid] += s_mu4[tid + s];
            s_mu5[tid] += s_mu5[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dw::safeAtomicAdd(&mu2[b], s_mu2[0]);
        dw::safeAtomicAdd(&mu3[b], s_mu3[0]);
        dw::safeAtomicAdd(&mu4[b], s_mu4[0]);
        dw::safeAtomicAdd(&mu5[b], s_mu5[0]);
    }
}
#endif

// Helper struct to dispatch batched moments evaluation based on the backend
template <typename Backend>
struct BatchedMomentsEvaluator {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;

    static void evaluate(const Complex* u, Real* d_batch_means, Real* d_mu2, Real* d_mu3,
                         Real* d_mu4, Real* d_mu5, std::size_t size, std::size_t batch) {
        // CPU fallback for batched moments
        std::size_t total_elements = size * batch;
        auto host_u = Backend::download_array(u, total_elements);
        std::vector<Real> h_batch_means(batch, 0.0);
        std::vector<Real> h_mu2(batch, 0.0), h_mu3(batch, 0.0), h_mu4(batch, 0.0), h_mu5(batch, 0.0);
        
        for (std::size_t b = 0; b < batch; ++b) {
            Real sum = 0.0;
            for (std::size_t i = 0; i < size; ++i) {
                sum += host_u[b * size + i].real();
            }
            h_batch_means[b] = sum / static_cast<Real>(size);
            
            Real m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0;
            for (std::size_t i = 0; i < size; ++i) {
                Real diff = host_u[b * size + i].real() - h_batch_means[b];
                Real d2 = diff * diff;
                Real d3 = d2 * diff;
                m2 += d2;
                m3 += d3;
                m4 += d2 * d2;
                m5 += d3 * d2;
            }
            h_mu2[b] = m2;
            h_mu3[b] = m3;
            h_mu4[b] = m4;
            h_mu5[b] = m5;
        }
        Backend::copy(h_batch_means.data(), d_batch_means, batch);
        Backend::copy(h_mu2.data(), d_mu2, batch);
        Backend::copy(h_mu3.data(), d_mu3, batch);
        Backend::copy(h_mu4.data(), d_mu4, batch);
        Backend::copy(h_mu5.data(), d_mu5, batch);
    }
};

#ifdef HAS_CUDA
template <typename Precision>
struct BatchedMomentsEvaluator<dw::GpuBackend<Precision>> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;

    static void evaluate(const Complex* u, Real* d_batch_means, Real* d_mu2, Real* d_mu3,
                         Real* d_mu4, Real* d_mu5, std::size_t size, std::size_t batch) {
        int blocks_per_batch = std::min<int>(256, static_cast<int>((size + 255) / 256));
        dim3 grid(blocks_per_batch, batch);
        int threads = 256;
        std::size_t smem_mean = threads * sizeof(Real);
        std::size_t smem_moments = 4 * threads * sizeof(Real);

        batched_mean_kernel<Real, Complex>
            <<<grid, threads, smem_mean>>>(u, d_batch_means, size);

        normalize_mean_kernel<Real>
            <<<(batch + 255) / 256, 256>>>(d_batch_means, size, batch);

        batched_moments_kernel<Real, Complex>
            <<<grid, threads, smem_moments>>>(u, d_batch_means, d_mu2, d_mu3, d_mu4, d_mu5, size);
    }
};
#endif

}  // namespace

template <typename Backend>
void run_time_evolution(std::size_t size, std::size_t batch, double L, double alpha, double h0,
                        double ha, double omega, double dt, double total_time, double start_time,
                        int num_checkpoints, int num_sq_checkpoints, double eps, unsigned int seed,
                        const std::string& output_prefix) {
    using Real = typename Backend::PrecisionType::Real;
    using VectorReal = typename Backend::template Vector<Real>;

    unsigned long long total_steps = static_cast<unsigned long long>(std::round(total_time / dt));
    unsigned long long start_step = static_cast<unsigned long long>(std::round(start_time / dt));
    if (start_step < 1) start_step = 1;

    std::cout << "Total simulation steps: " << total_steps << "\n"
              << "Recording starts at step: " << start_step << "\n"
              << "Requested moment checkpoints: " << num_checkpoints << "\n"
              << "Requested S(q,t) checkpoints: " << num_sq_checkpoints << "\n"
              << "Simulating " << batch << " concurrent systems\n"
              << std::endl;

    auto generate_checkpoints = [&](int num_cp) {
        std::vector<unsigned long long> cps = dw::utils::generate_log_checkpoints(total_steps, num_cp, start_step);
        // Enforce a minimum interval of 50 steps to prevent synchronization stalls
        std::vector<unsigned long long> filtered_cps;
        unsigned long long last_cp = 0;
        for (auto cp : cps) {
            if (cp - last_cp >= 50) {
                filtered_cps.push_back(cp);
                last_cp = cp;
            }
        }
        return filtered_cps;
    };

    std::vector<unsigned long long> checkpoints = generate_checkpoints(num_checkpoints);
    std::vector<unsigned long long> sq_checkpoints = generate_checkpoints(num_sq_checkpoints);

    std::string out_file = output_prefix + "_moments.txt";
    std::ofstream out(out_file);
    out << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out << "time step "
        << "glob_mu2 glob_mu3 glob_mu4 glob_mu5 glob_k2 glob_k3 glob_k4 glob_k5 glob_skew "
           "glob_kurt "
        << "avg_mu2 avg_mu3 avg_mu4 avg_mu5 avg_k2 avg_k3 avg_k4 avg_k5 "
        << "avg_skew avg_kurt "
        << "err_mu2 err_mu3 err_mu4 err_mu5 err_k2 err_k3 err_k4 err_k5 "
        << "err_skew err_kurt\n";

    std::string sq_file = output_prefix + "_S_q_t.txt";
    std::ofstream out_sq(sq_file);
    out_sq << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_sq << "time q S_q\n";

    std::vector<double> h0_batch(batch, h0);
    std::vector<double> ha_batch(batch, ha);
    dw::DomainWall<Backend> sim(size, L, alpha, h0_batch, ha_batch, omega, dt, eps, seed);

    Real current_time = 0.0;
    Backend::synchronize();

    VectorReal d_batch_means(batch);
    VectorReal d_mu2(batch);
    VectorReal d_mu3(batch);
    VectorReal d_mu4(batch);
    VectorReal d_mu5(batch);

    auto next_checkpoint = checkpoints.begin();
    auto next_sq_checkpoint = sq_checkpoints.begin();

    dw::utils::ProgressPrinter progress(total_steps, 100);

    for (unsigned long long step = 1; step <= total_steps; ++step) {
        sim.step(current_time);

        bool save_moments = (next_checkpoint != checkpoints.end() && step == *next_checkpoint);
        bool save_sq = (next_sq_checkpoint != sq_checkpoints.end() && step == *next_sq_checkpoint);

        if (save_moments) {
            // ====================================================================
            // Global Ensemble Statistics
            // ====================================================================
            std::size_t total_elements = size * batch;
            auto glob_obs = Backend::compute_observables(sim.get_state().u.data(), total_elements);
            Real glob_mean = glob_obs.u / static_cast<Real>(total_elements);
            auto glob_mu =
                Backend::compute_central_moments(sim.get_state().u.data(), total_elements, 5);
            auto glob_k = Backend::compute_cumulants(glob_mean, glob_mu);

            Real glob_var = glob_mu[2];
            Real glob_skew =
                (glob_var > Real(0.0)) ? glob_mu[3] / std::pow(glob_var, Real(1.5)) : Real(0.0);
            Real glob_kurt =
                (glob_var > Real(0.0)) ? glob_mu[4] / (glob_var * glob_var) - Real(3.0) : Real(0.0);

            // ====================================================================
            // Sample-Averaged Moments, Derived Statistics, and Standard Errors
            // ====================================================================
            Backend::fill_zero(d_batch_means.data(), batch);
            Backend::fill_zero(d_mu2.data(), batch);
            Backend::fill_zero(d_mu3.data(), batch);
            Backend::fill_zero(d_mu4.data(), batch);
            Backend::fill_zero(d_mu5.data(), batch);

            BatchedMomentsEvaluator<Backend>::evaluate(sim.get_state().u.data(), d_batch_means.data(),
                                                       d_mu2.data(), d_mu3.data(), d_mu4.data(),
                                                       d_mu5.data(), size, batch);

            std::vector<Real> b_mean = Backend::download_array(d_batch_means.data(), batch);
            std::vector<Real> b_mu2 = Backend::download_array(d_mu2.data(), batch);
            std::vector<Real> b_mu3 = Backend::download_array(d_mu3.data(), batch);
            std::vector<Real> b_mu4 = Backend::download_array(d_mu4.data(), batch);
            std::vector<Real> b_mu5 = Backend::download_array(d_mu5.data(), batch);

            std::vector<Real> b_k2(batch), b_k3(batch), b_k4(batch), b_k5(batch);
            std::vector<Real> b_skew(batch), b_kurt(batch);

            std::vector<Real> avg_mu(6, Real(0.0));
            std::vector<Real> avg_k(6, Real(0.0));
            Real avg_skew = Real(0.0);
            Real avg_kurt = Real(0.0);

            Real invN = Real(1.0) / static_cast<Real>(size);

            for (std::size_t b = 0; b < batch; ++b) {
                b_mu2[b] *= invN;
                b_mu3[b] *= invN;
                b_mu4[b] *= invN;
                b_mu5[b] *= invN;

                Real mean_u = b_mean[b];
                std::vector<Real> mu_vec = {Real(1.0), Real(0.0), b_mu2[b],
                                            b_mu3[b],  b_mu4[b],  b_mu5[b]};
                auto k = Backend::compute_cumulants(mean_u, mu_vec);

                b_k2[b] = k[2];
                b_k3[b] = k[3];
                b_k4[b] = k[4];
                b_k5[b] = k[5];

                Real variance = b_mu2[b];
                b_skew[b] =
                    (variance > Real(0.0)) ? b_mu3[b] / std::pow(variance, Real(1.5)) : Real(0.0);
                b_kurt[b] = (variance > Real(0.0)) ? b_mu4[b] / (variance * variance) - Real(3.0)
                                                   : Real(0.0);

                for (int i = 2; i <= 5; ++i) {
                    avg_mu[i] += mu_vec[i];
                    avg_k[i] += k[i];
                }
                avg_skew += b_skew[b];
                avg_kurt += b_kurt[b];
            }

            Real inv_batch = Real(1.0) / static_cast<Real>(batch);
            for (int i = 2; i <= 5; ++i) {
                avg_mu[i] *= inv_batch;
                avg_k[i] *= inv_batch;
            }
            avg_skew *= inv_batch;
            avg_kurt *= inv_batch;

            std::vector<Real> err_mu(6, Real(0.0));
            std::vector<Real> err_k(6, Real(0.0));
            Real err_skew = Real(0.0);
            Real err_kurt = Real(0.0);

            if (batch > 1) {
                Real var_mu2 = 0, var_mu3 = 0, var_mu4 = 0, var_mu5 = 0;
                Real var_k2 = 0, var_k3 = 0, var_k4 = 0, var_k5 = 0;
                Real var_skew = 0, var_kurt = 0;

                for (std::size_t b = 0; b < batch; ++b) {
                    var_mu2 += (b_mu2[b] - avg_mu[2]) * (b_mu2[b] - avg_mu[2]);
                    var_mu3 += (b_mu3[b] - avg_mu[3]) * (b_mu3[b] - avg_mu[3]);
                    var_mu4 += (b_mu4[b] - avg_mu[4]) * (b_mu4[b] - avg_mu[4]);
                    var_mu5 += (b_mu5[b] - avg_mu[5]) * (b_mu5[b] - avg_mu[5]);

                    var_k2 += (b_k2[b] - avg_k[2]) * (b_k2[b] - avg_k[2]);
                    var_k3 += (b_k3[b] - avg_k[3]) * (b_k3[b] - avg_k[3]);
                    var_k4 += (b_k4[b] - avg_k[4]) * (b_k4[b] - avg_k[4]);
                    var_k5 += (b_k5[b] - avg_k[5]) * (b_k5[b] - avg_k[5]);

                    var_skew += (b_skew[b] - avg_skew) * (b_skew[b] - avg_skew);
                    var_kurt += (b_kurt[b] - avg_kurt) * (b_kurt[b] - avg_kurt);
                }

                Real err_factor = Real(1.0) / static_cast<Real>(batch * (batch - 1));
                err_mu[2] = std::sqrt(var_mu2 * err_factor);
                err_mu[3] = std::sqrt(var_mu3 * err_factor);
                err_mu[4] = std::sqrt(var_mu4 * err_factor);
                err_mu[5] = std::sqrt(var_mu5 * err_factor);

                err_k[2] = std::sqrt(var_k2 * err_factor);
                err_k[3] = std::sqrt(var_k3 * err_factor);
                err_k[4] = std::sqrt(var_k4 * err_factor);
                err_k[5] = std::sqrt(var_k5 * err_factor);

                err_skew = std::sqrt(var_skew * err_factor);
                err_kurt = std::sqrt(var_kurt * err_factor);
            }

            out << current_time << " " << step << " " << glob_mu[2] << " " << glob_mu[3] << " "
                << glob_mu[4] << " " << glob_mu[5] << " " << glob_k[2] << " " << glob_k[3] << " "
                << glob_k[4] << " " << glob_k[5] << " " << glob_skew << " " << glob_kurt << " "
                << avg_mu[2] << " " << avg_mu[3] << " " << avg_mu[4] << " " << avg_mu[5] << " "
                << avg_k[2] << " " << avg_k[3] << " " << avg_k[4] << " " << avg_k[5] << " "
                << avg_skew << " " << avg_kurt << " " << err_mu[2] << " " << err_mu[3] << " "
                << err_mu[4] << " " << err_mu[5] << " " << err_k[2] << " " << err_k[3] << " "
                << err_k[4] << " " << err_k[5] << " " << err_skew << " " << err_kurt << "\n";
            out.flush();
            ++next_checkpoint;
        }

        if (save_sq) {
            // ====================================================================
            // Structure Factor S(q, t)
            // ====================================================================
            VectorReal ps_accum(size);
            Backend::fill_zero(ps_accum.data(), size);

            for (std::size_t b = 0; b < batch; ++b) {
                Backend::accumulate_power_spectrum(sim.get_state().get_u_hat(b), ps_accum.data(),
                                                   size);
            }

            auto host_ps = Backend::download_array(ps_accum.data(), size);
            double dk = 2.0 * M_PI / L;

            Real norm_factor = static_cast<Real>(L) / static_cast<Real>(batch);

            for (std::size_t i = 1; i <= size / 2; ++i) {
                double q = i * dk;
                out_sq << current_time << " " << q << " " << host_ps[i] * norm_factor << "\n";
            }
            out_sq.flush();
            ++next_sq_checkpoint;
        }

        progress.print("Simulating", step);
    }

    out.close();
    out_sq.close();
    std::cout << "Simulation completed successfully. Moments saved to " << out_file
              << " and Structure Factor saved to " << sq_file << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 14) {
        std::cerr << "Usage: " << argv[0] << " <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> "
                  << "<total_time> <start_time> <num_checkpoints> <num_sq_checkpoints> "
                  << "<output_prefix> [eps=0.05] [seed=42] [backend=gpu]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    std::size_t batch = std::stoull(argv[2]);
    double L = std::stod(argv[3]);
    double alpha = std::stod(argv[4]);
    double h0 = std::stod(argv[5]);
    double ha = std::stod(argv[6]);
    double omega = std::stod(argv[7]);
    double dt = std::stod(argv[8]);

    double total_time = std::stod(argv[9]);
    double start_time = std::stod(argv[10]);
    int num_checkpoints = std::stoi(argv[11]);
    int num_sq_checkpoints = std::stoi(argv[12]);
    std::string output_prefix = argv[13];

    double eps = (argc > 14) ? std::stod(argv[14]) : 0.05;
    unsigned int seed = (argc > 15) ? std::stoul(argv[15]) : 42;
    std::string backend = (argc > 16) ? argv[16] : "gpu";

    if (backend == "cpu") {
        run_time_evolution<dw::CpuBackend<dw::FftwDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, total_time, start_time, num_checkpoints,
            num_sq_checkpoints, eps, seed, output_prefix);
    } else {
#ifdef HAS_CUDA
        run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, total_time, start_time, num_checkpoints,
            num_sq_checkpoints, eps, seed, output_prefix);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
