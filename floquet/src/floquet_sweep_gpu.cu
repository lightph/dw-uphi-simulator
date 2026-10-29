#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using real_t = double;

const real_t PI_VAL = 3.14159265358979323846;
const real_t TWO_PI_VAL = 2.0 * PI_VAL;
#define STEPS_PER_PERIOD 1000

__constant__ real_t d_cos_t0[STEPS_PER_PERIOD];
__constant__ real_t d_cos_thalf[STEPS_PER_PERIOD];

struct Result {
    real_t h;
    real_t h_ac;
    real_t omega;
    real_t x;
    real_t max_k_mu;
    real_t max_mu;
    real_t max_k_sigma;
    real_t max_sigma;
    bool converged;
};

struct StabilityMetrics {
    real_t mu;
    real_t sigma;
};

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err)         \
                << "\"\n";                                                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__device__ inline real_t rk4_step_device(real_t psi, real_t A, real_t B, real_t C,
                                         int step_idx, real_t dt) {
    int inext = (step_idx + 1) % STEPS_PER_PERIOD;
    real_t k1 = dt * (A - B * sin(psi) + C * d_cos_t0[step_idx]);
    real_t k2 = dt * (A - B * sin(psi + (real_t)0.5 * k1) + C * d_cos_thalf[step_idx]);
    real_t k3 = dt * (A - B * sin(psi + (real_t)0.5 * k2) + C * d_cos_thalf[step_idx]);
    real_t k4 = dt * (A - B * sin(psi + k3) + C * d_cos_t0[inext]);
    real_t next_psi = psi + (k1 + (real_t)2.0 * k2 + (real_t)2.0 * k3 + k4) / (real_t)6.0;
    return next_psi;
}

__device__ inline real_t circle_distance_device(real_t a, real_t b) {
    real_t diff = fmod(fabs(a - b), (real_t)TWO_PI_VAL);
    return diff > (real_t)PI_VAL ? (real_t)TWO_PI_VAL - diff : diff;
}

__device__ StabilityMetrics eval_stability_device(real_t x_start, real_t A, real_t B, real_t C,
                                                 real_t kappa, real_t omega, real_t alpha,
                                                 real_t dt) {
    real_t kappa2 = (real_t)0.5 * kappa * kappa;
    real_t theta = x_start;
    real_t q1 = 1.0, p1 = 0.0;
    real_t q2 = 0.0, p2 = 1.0;

    for (int s = 0; s < STEPS_PER_PERIOD; ++s) {
        int inext = (s + 1) % STEPS_PER_PERIOD;
        real_t c_t0 = d_cos_t0[s];
        real_t c_thalf = d_cos_thalf[s];
        real_t c_t1 = d_cos_t0[inext];

        real_t cos_th1 = cos(theta);
        real_t term1 = kappa2 + cos_th1;
        real_t k1_th = dt * (A - B * sin(theta) + C * c_t0);
        real_t k1_q1 = dt * (-alpha * kappa2 * q1 + term1 * p1) / omega;
        real_t k1_p1 = dt * (-kappa2 * q1 - alpha * term1 * p1) / omega;
        real_t k1_q2 = dt * (-alpha * kappa2 * q2 + term1 * p2) / omega;
        real_t k1_p2 = dt * (-kappa2 * q2 - alpha * term1 * p2) / omega;

        real_t th2 = theta + (real_t)0.5 * k1_th;
        real_t cos_th2 = cos(th2);
        real_t term2 = kappa2 + cos_th2;
        real_t q1_2 = q1 + (real_t)0.5 * k1_q1;
        real_t p1_2 = p1 + (real_t)0.5 * k1_p1;
        real_t q2_2 = q2 + (real_t)0.5 * k1_q2;
        real_t p2_2 = p2 + (real_t)0.5 * k1_p2;

        real_t k2_th = dt * (A - B * sin(th2) + C * c_thalf);
        real_t k2_q1 = dt * (-alpha * kappa2 * q1_2 + term2 * p1_2) / omega;
        real_t k2_p1 = dt * (-kappa2 * q1_2 - alpha * term2 * p1_2) / omega;
        real_t k2_q2 = dt * (-alpha * kappa2 * q2_2 + term2 * p2_2) / omega;
        real_t k2_p2 = dt * (-kappa2 * q2_2 - alpha * term2 * p2_2) / omega;

        real_t th3 = theta + (real_t)0.5 * k2_th;
        real_t cos_th3 = cos(th3);
        real_t term3 = kappa2 + cos_th3;
        real_t q1_3 = q1 + (real_t)0.5 * k2_q1;
        real_t p1_3 = p1 + (real_t)0.5 * k2_p1;
        real_t q2_3 = q2 + (real_t)0.5 * k2_q2;
        real_t p2_3 = p2 + (real_t)0.5 * k2_p2;

        real_t k3_th = dt * (A - B * sin(th3) + C * c_thalf);
        real_t k3_q1 = dt * (-alpha * kappa2 * q1_3 + term3 * p1_3) / omega;
        real_t k3_p1 = dt * (-kappa2 * q1_3 - alpha * term3 * p1_3) / omega;
        real_t k3_q2 = dt * (-alpha * kappa2 * q2_3 + term3 * p2_3) / omega;
        real_t k3_p2 = dt * (-kappa2 * q2_3 - alpha * term3 * p2_3) / omega;

        real_t th4 = theta + k3_th;
        real_t cos_th4 = cos(th4);
        real_t term4 = kappa2 + cos_th4;
        real_t q1_4 = q1 + k3_q1;
        real_t p1_4 = p1 + k3_p1;
        real_t q2_4 = q2 + k3_q2;
        real_t p2_4 = p2 + k3_p2;

        real_t k4_th = dt * (A - B * sin(th4) + C * c_t1);
        real_t k4_q1 = dt * (-alpha * kappa2 * q1_4 + term4 * p1_4) / omega;
        real_t k4_p1 = dt * (-kappa2 * q1_4 - alpha * term4 * p1_4) / omega;
        real_t k4_q2 = dt * (-alpha * kappa2 * q2_4 + term4 * p2_4) / omega;
        real_t k4_p2 = dt * (-kappa2 * q2_4 - alpha * term4 * p2_4) / omega;

        theta += (k1_th + (real_t)2.0 * k2_th + (real_t)2.0 * k3_th + k4_th) / (real_t)6.0;
        q1 += (k1_q1 + (real_t)2.0 * k2_q1 + (real_t)2.0 * k3_q1 + k4_q1) / (real_t)6.0;
        p1 += (k1_p1 + (real_t)2.0 * k2_p1 + (real_t)2.0 * k3_p1 + k4_q1) / (real_t)6.0;
        q2 += (k1_q2 + (real_t)2.0 * k2_q2 + (real_t)2.0 * k3_q2 + k4_q2) / (real_t)6.0;
        p2 += (k1_p2 + (real_t)2.0 * k2_p2 + (real_t)2.0 * k3_p2 + k4_p2) / (real_t)6.0;
    }

    StabilityMetrics res;
    real_t T_tr = q1 + p2;
    real_t D = q1 * p2 - q2 * p1;
    real_t disc_mu = T_tr * T_tr - (real_t)4.0 * D;

    if (disc_mu >= 0.0) {
        real_t root = sqrt(disc_mu);
        real_t e1 = fabs((T_tr + root) * (real_t)0.5);
        real_t e2 = fabs((T_tr - root) * (real_t)0.5);
        res.mu = (e1 > e2) ? e1 : e2;
    } else {
        res.mu = sqrt(D > 0 ? D : 0.0);
    }

    real_t S = q1 * q1 + p1 * p1 + q2 * q2 + p2 * p2;
    real_t disc_sigma = S * S - (real_t)4.0 * D * D;
    if (disc_sigma < 0.0) disc_sigma = 0.0;
    res.sigma = sqrt((real_t)0.5 * (S + sqrt(disc_sigma)));
    return res;
}

__global__ void floquet_sweep_kernel(Result *results, int num_h, int num_hac,
                                     real_t h_min, real_t h_max_val,
                                     real_t hac_min, real_t hac_max_val,
                                     real_t omega_val, real_t alpha, real_t dt,
                                     int transient_periods, int max_check_periods,
                                     real_t tol, real_t L,
                                     long long offset, long long current_chunk_size) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_chunk_size) return;

    long long flat_index = offset + idx;
    int j = flat_index % num_hac;
    int i = flat_index / num_hac;

    real_t h = (num_h > 1) ? h_min + i * (h_max_val - h_min) / (num_h - 1) : h_min;
    real_t hac = (num_hac > 1) ? hac_min + j * (hac_max_val - hac_min) / (num_hac - 1) : hac_min;

    Result res;
    res.h = h;
    res.h_ac = hac;
    res.omega = omega_val;
    res.max_mu = 0.0;
    res.max_sigma = 0.0;
    res.max_k_mu = 0.0;
    res.max_k_sigma = 0.0;
    res.x = 0.0;
    res.converged = false;

    if (omega_val < 1e-8) {
        results[flat_index] = res;
        return;
    }

    real_t A = (alpha * h) / omega_val;
    real_t B = alpha / omega_val;
    real_t C = (alpha * hac) / omega_val;

    real_t psi = 0.0;
    for (int p = 0; p < transient_periods; ++p) {
        for (int s = 0; s < STEPS_PER_PERIOD; ++s) {
            psi = rk4_step_device(psi, A, B, C, s, dt);
        }
    }

    real_t prev_strobe = fmod(psi, (real_t)TWO_PI_VAL);
    if (prev_strobe < 0) prev_strobe += (real_t)TWO_PI_VAL;
    bool converged = false;

    for (int p = 0; p < max_check_periods; ++p) {
        for (int s = 0; s < STEPS_PER_PERIOD; ++s) {
            psi = rk4_step_device(psi, A, B, C, s, dt);
        }
        real_t curr_strobe = fmod(psi, (real_t)TWO_PI_VAL);
        if (curr_strobe < 0) curr_strobe += (real_t)TWO_PI_VAL;
        if (circle_distance_device(curr_strobe, prev_strobe) < tol) {
            converged = true;
            prev_strobe = curr_strobe;
            break;
        }
        prev_strobe = curr_strobe;
    }

    res.x = prev_strobe;
    res.converged = converged;

    if (converged) {
        real_t fundamental_k = (real_t)TWO_PI_VAL / L;
        const real_t K_MAX = 5.0;
        long long n_max = (long long)floor(K_MAX / fundamental_k);
        if (n_max < 1) n_max = 1;

        StabilityMetrics fund_metrics = eval_stability_device(
            prev_strobe, A, B, C, fundamental_k, omega_val, alpha, dt);
        res.max_k_mu = fundamental_k;
        res.max_mu = fund_metrics.mu;
        res.max_k_sigma = fundamental_k;
        res.max_sigma = fund_metrics.sigma;

        for (long long n = 2; n <= n_max; ++n) {
            real_t test_k = n * fundamental_k;
            StabilityMetrics metrics = eval_stability_device(
                prev_strobe, A, B, C, test_k, omega_val, alpha, dt);

            if (metrics.mu > res.max_mu) {
                res.max_mu = metrics.mu;
                res.max_k_mu = test_k;
            }
            if (metrics.sigma > res.max_sigma) {
                res.max_sigma = metrics.sigma;
                res.max_k_sigma = test_k;
            }
        }
    }

    results[flat_index] = res;
}

int main(int argc, char** argv) {
    real_t alpha = 0.27;
    real_t omega_val = 1.0;
    real_t L = 1024.0;
    real_t h0_min = 0.0, h0_max = 10.0;
    int num_h0 = 1000;
    real_t ha_min = 0.0, ha_max = 20.0;
    int num_ha = 1000;
    std::string out_file = "floquet_sweep_gpu.bin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--alpha" && i + 1 < argc) alpha = std::stod(argv[++i]);
        else if (arg == "--omega" && i + 1 < argc) omega_val = std::stod(argv[++i]);
        else if (arg == "-L" && i + 1 < argc) L = std::stod(argv[++i]);
        else if (arg == "--h0-min" && i + 1 < argc) h0_min = std::stod(argv[++i]);
        else if (arg == "--h0-max" && i + 1 < argc) h0_max = std::stod(argv[++i]);
        else if (arg == "--num-h0" && i + 1 < argc) num_h0 = std::stoi(argv[++i]);
        else if (arg == "--ha-min" && i + 1 < argc) ha_min = std::stod(argv[++i]);
        else if (arg == "--ha-max" && i + 1 < argc) ha_max = std::stod(argv[++i]);
        else if (arg == "--num-ha" && i + 1 < argc) num_ha = std::stoi(argv[++i]);
        else if (arg == "-o" || arg == "--output") { if (i + 1 < argc) out_file = argv[++i]; }
    }

    const real_t dt = TWO_PI_VAL / STEPS_PER_PERIOD;
    const int transient_periods = 300, max_check_periods = 200;
    const real_t tol = 1e-6;

    long long total_combinations = (long long)num_h0 * num_ha;
    size_t bytes = total_combinations * sizeof(Result);

    std::cout << "========================================\n";
    std::cout << "Starting GPU Floquet Stability Sweep for L = " << L << "\n";
    std::cout << "Combinations: " << total_combinations << " (" << num_h0 << " x " << num_ha << ")\n";
    std::cout << "Omega = " << omega_val << ", Alpha = " << alpha << "\n";
    std::cout << "========================================\n";

    std::vector<real_t> h_cos_t0(STEPS_PER_PERIOD), h_cos_thalf(STEPS_PER_PERIOD);
    for (int i = 0; i < STEPS_PER_PERIOD; ++i) {
        h_cos_t0[i] = std::cos(i * dt);
        h_cos_thalf[i] = std::cos((i + 0.5) * dt);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_cos_t0, h_cos_t0.data(), STEPS_PER_PERIOD * sizeof(real_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cos_thalf, h_cos_thalf.data(), STEPS_PER_PERIOD * sizeof(real_t)));

    Result *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, bytes));
    std::vector<Result> h_results(total_combinations);

    int threadsPerBlock = 256;
    const int num_chunks = 20;
    long long chunk_size_base = total_combinations / num_chunks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        long long offset = chunk * chunk_size_base;
        long long current_chunk_size = (chunk == num_chunks - 1) ? (total_combinations - offset) : chunk_size_base;
        int blocksPerGrid = (current_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

        floquet_sweep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_results, num_h0, num_ha, h0_min, h0_max, ha_min, ha_max,
            omega_val, alpha, dt, transient_periods, max_check_periods,
            tol, L, offset, current_chunk_size);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(2) << elapsed << " s\n";

    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_results));

    std::ofstream outfile(out_file, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(h_results.data()), bytes);
    outfile.close();

    std::cout << "Results saved to: " << out_file << "\n";
    std::cout << "Done.\n";
    return 0;
}

