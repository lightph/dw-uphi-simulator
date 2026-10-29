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

// Constant memory for fast broadcast reading on the GPU
__constant__ real_t d_cos_t0[STEPS_PER_PERIOD];
__constant__ real_t d_cos_thalf[STEPS_PER_PERIOD];

struct Result {
    real_t h0;
    real_t ha;
    real_t omega;
    real_t psi_eq;
    real_t v_psi;
    real_t v_q;
    real_t a0_cos;
    bool converged;
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

__global__ void rigid_wall_sweep_kernel(Result *results, int num_h0, int num_ha,
                                        real_t h0_min, real_t h0_max,
                                        real_t ha_min, real_t ha_max,
                                        real_t omega_val, real_t alpha, real_t dt,
                                        int transient_periods, int max_check_periods,
                                        int eval_periods, real_t tol,
                                        long long offset, long long current_chunk_size) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_chunk_size) return;

    long long flat_index = offset + idx;
    int j = flat_index % num_ha;
    int i = flat_index / num_ha;

    real_t h0 = (num_h0 > 1) ? h0_min + i * (h0_max - h0_min) / (num_h0 - 1) : h0_min;
    real_t ha = (num_ha > 1) ? ha_min + j * (ha_max - ha_min) / (num_ha - 1) : ha_min;

    Result res;
    res.h0 = h0;
    res.ha = ha;
    res.omega = omega_val;
    res.psi_eq = 0.0;
    res.v_psi = 0.0;
    res.v_q = 0.0;
    res.a0_cos = 0.0;
    res.converged = false;

    if (omega_val < 1e-8) {
        results[flat_index] = res;
        return;
    }

    real_t A = (alpha * h0) / omega_val;
    real_t B = alpha / omega_val;
    real_t C = (alpha * ha) / omega_val;

    real_t psi = 0.0;

    // Transient phase
    for (int p = 0; p < transient_periods; ++p) {
        for (int s = 0; s < STEPS_PER_PERIOD; ++s) {
            psi = rk4_step_device(psi, A, B, C, s, dt);
        }
    }

    // Limit cycle check
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

    res.psi_eq = prev_strobe;
    res.converged = converged;

    // Evaluation phase
    real_t psi_start = psi;
    real_t a0_acc = 0.0;
    for (int p = 0; p < eval_periods; ++p) {
        for (int s = 0; s < STEPS_PER_PERIOD; ++s) {
            a0_acc += cos(psi);
            psi = rk4_step_device(psi, A, B, C, s, dt);
        }
    }

    real_t total_tau = (real_t)eval_periods * (real_t)TWO_PI_VAL;
    res.v_psi = (psi - psi_start) / total_tau;
    res.v_q = ((alpha * alpha + (real_t)1.0) / ((real_t)2.0 * omega_val)) * h0 - ((real_t)1.0 / ((real_t)2.0 * alpha)) * res.v_psi;
    res.a0_cos = a0_acc / ((real_t)eval_periods * STEPS_PER_PERIOD);

    results[flat_index] = res;
}

int main(int argc, char** argv) {
    real_t alpha = 0.27;
    real_t omega_val = 1.0;
    real_t h0_min = 0.0, h0_max = 10.0;
    int num_h0 = 1000;
    real_t ha_min = 0.0, ha_max = 20.0;
    int num_ha = 1000;
    std::string out_file = "rigid_wall_gpu_results.bin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--alpha" && i + 1 < argc) alpha = std::stod(argv[++i]);
        else if (arg == "--omega" && i + 1 < argc) omega_val = std::stod(argv[++i]);
        else if (arg == "--h0-min" && i + 1 < argc) h0_min = std::stod(argv[++i]);
        else if (arg == "--h0-max" && i + 1 < argc) h0_max = std::stod(argv[++i]);
        else if (arg == "--num-h0" && i + 1 < argc) num_h0 = std::stoi(argv[++i]);
        else if (arg == "--ha-min" && i + 1 < argc) ha_min = std::stod(argv[++i]);
        else if (arg == "--ha-max" && i + 1 < argc) ha_max = std::stod(argv[++i]);
        else if (arg == "--num-ha" && i + 1 < argc) num_ha = std::stoi(argv[++i]);
        else if (arg == "-o" || arg == "--output") { if (i + 1 < argc) out_file = argv[++i]; }
    }

    const real_t dt = TWO_PI_VAL / STEPS_PER_PERIOD;
    const int transient_periods = 300, max_check_periods = 200, eval_periods = 200;
    const real_t tol = 1e-6;

    long long total_combinations = (long long)num_h0 * num_ha;
    size_t bytes = total_combinations * sizeof(Result);

    std::cout << "========================================\n";
    std::cout << "Starting GPU Parameter Sweep for Rigid Wall ODE\n";
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
    const int num_chunks = 10;
    long long chunk_size_base = total_combinations / num_chunks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        long long offset = chunk * chunk_size_base;
        long long current_chunk_size = (chunk == num_chunks - 1) ? (total_combinations - offset) : chunk_size_base;
        int blocksPerGrid = (current_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

        rigid_wall_sweep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_results, num_h0, num_ha, h0_min, h0_max, ha_min, ha_max,
            omega_val, alpha, dt, transient_periods, max_check_periods,
            eval_periods, tol, offset, current_chunk_size);

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

    std::cout << "Results saved to: " << out_file << " (" << bytes / (1024 * 1024) << " MB)\n";
    std::cout << "Done.\n";
    return 0;
}

