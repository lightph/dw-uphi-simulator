#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cpu_backend.hpp"
#include "domain_wall.hpp"
#include "fftw_precision.hpp"

#ifdef HAS_CUDA
#include "cuda_precision.cuh"
#include "gpu_backend.cuh"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename Backend>
void run_time_evolution(std::size_t size, double L, double alpha, double h0, double ha,
                        double omega, double dt, int max_power, double eps, unsigned int seed,
                        const std::string& output_prefix) {
    using Real = typename Backend::PrecisionType::Real;
    using VectorReal = typename Backend::template Vector<Real>;

    std::string summary_file = output_prefix + "_time_summary.txt";
    std::ofstream out_summary(summary_file);
    out_summary << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_summary << "total_steps var_u u_dot phi_dot spectral_entropy\n";

    unsigned long long acc_steps_max = 10000;
    if (std::abs(omega) > 1e-7) {
        double period = 2.0 * M_PI / std::abs(omega);
        acc_steps_max = static_cast<unsigned long long>(std::round(100.0 * period / dt));
    }

    std::cout << "Max accumulation steps (100 periods cap): " << acc_steps_max << "\n\n";

    dw::DomainWall<Backend> sim(size, L, alpha, h0, ha, omega, dt, eps, seed);
    Real current_time = 0.0;
    Backend::synchronize();

    unsigned long long total_steps_done = 0;

    for (int n = 0; n <= max_power; ++n) {
        unsigned long long target_steps = 1ULL << n;
        unsigned long long batch_steps = target_steps - total_steps_done;

        // The window of time to accumulate is the minimum between the current batch size and 100
        // periods
        unsigned long long acc_steps = std::min(batch_steps, acc_steps_max);
        unsigned long long transient_batch_steps = batch_steps - acc_steps;

        std::cout << "Target total steps: " << target_steps << " | Batch steps: " << batch_steps
                  << " | Accumulating over last: " << acc_steps << "\n";

        // Advance through the non-accumulated portion of the batch
        for (unsigned long long t = 0; t < transient_batch_steps; ++t) {
            sim.step(current_time);
        }
        Backend::synchronize();

        Real sum_var_u = 0.0;
        Real sum_u_dot = 0.0;
        Real sum_phi_dot = 0.0;
        Real sum_entropy = 0.0;

        VectorReal ps_accum(size);
        Backend::fill_zero(ps_accum);

        std::vector<Real> u_dot_series(acc_steps);

        // Advance through the accumulation window at the end of the batch
        for (unsigned long long t = 0; t < acc_steps; ++t) {
            sim.step(current_time);

            auto obs = Backend::compute_observables(sim.get_state().u);
            Real N_grid = static_cast<Real>(size);

            Real mean_u = obs.u / N_grid;
            Real mean_u2 = obs.u2 / N_grid;
            Real var_u = mean_u2 - (mean_u * mean_u);

            Real mean_sin2phi = obs.sin2phi / N_grid;
            Real h_val = h0 + ha * std::cos(omega * current_time);

            Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
            Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

            u_dot_series[t] = u_dot;

            sum_var_u += var_u;
            sum_u_dot += u_dot;
            sum_phi_dot += phi_dot;
            sum_entropy += Backend::compute_spectral_entropy(sim.get_state().u_hat);
            Backend::accumulate_power_spectrum(sim.get_state().u_hat, ps_accum);
        }

        Real avg_var_u = sum_var_u / acc_steps;
        Real avg_u_dot = sum_u_dot / acc_steps;
        Real avg_phi_dot = sum_phi_dot / acc_steps;
        Real avg_entropy = sum_entropy / acc_steps;

        out_summary << target_steps << " " << avg_var_u << " " << avg_u_dot << " " << avg_phi_dot
                    << " " << avg_entropy << "\n";
        out_summary.flush();

        std::vector<Real> host_ps = Backend::download_array(ps_accum);
        std::string ps_file = output_prefix + "_ps_step_" + std::to_string(target_steps) + ".txt";
        std::ofstream out_ps(ps_file);
        out_ps << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ps << "k power\n";

        double dk = 2.0 * M_PI / L;
        for (std::size_t i = 0; i < size; ++i) {
            long long k_idx =
                (i <= size / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size);
            double k_phys = k_idx * dk;
            out_ps << k_phys << " " << (host_ps[i] / acc_steps) << "\n";
        }
        out_ps.close();
        std::string ts_file =
            output_prefix + "_mean_u_ts_step_" + std::to_string(target_steps) + ".txt";
        std::ofstream out_ts(ts_file);
        out_ts << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ts << "time u_dot\n";
        for (unsigned long long i = 0; i < acc_steps; ++i) {
            out_ts << (i * dt) << " " << u_dot_series[i] << "\n";
        }
        out_ts.close();

        total_steps_done = target_steps;
        std::cout << "Finished step " << target_steps << " and saved to disk.\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> "
                     "<backend> <output_prefix> "
                     "[eps=0.05] [seed=42] [precision=double]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double L = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    double h0 = std::stod(argv[4]);
    double ha = std::stod(argv[5]);
    double omega = std::stod(argv[6]);
    double dt = std::stod(argv[7]);
    int max_power = std::stoi(argv[8]);
    std::string backend = argv[9];
    std::string output_prefix = argv[10];

    double eps = (argc > 11) ? std::stod(argv[11]) : 0.05;
    unsigned int seed = (argc > 12) ? std::stoul(argv[12]) : 42;
    std::string precision = (argc > 13) ? argv[13] : "double";

    if (backend == "cpu") {
        if (precision == "single") {
            run_time_evolution<dw::CpuBackend<dw::FftwSinglePrecision>>(
                size, L, alpha, h0, ha, omega, dt, max_power, eps, seed, output_prefix);
        } else {
            run_time_evolution<dw::CpuBackend<dw::FftwDoublePrecision>>(
                size, L, alpha, h0, ha, omega, dt, max_power, eps, seed, output_prefix);
        }
    } else if (backend == "gpu") {
#ifdef HAS_CUDA
        if (precision == "single") {
            run_time_evolution<dw::GpuBackend<dw::CudaSinglePrecision>>(
                size, L, alpha, h0, ha, omega, dt, max_power, eps, seed, output_prefix);
        } else {
            run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
                size, L, alpha, h0, ha, omega, dt, max_power, eps, seed, output_prefix);
        }
#else
        std::cerr << "Binary compiled without CUDA.\n";
        return 1;
#endif
    } else {
        std::cerr << "Invalid backend.\n";
        return 1;
    }

    return 0;
}