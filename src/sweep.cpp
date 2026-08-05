#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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
void run_h0_sweep(std::size_t size, double L, double alpha, double ha, double omega, double dt,
                  double h0_start, double h0_end, int h0_steps, double eps, unsigned int seed,
                  const std::string& output_prefix, bool randomize_sweep) {
    using Real = typename Backend::PrecisionType::Real;
    using VectorReal = typename Backend::template Vector<Real>;

    std::string summary_file = output_prefix + "_summary.txt";
    std::ofstream out_summary(summary_file);
    out_summary << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_summary << "h0 var_u u_dot phi_dot spectral_entropy\n";

    double h0_step_size = (h0_steps > 1) ? (h0_end - h0_start) / (h0_steps - 1) : 0.0;

    std::vector<double> h0_vals(h0_steps);
    for (int step = 0; step < h0_steps; ++step) {
        h0_vals[step] = h0_start + step * h0_step_size;
    }

    if (randomize_sweep) {
        std::mt19937 g(seed);
        std::shuffle(h0_vals.begin(), h0_vals.end(), g);
    }

    unsigned long long transient_steps = 1ULL << 22;

    unsigned long long acc_steps = 10000;
    if (std::abs(omega) > 1e-7) {
        double period = 2.0 * M_PI / std::abs(omega);
        acc_steps = static_cast<unsigned long long>(std::round(100.0 * period / dt));
    }

    std::cout << "Transient steps: " << transient_steps << "\n";
    std::cout << "Accumulation steps (100 periods): " << acc_steps << "\n\n";

    for (int step = 0; step < h0_steps; ++step) {
        double current_h0 = h0_vals[step];
        std::cout << "Running h0 = " << current_h0 << "...\n";

        dw::DomainWall<Backend> sim(size, L, alpha, current_h0, ha, omega, dt, eps, seed);
        Real current_time = 0.0;
        Backend::synchronize();

        for (unsigned long long t = 0; t < transient_steps; ++t) {
            sim.step(current_time);
        }
        Backend::synchronize();

        Real sum_var_u = 0.0;
        Real sum_u_dot = 0.0;
        Real sum_phi_dot = 0.0;
        Real sum_entropy = 0.0;

        VectorReal ps_accum(size);
        Backend::fill_zero(ps_accum);

        for (unsigned long long t = 0; t < acc_steps; ++t) {
            sim.step(current_time);

            auto obs = Backend::compute_observables(sim.get_state().u);
            Real N_grid = static_cast<Real>(size);

            Real mean_u = obs.u / N_grid;
            Real mean_u2 = obs.u2 / N_grid;
            Real var_u = mean_u2 - (mean_u * mean_u);

            Real mean_sin2phi = obs.sin2phi / N_grid;
            Real h_val = current_h0 + ha * std::cos(omega * current_time);

            Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
            Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

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

        out_summary << current_h0 << " " << avg_var_u << " " << avg_u_dot << " " << avg_phi_dot
                    << " " << avg_entropy << "\n";
        out_summary.flush();

        std::vector<Real> host_ps = Backend::download_array(ps_accum);
        std::string ps_file = output_prefix + "_ps_h0_" + std::to_string(current_h0) + ".txt";
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
    }
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <size> <L> <alpha> <ha> <omega> <dt> <h0_start> <h0_end> <h0_steps> "
                     "<backend> <output_prefix> "
                     "[eps=0.05] [seed=42] [precision=double] [randomize=0]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double L = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    double ha = std::stod(argv[4]);
    double omega = std::stod(argv[5]);
    double dt = std::stod(argv[6]);
    double h0_start = std::stod(argv[7]);
    double h0_end = std::stod(argv[8]);
    int h0_steps = std::stoi(argv[9]);
    std::string backend = argv[10];
    std::string output_prefix = argv[11];

    double eps = (argc > 12) ? std::stod(argv[12]) : 0.05;
    unsigned int seed = (argc > 13) ? std::stoul(argv[13]) : 42;
    std::string precision = (argc > 14) ? argv[14] : "double";

    bool randomize = false;
    if (argc > 15) {
        std::string rand_str = argv[15];
        if (rand_str == "1" || rand_str == "true" || rand_str == "True") {
            randomize = true;
        }
    }

    if (backend == "cpu") {
        if (precision == "single") {
            run_h0_sweep<dw::CpuBackend<dw::FftwSinglePrecision>>(size, L, alpha, ha, omega, dt,
                                                                  h0_start, h0_end, h0_steps, eps,
                                                                  seed, output_prefix, randomize);
        } else {
            run_h0_sweep<dw::CpuBackend<dw::FftwDoublePrecision>>(size, L, alpha, ha, omega, dt,
                                                                  h0_start, h0_end, h0_steps, eps,
                                                                  seed, output_prefix, randomize);
        }
    } else if (backend == "gpu") {
#ifdef HAS_CUDA
        if (precision == "single") {
            run_h0_sweep<dw::GpuBackend<dw::CudaSinglePrecision>>(size, L, alpha, ha, omega, dt,
                                                                  h0_start, h0_end, h0_steps, eps,
                                                                  seed, output_prefix, randomize);
        } else {
            run_h0_sweep<dw::GpuBackend<dw::CudaDoublePrecision>>(size, L, alpha, ha, omega, dt,
                                                                  h0_start, h0_end, h0_steps, eps,
                                                                  seed, output_prefix, randomize);
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