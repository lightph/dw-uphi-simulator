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
    using VectorUll = typename Backend::template Vector<unsigned long long>;

    std::string summary_file = output_prefix + "_time_summary.txt";
    std::ofstream out_summary(summary_file);
    out_summary << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_summary << "total_steps var_u u_dot phi_dot spectral_entropy\n";

    std::string transient_file = output_prefix + "_transient_w2.txt";
    std::ofstream out_transient(transient_file);
    out_transient << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_transient << "step time mean_u var_u\n";

    unsigned long long current_step = 0;
    unsigned long long log_interval = std::max(1ULL, (1ULL << max_power) / 10000ULL);

    unsigned long long acc_steps_max = 100000;
    if (std::abs(omega) > 1e-7) {
        double period = 2.0 * M_PI / std::abs(omega);
        acc_steps_max = static_cast<unsigned long long>(std::round(1000.0 * period / dt));
    }

    std::cout << "Max accumulation steps (100 periods cap): " << acc_steps_max << "\n\n";

    dw::DomainWall<Backend> sim(size, L, alpha, h0, ha, omega, dt, eps, seed);
    Real current_time = 0.0;
    Backend::synchronize();

    unsigned long long total_steps_done = 0;

    for (int n = 0; n <= max_power; ++n) {
        unsigned long long target_steps = 1ULL << n;
        unsigned long long batch_steps = target_steps - total_steps_done;

        unsigned long long acc_steps = std::min(batch_steps, acc_steps_max);
        unsigned long long transient_batch_steps = batch_steps - acc_steps;

        std::cout << "Target total steps: " << target_steps << " | Batch steps: " << batch_steps
                  << " | Accumulating over last: " << acc_steps << "\n";

        for (unsigned long long t = 0; t < transient_batch_steps; ++t) {
            sim.step(current_time);
            current_step++;

            if (current_step % log_interval == 0) {
                auto obs = Backend::compute_observables(sim.get_state().u);
                Real N_grid = static_cast<Real>(size);
                Real mean_u = obs.u / N_grid;
                Real mean_u2 = obs.u2 / N_grid;
                Real var_u = mean_u2 - (mean_u * mean_u);
                out_transient << current_step << " " << current_time << " " << mean_u << " "
                              << var_u << "\n";
            }
        }
        Backend::synchronize();

        Real sum_var_u = 0.0;
        Real sum_u_dot = 0.0;
        Real sum_phi_dot = 0.0;
        Real sum_entropy = 0.0;

        VectorReal ps_accum(size);
        Backend::fill_zero(ps_accum);

        std::vector<Real> mean_u_series(acc_steps);
        std::vector<Real> u_dot_series(acc_steps);

        std::size_t num_bins = 1000;
        Real hist_min = -10.0;
        Real hist_max = 10.0;
        VectorUll d_hist(num_bins);
        Backend::fill_zero_ull(d_hist);

        for (unsigned long long t = 0; t < acc_steps; ++t) {
            sim.step(current_time);

            auto obs = Backend::compute_observables(sim.get_state().u);
            Real N_grid = static_cast<Real>(size);

            Real mean_u = obs.u / N_grid;
            Real mean_u2 = obs.u2 / N_grid;
            Real var_u = mean_u2 - (mean_u * mean_u);
            Real sigma_u = std::sqrt(std::max(var_u, Real(0.0)));

            current_step++;
            if (current_step % log_interval == 0) {
                out_transient << current_step << " " << current_time << " " << mean_u << " "
                              << var_u << "\n";
            }

            Real mean_sin2phi = obs.sin2phi / N_grid;
            Real h_val = h0 + ha * std::cos(omega * current_time);

            Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
            Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

            mean_u_series[t] = mean_u;
            u_dot_series[t] = u_dot;

            if (sigma_u > 0.0) {
                Backend::accumulate_height_histogram(sim.get_state().u, mean_u, sigma_u, d_hist,
                                                     hist_min, hist_max);
            }

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
        out_ts << "time mean_u u_dot\n";
        for (unsigned long long i = 0; i < acc_steps; ++i) {
            out_ts << (i * dt) << " " << mean_u_series[i] << " " << u_dot_series[i] << "\n";
        }
        out_ts.close();

        std::vector<unsigned long long> h_hist = Backend::download_array_ull(d_hist);
        std::string hist_file =
            output_prefix + "_height_hist_step_" + std::to_string(target_steps) + ".txt";
        std::ofstream out_hist(hist_file);
        out_hist << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_hist << "bin_center count probability_density\n";
        Real bin_width = (hist_max - hist_min) / static_cast<Real>(num_bins);
        unsigned long long total_counts = 0;
        for (auto c : h_hist) total_counts += c;

        for (std::size_t i = 0; i < num_bins; ++i) {
            Real bin_center = hist_min + (i + 0.5) * bin_width;
            Real prob = total_counts > 0
                            ? static_cast<Real>(h_hist[i]) / static_cast<Real>(total_counts)
                            : 0.0;
            Real pdf = prob / bin_width;
            out_hist << bin_center << " " << h_hist[i] << " " << pdf << "\n";
        }
        out_hist.close();

        auto host_u = Backend::download_array(sim.get_state().u);
        std::string state_file =
            output_prefix + "_real_state_step_" + std::to_string(target_steps) + ".txt";
        std::ofstream out_state(state_file);
        out_state << std::setprecision(std::numeric_limits<Real>::max_digits10);

        out_state << "x Re(u) Im(u)\n";

        double dx = L / size;
        for (std::size_t i = 0; i < size; ++i) {
            out_state << (i * dx) << " " << host_u[i].real() << " " << host_u[i].imag() << "\n";
        }
        out_state.close();

        VectorReal inst_ps(size);
        Backend::fill_zero(inst_ps);
        Backend::accumulate_power_spectrum(sim.get_state().u_hat, inst_ps);
        std::vector<Real> host_inst_ps = Backend::download_array(inst_ps);

        std::string inst_ps_file =
            output_prefix + "_inst_ps_step_" + std::to_string(target_steps) + ".txt";
        std::ofstream out_inst_ps(inst_ps_file);
        out_inst_ps << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_inst_ps << "k power\n";
        for (std::size_t i = 0; i < size; ++i) {
            long long k_idx =
                (i <= size / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size);
            double k_phys = k_idx * dk;
            out_inst_ps << k_phys << " " << host_inst_ps[i] << "\n";  // Raw instantaneous value
        }
        out_inst_ps.close();

        auto final_obs = Backend::compute_observables(sim.get_state().u);
        Real inst_mean_u = final_obs.u / static_cast<Real>(size);
        Real inst_mean_u2 = final_obs.u2 / static_cast<Real>(size);
        Real inst_var_u = inst_mean_u2 - (inst_mean_u * inst_mean_u);
        Real inst_sigma_u = std::sqrt(std::max(inst_var_u, Real(0.0)));

        if (inst_sigma_u > 0.0) {
            VectorUll inst_hist(num_bins);
            Backend::fill_zero_ull(inst_hist);
            Backend::accumulate_height_histogram(sim.get_state().u, inst_mean_u, inst_sigma_u,
                                                 inst_hist, hist_min, hist_max);

            std::vector<unsigned long long> h_inst_hist = Backend::download_array_ull(inst_hist);
            std::string inst_hist_file =
                output_prefix + "_inst_height_hist_step_" + std::to_string(target_steps) + ".txt";
            std::ofstream out_inst_hist(inst_hist_file);
            out_inst_hist << std::setprecision(std::numeric_limits<Real>::max_digits10);
            out_inst_hist << "bin_center count probability_density\n";
            unsigned long long inst_total_counts = 0;
            for (auto c : h_inst_hist) inst_total_counts += c;

            for (std::size_t i = 0; i < num_bins; ++i) {
                Real bin_center = hist_min + (i + 0.5) * bin_width;
                Real prob = inst_total_counts > 0 ? static_cast<Real>(h_inst_hist[i]) /
                                                        static_cast<Real>(inst_total_counts)
                                                  : 0.0;
                out_inst_hist << bin_center << " " << h_inst_hist[i] << " " << (prob / bin_width)
                              << "\n";
            }
            out_inst_hist.close();
        }

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