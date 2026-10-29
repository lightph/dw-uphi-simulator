#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
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

template <typename Backend>
void run_time_evolution(std::size_t size, std::size_t batch, double L, double alpha, double h0,
                        double ha, double omega, double dt, double transient_time,
                        double averaging_time, double window_time,
                        unsigned long long subsample_rate, double eps, unsigned int seed,
                        const std::string& output_prefix) {
    using Real = typename Backend::PrecisionType::Real;
    using VectorReal = typename Backend::template Vector<Real>;
    using VectorUll = typename Backend::template Vector<unsigned long long>;

    unsigned long long N_trans = static_cast<unsigned long long>(std::round(transient_time / dt));
    unsigned long long N_avg = static_cast<unsigned long long>(std::round(averaging_time / dt));
    unsigned long long N_win = static_cast<unsigned long long>(std::round(window_time / dt));
    if (N_win == 0) N_win = 1;

    unsigned long long total_sim_steps = N_trans + N_avg;

    std::cout << "Transient steps: " << N_trans << "\n"
              << "Averaging steps: " << N_avg << "\n"
              << "Window steps: " << N_win << "\n"
              << "Total steps: " << total_sim_steps << "\n"
              << "Simulating " << batch << " concurrent systems\n"
              << std::endl;

    std::string transient_file = output_prefix + "_transient_metrics.txt";
    std::ofstream out_transient(transient_file);
    out_transient << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_transient << "batch step time avg_var_u avg_u_dot avg_phi_dot avg_entropy\n";

    unsigned long long expected_ts_size =
        ((N_trans + N_avg) / (subsample_rate > 0 ? subsample_rate : 1) + 2) * batch;
    std::vector<int> ts_batch;
    std::vector<Real> ts_time;
    std::vector<Real> ts_mean_u;
    std::vector<Real> ts_u_dot;
    ts_batch.reserve(expected_ts_size);
    ts_time.reserve(expected_ts_size);
    ts_mean_u.reserve(expected_ts_size);
    ts_u_dot.reserve(expected_ts_size);

    // Expand scalar inputs into uniform batch vectors
    std::vector<double> h0_vec(batch, h0);
    std::vector<double> ha_vec(batch, ha);

    dw::DomainWall<Backend> sim(size, L, alpha, h0_vec, ha_vec, omega, dt, eps, seed);
    Real current_time = 0.0;
    Backend::synchronize();

    std::vector<unsigned long long> checkpoints = dw::utils::generate_log_checkpoints(N_trans, 20);

    struct Metrics {
        Real var_u, u_dot, phi_dot, entropy;
    };
    std::vector<std::deque<Metrics>> history(batch);
    std::vector<std::deque<unsigned long long>> history_steps(batch);

    unsigned long long global_step = 0;
    dw::utils::ProgressPrinter progress(total_sim_steps, 20);

    // --- TRANSIENT PHASE ---
    std::cout << "Starting transient phase..." << std::endl;
    for (unsigned long long step = 1; step <= N_trans; ++step) {
        sim.step(current_time);
        global_step++;

        bool in_window = false;
        for (auto cp : checkpoints) {
            unsigned long long start = (cp > N_win) ? (cp - N_win + 1) : 1;
            if (step >= start && step <= cp) {
                in_window = true;
                break;
            }
        }

        bool need_ts = (subsample_rate > 0) && (step % subsample_rate == 0);

        if (in_window || need_ts) {
            Real h_val = h0 + ha * std::cos(omega * current_time);
            Real N_grid = static_cast<Real>(size);

            for (std::size_t b = 0; b < batch; ++b) {
                auto obs = Backend::compute_observables(sim.get_state().get_u(b), size);
                Real mean_u = obs.u / N_grid;
                Real var_u = obs.var_sum / N_grid;
                Real mean_sin2phi = obs.sin2phi / N_grid;

                Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
                Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

                if (in_window) {
                    Real entropy =
                        Backend::compute_spectral_entropy(sim.get_state().get_u_hat(b), size);
                    history[b].push_back({var_u, u_dot, phi_dot, entropy});
                    history_steps[b].push_back(step);
                }

                if (need_ts) {
                    ts_batch.push_back(b);
                    ts_time.push_back(current_time);
                    ts_mean_u.push_back(mean_u);
                    ts_u_dot.push_back(u_dot);
                }
            }
        }

        auto it = std::find(checkpoints.begin(), checkpoints.end(), step);
        if (it != checkpoints.end()) {
            unsigned long long start = (step > N_win) ? (step - N_win + 1) : 1;

            for (std::size_t b = 0; b < batch; ++b) {
                Real s_var = 0, s_udot = 0, s_phidot = 0, s_ent = 0;
                unsigned long long count = 0;
                for (size_t i = 0; i < history[b].size(); ++i) {
                    if (history_steps[b][i] >= start && history_steps[b][i] <= step) {
                        s_var += history[b][i].var_u;
                        s_udot += history[b][i].u_dot;
                        s_phidot += history[b][i].phi_dot;
                        s_ent += history[b][i].entropy;
                        count++;
                    }
                }
                if (count > 0) {
                    out_transient << b << " " << step << " " << current_time << " " << s_var / count
                                  << " " << s_udot / count << " " << s_phidot / count << " "
                                  << s_ent / count << "\n";
                }
            }
            out_transient.flush();

            auto next_it = it + 1;
            unsigned long long next_start = (next_it != checkpoints.end())
                                                ? ((*next_it > N_win) ? (*next_it - N_win + 1) : 1)
                                                : step + 1;

            for (std::size_t b = 0; b < batch; ++b) {
                while (!history_steps[b].empty() && history_steps[b].front() < next_start) {
                    history_steps[b].pop_front();
                    history[b].pop_front();
                }
            }
        }
        progress.print("Transient", global_step);
    }
    out_transient.close();

    // --- AVERAGING PHASE ---
    std::cout << "Starting averaging phase..." << std::endl;
    std::vector<Real> sum_var_u_final(batch, 0.0);
    std::vector<Real> sum_u_dot_final(batch, 0.0);
    std::vector<Real> sum_phi_dot_final(batch, 0.0);
    std::vector<Real> sum_entropy_final(batch, 0.0);

    VectorReal ps_accum(size * batch);
    Backend::fill_zero(ps_accum.data(), ps_accum.size());

    // Single pre-pass to establish safe global histogram bounds across all batches
    Real max_sigma = 1.0;
    for (std::size_t b = 0; b < batch; ++b) {
        auto obs_est = Backend::compute_observables(sim.get_state().get_u(b), size);
        Real var_est = obs_est.var_sum / static_cast<Real>(size);
        max_sigma = std::max(max_sigma, std::sqrt(var_est));
    }

    std::size_t num_bins = 1000;
    Real hist_min = -10.0 * max_sigma;
    Real hist_max = 10.0 * max_sigma;
    VectorUll d_hist(num_bins * batch);
    Backend::fill_zero_ull(d_hist.data(), d_hist.size());

    for (unsigned long long step = 1; step <= N_avg; ++step) {
        sim.step(current_time);
        global_step++;

        bool need_ts = (subsample_rate > 0) && (global_step % subsample_rate == 0);
        Real h_val = h0 + ha * std::cos(omega * current_time);
        Real N_grid = static_cast<Real>(size);

        for (std::size_t b = 0; b < batch; ++b) {
            auto obs = Backend::compute_observables(sim.get_state().get_u(b), size);
            Real mean_u = obs.u / N_grid;
            Real var_u = obs.var_sum / N_grid;
            Real mean_sin2phi = obs.sin2phi / N_grid;

            Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
            Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

            sum_var_u_final[b] += var_u;
            sum_u_dot_final[b] += u_dot;
            sum_phi_dot_final[b] += phi_dot;
            sum_entropy_final[b] +=
                Backend::compute_spectral_entropy(sim.get_state().get_u_hat(b), size);

            Backend::accumulate_power_spectrum(sim.get_state().get_u_hat(b),
                                               ps_accum.data() + b * size, size);

            Backend::accumulate_height_histogram(sim.get_state().get_u(b), size, mean_u, 1.0,
                                                 d_hist.data() + b * num_bins, num_bins, hist_min,
                                                 hist_max);

            if (need_ts) {
                ts_batch.push_back(b);
                ts_time.push_back(current_time);
                ts_mean_u.push_back(mean_u);
                ts_u_dot.push_back(u_dot);
            }
        }
        progress.print("Averaging", global_step);
    }

    // --- FINAL DATA EXPORT ---
    std::cout << "Writing time series data to disk..." << std::endl;
    std::string ts_file = output_prefix + "_time_series.txt";
    std::ofstream out_ts(ts_file);
    out_ts << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_ts << "batch time mean_u u_dot\n";
    for (size_t i = 0; i < ts_time.size(); ++i) {
        out_ts << ts_batch[i] << " " << ts_time[i] << " " << ts_mean_u[i] << " " << ts_u_dot[i]
               << "\n";
    }
    out_ts.close();

    if (N_avg > 0) {
        std::string summary_file = output_prefix + "_final_summary.txt";
        std::ofstream out_summary(summary_file);
        out_summary << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_summary << "batch total_steps var_u u_dot phi_dot spectral_entropy\n";

        for (std::size_t b = 0; b < batch; ++b) {
            out_summary << b << " " << (N_trans + N_avg) << " " << sum_var_u_final[b] / N_avg << " "
                        << sum_u_dot_final[b] / N_avg << " " << sum_phi_dot_final[b] / N_avg << " "
                        << sum_entropy_final[b] / N_avg << "\n";
        }
        out_summary.close();

        std::string ps_file = output_prefix + "_final_ps.txt";
        std::ofstream out_ps(ps_file);
        out_ps << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ps << "batch k power\n";

        std::vector<Real> host_ps = Backend::download_array(ps_accum.data(), ps_accum.size());
        double dk = 2.0 * M_PI / L;
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < size; ++i) {
                long long k_idx =
                    (i <= size / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size);
                double k_phys = k_idx * dk;
                std::size_t global_idx = b * size + i;
                out_ps << b << " " << k_phys << " " << L * (host_ps[global_idx] / N_avg) << "\n";
            }
        }
        out_ps.close();

        std::vector<unsigned long long> h_hist =
            Backend::download_array_ull(d_hist.data(), d_hist.size());
        std::string hist_file = output_prefix + "_final_height_hist.txt";
        std::ofstream out_hist(hist_file);
        out_hist << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_hist << "batch bin_center count probability_density\n";

        Real bin_width = (hist_max - hist_min) / static_cast<Real>(num_bins);

        for (std::size_t b = 0; b < batch; ++b) {
            unsigned long long total_counts = 0;
            for (std::size_t i = 0; i < num_bins; ++i) {
                total_counts += h_hist[b * num_bins + i];
            }

            Real global_sigma = std::sqrt(sum_var_u_final[b] / N_avg);

            for (std::size_t i = 0; i < num_bins; ++i) {
                Real bin_center = hist_min + (i + 0.5) * bin_width;
                Real z_center = bin_center / global_sigma;

                Real prob = total_counts > 0 ? static_cast<Real>(h_hist[b * num_bins + i]) /
                                                   static_cast<Real>(total_counts)
                                             : 0.0;
                Real pdf = (prob / bin_width) * global_sigma;

                out_hist << b << " " << z_center << " " << h_hist[b * num_bins + i] << " " << pdf
                         << "\n";
            }
        }
        out_hist.close();
    }

    std::cout << "Simulation completed successfully." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 14) {
        std::cerr << "Usage: " << argv[0] << " <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> "
                  << "<transient_time> <averaging_time> <window_time> <subsample_rate> "
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

    double transient_time = std::stod(argv[9]);
    double averaging_time = std::stod(argv[10]);
    double window_time = std::stod(argv[11]);
    unsigned long long subsample_rate = std::stoull(argv[12]);
    std::string output_prefix = argv[13];

    double eps = (argc > 14) ? std::stod(argv[14]) : 0.0005;
    unsigned int seed = (argc > 15) ? std::stoul(argv[15]) : 42;
    std::string backend = (argc > 16) ? argv[16] : "gpu";

    if (backend == "cpu") {
        run_time_evolution<dw::CpuBackend<dw::FftwDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, averaging_time, window_time,
            subsample_rate, eps, seed, output_prefix);
    } else {
#ifdef HAS_CUDA
        run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, averaging_time, window_time,
            subsample_rate, eps, seed, output_prefix);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
