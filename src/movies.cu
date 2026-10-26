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
                        double ha, double omega, double dt, double t_max, double window_time,
                        double movie_start, double movie_end, unsigned long long movie_rate,
                        double eps, unsigned int seed, const std::string& output_prefix) {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;

    unsigned long long total_sim_steps = static_cast<unsigned long long>(std::round(t_max / dt));
    unsigned long long movie_start_step =
        static_cast<unsigned long long>(std::round(movie_start / dt));
    unsigned long long movie_end_step = static_cast<unsigned long long>(std::round(movie_end / dt));
    unsigned long long N_win = static_cast<unsigned long long>(std::round(window_time / dt));
    if (N_win == 0) N_win = 1;

    std::cout << "Total simulation steps: " << total_sim_steps << "\n"
              << "Window steps: " << N_win << "\n"
              << "Movie frames between steps " << movie_start_step << " and " << movie_end_step
              << "\n"
              << "Taking snapshot every " << movie_rate << " steps\n"
              << "Simulating " << batch << " concurrent systems\n"
              << std::endl;

    std::string metrics_file = output_prefix + "_metrics.txt";
    std::ofstream out_metrics(metrics_file);
    out_metrics << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_metrics << "batch step time avg_var_u avg_u_dot avg_phi_dot avg_entropy\n";

    std::string snap_file = output_prefix + "_snapshots.bin";
    std::ofstream out_snap(snap_file, std::ios::binary);

    std::vector<double> h0_vec(batch, h0);
    std::vector<double> ha_vec(batch, ha);
    dw::DomainWall<Backend> sim(size, L, alpha, h0_vec, ha_vec, omega, dt, eps, seed);

    Real current_time = 0.0;
    Backend::synchronize();

    std::vector<unsigned long long> checkpoints = dw::utils::generate_log_checkpoints(total_sim_steps, 20);

    struct Metrics {
        Real var_u, u_dot, phi_dot, entropy;
    };
    std::vector<std::deque<Metrics>> history(batch);
    std::vector<std::deque<unsigned long long>> history_steps(batch);

    dw::utils::ProgressPrinter progress(total_sim_steps, 20);

    std::cout << "Starting time evolution..." << std::endl;
    for (unsigned long long step = 1; step <= total_sim_steps; ++step) {
        sim.step(current_time);

        // --- METRICS RECORDING ---
        bool in_window = false;
        for (auto cp : checkpoints) {
            unsigned long long start = (cp > N_win) ? (cp - N_win + 1) : 1;
            if (step >= start && step <= cp) {
                in_window = true;
                break;
            }
        }

        if (in_window) {
            Real h_val = h0 + ha * std::cos(omega * current_time);
            Real N_grid = static_cast<Real>(size);

            for (std::size_t b = 0; b < batch; ++b) {
                auto obs = Backend::compute_observables(sim.get_state().get_u(b), size);
                Real var_u = obs.var_sum / N_grid;
                Real mean_sin2phi = obs.sin2phi / N_grid;

                Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
                Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);
                Real entropy =
                    Backend::compute_spectral_entropy(sim.get_state().get_u_hat(b), size);

                history[b].push_back({var_u, u_dot, phi_dot, entropy});
                history_steps[b].push_back(step);
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
                    out_metrics << b << " " << step << " " << current_time << " " << s_var / count
                                << " " << s_udot / count << " " << s_phidot / count << " "
                                << s_ent / count << "\n";
                }
            }
            out_metrics.flush();

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

        // --- MOVIE SNAPSHOT PHASE ---
        if (step >= movie_start_step && step <= movie_end_step) {
            if (movie_rate > 0 && (step - movie_start_step) % movie_rate == 0) {
                for (std::size_t b = 0; b < batch; ++b) {
                    // Download the full complex array from the GPU
                    std::vector<Complex> host_u =
                        Backend::download_array(sim.get_state().get_u(b), size);

                    // Write batch ID, current time, and then dump the entire array from memory
                    out_snap.write(reinterpret_cast<const char*>(&b), sizeof(std::size_t));
                    out_snap.write(reinterpret_cast<const char*>(&current_time), sizeof(Real));
                    out_snap.write(reinterpret_cast<const char*>(host_u.data()),
                                   size * sizeof(Complex));
                }
                out_snap.flush();
            }
        }

        progress.print("Simulating", step);
    }

    out_metrics.close();
    out_snap.close();
    std::cout << "Simulation completed successfully." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 15) {
        std::cerr << "Usage: " << argv[0] << " <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> "
                  << "<t_max> <window_time> <movie_start> <movie_end> <movie_rate> "
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

    double t_max = std::stod(argv[9]);
    double window_time = std::stod(argv[10]);
    double movie_start = std::stod(argv[11]);
    double movie_end = std::stod(argv[12]);
    unsigned long long movie_rate = std::stoull(argv[13]);
    std::string output_prefix = argv[14];

    double eps = (argc > 15) ? std::stod(argv[15]) : 0.05;
    unsigned int seed = (argc > 16) ? std::stoul(argv[16]) : 42;
    std::string backend = (argc > 17) ? argv[17] : "gpu";

    if (backend == "cpu") {
        run_time_evolution<dw::CpuBackend<dw::FftwDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, t_max, window_time, movie_start, movie_end,
            movie_rate, eps, seed, output_prefix);
    } else {
#ifdef HAS_CUDA
        run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, t_max, window_time, movie_start, movie_end,
            movie_rate, eps, seed, output_prefix);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
