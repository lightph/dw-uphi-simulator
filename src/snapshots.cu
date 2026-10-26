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
                        double snapshot_time, double window_time, unsigned long long snapshot_rate,
                        double eps, unsigned int seed, const std::string& output_prefix) {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using VectorReal = typename Backend::template Vector<Real>;

    unsigned long long N_trans = static_cast<unsigned long long>(std::round(transient_time / dt));
    unsigned long long N_snap = static_cast<unsigned long long>(std::round(snapshot_time / dt));
    unsigned long long N_win = static_cast<unsigned long long>(std::round(window_time / dt));
    if (N_win == 0) N_win = 1;

    unsigned long long total_sim_steps = N_trans + N_snap;

    std::cout << "Transient steps: " << N_trans << "\n"
              << "Snapshot steps: " << N_snap << "\n"
              << "Window steps: " << N_win << "\n"
              << "Total steps: " << total_sim_steps << "\n"
              << "Snapshot every " << snapshot_rate << " steps\n"
              << "Simulating " << batch << " concurrent systems\n"
              << std::endl;

    std::string transient_file = output_prefix + "_transient_metrics.txt";
    std::ofstream out_transient(transient_file);
    out_transient << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out_transient << "batch step time avg_var_u avg_u_dot avg_phi_dot avg_entropy\n";

    std::vector<double> h0_batch(batch, h0);
    std::vector<double> ha_batch(batch, ha);
    dw::DomainWall<Backend> sim(size, L, alpha, h0_batch, ha_batch, omega, dt, eps, seed);
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

    // --- SNAPSHOT PHASE ---
    std::cout << "Starting snapshot phase..." << std::endl;
    std::string snap_file = output_prefix + "_snapshots.txt";
    std::ofstream out_snap(snap_file);
    out_snap << std::setprecision(std::numeric_limits<Real>::max_digits10);

    // Write Header mapping u_i (real) and phi_i (imaginary)
    out_snap << "batch time";
    for (std::size_t i = 0; i < size; ++i) {
        out_snap << " u_" << i << " phi_" << i;
    }
    out_snap << "\n";

    for (unsigned long long step = 1; step <= N_snap; ++step) {
        sim.step(current_time);
        global_step++;

        if (snapshot_rate > 0 && step % snapshot_rate == 0) {
            for (std::size_t b = 0; b < batch; ++b) {
                // Download the full complex array from the GPU
                std::vector<Complex> host_u =
                    Backend::download_array(sim.get_state().get_u(b), size);

                out_snap << b << " " << current_time;
                for (std::size_t i = 0; i < size; ++i) {
                    // Extract real (u) and imaginary (phi) components safely via memory cast
                    Real u_val = reinterpret_cast<Real*>(&host_u[i])[0];
                    Real phi_val = reinterpret_cast<Real*>(&host_u[i])[1];
                    out_snap << " " << u_val << " " << phi_val;
                }
                out_snap << "\n";
            }
        }
        progress.print("Snapshot", global_step);
    }
    out_snap.close();

    std::cout << "Simulation completed successfully." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 14) {
        std::cerr << "Usage: " << argv[0] << " <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> "
                  << "<transient_time> <snapshot_time> <window_time> <snapshot_rate> "
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
    double snapshot_time = std::stod(argv[10]);
    double window_time = std::stod(argv[11]);
    unsigned long long snapshot_rate = std::stoull(argv[12]);
    std::string output_prefix = argv[13];

    double eps = (argc > 14) ? std::stod(argv[14]) : 0.05;
    unsigned int seed = (argc > 15) ? std::stoul(argv[15]) : 42;
    std::string backend = (argc > 16) ? argv[16] : "gpu";

    if (backend == "cpu") {
        run_time_evolution<dw::CpuBackend<dw::FftwDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, snapshot_time, window_time,
            snapshot_rate, eps, seed, output_prefix);
    } else {
#ifdef HAS_CUDA
        run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, snapshot_time, window_time,
            snapshot_rate, eps, seed, output_prefix);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
