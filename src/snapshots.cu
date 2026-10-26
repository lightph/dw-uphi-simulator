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

#include "cuda_precision.cuh"
#include "domain_wall.cuh"
#include "gpu_backend.cuh"

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
    unsigned long long print_interval = std::max<unsigned long long>(1, total_sim_steps / 20);

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

    std::vector<unsigned long long> checkpoints;
    if (N_trans > 0) {
        double log_min = 0.0;
        double log_max = std::log10(static_cast<double>(N_trans));
        for (int i = 0; i < 20; ++i) {
            double log_val = log_min + i * (log_max - log_min) / 19.0;
            unsigned long long cp =
                static_cast<unsigned long long>(std::round(std::pow(10.0, log_val)));
            if (cp == 0) cp = 1;
            if (cp > N_trans) cp = N_trans;
            checkpoints.push_back(cp);
        }
        std::sort(checkpoints.begin(), checkpoints.end());
        checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    }

    struct Metrics {
        Real var_u, u_dot, phi_dot, entropy;
    };
    std::vector<std::deque<Metrics>> history(batch);
    std::vector<std::deque<unsigned long long>> history_steps(batch);

    auto start_time_chrono = std::chrono::high_resolution_clock::now();
    unsigned long long global_step = 0;

    auto print_progress = [&](const std::string& phase) {
        if (global_step % print_interval == 0 || global_step == total_sim_steps) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start_time_chrono;
            double steps_per_sec = global_step / elapsed.count();
            double eta_sec = (total_sim_steps - global_step) / steps_per_sec;

            int eta_h = eta_sec / 3600;
            int eta_m = (static_cast<int>(eta_sec) % 3600) / 60;
            int eta_s = static_cast<int>(eta_sec) % 60;

            std::cout << "[" << phase << "] Step " << global_step << " / " << total_sim_steps
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * global_step / total_sim_steps) << "%) "
                      << "| Elapsed: " << std::setprecision(1) << elapsed.count() << "s "
                      << "| ETA: " << std::setfill('0') << std::setw(2) << eta_h << ":"
                      << std::setw(2) << eta_m << ":" << std::setw(2) << eta_s << " ("
                      << std::setprecision(0) << steps_per_sec << " steps/s)" << std::endl;
        }
    };

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
        print_progress("Transient");
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
        print_progress("Snapshot");
    }
    out_snap.close();

    std::cout << "Simulation completed successfully." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 14) {
        std::cerr << "Usage: " << argv[0] << " <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> "
                  << "<transient_time> <snapshot_time> <window_time> <snapshot_rate> "
                  << "<output_prefix> [eps=0.05] [seed=42] [precision=double]\n";
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
    std::string precision = (argc > 16) ? argv[16] : "double";

    if (precision == "single") {
        run_time_evolution<dw::GpuBackend<dw::CudaSinglePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, snapshot_time, window_time,
            snapshot_rate, eps, seed, output_prefix);
    } else {
        run_time_evolution<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, batch, L, alpha, h0, ha, omega, dt, transient_time, snapshot_time, window_time,
            snapshot_rate, eps, seed, output_prefix);
    }

    return 0;
}