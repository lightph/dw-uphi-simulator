#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#include "floquet.hpp"

// Binary struct matching publication plotter (load_floquet_boundaries in plot_prb_publication.py)
struct Result {
    double h;
    double h_ac;
    double omega;
    double x;
    double max_k_mu;
    double max_mu;
    double max_k_sigma;
    double max_sigma;
    bool converged;
};

int main(int argc, char** argv) {
    double alpha = 0.27;
    double omega_val = 1.0;
    double L = 1024.0;
    double h0_min = 0.0, h0_max = 10.0;
    int num_h0 = 500;
    double ha_min = 0.0, ha_max = 20.0;
    int num_ha = 500;
    double fixed_ha = -1.0; // If >= 0, performs (h0, kappa) slice for stability diagram
    double k_min = 0.7, k_max = 1.2;
    int num_k = 200;
    std::string out_file = "floquet_sweep_cpu.bin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--alpha" && i + 1 < argc) alpha = std::stod(argv[++i]);
        else if (arg == "--omega" && i + 1 < argc) omega_val = std::stod(argv[++i]);
        else if (arg == "-L" && i + 1 < argc) L = std::stod(argv[++i]);
        else if (arg == "--h0-min" && i + 1 < argc) h0_min = std::stod(argv[++i]);
        else if (arg == "--h0-max" && i + 1 < argc) h0_max = std::stod(argv[++i]);
        else if (arg == "--num-h0" && i + 1 < argc) num_h0 = std::stoi(argv[++i]);
        else if (arg == "--ha" && i + 1 < argc) { fixed_ha = std::stod(argv[++i]); }
        else if (arg == "--ha-min" && i + 1 < argc) ha_min = std::stod(argv[++i]);
        else if (arg == "--ha-max" && i + 1 < argc) ha_max = std::stod(argv[++i]);
        else if (arg == "--num-ha" && i + 1 < argc) num_ha = std::stoi(argv[++i]);
        else if (arg == "--k-min" && i + 1 < argc) k_min = std::stod(argv[++i]);
        else if (arg == "--k-max" && i + 1 < argc) k_max = std::stod(argv[++i]);
        else if (arg == "--num-k" && i + 1 < argc) num_k = std::stoi(argv[++i]);
        else if (arg == "-o" || arg == "--output") { if (i + 1 < argc) out_file = argv[++i]; }
    }

    const int steps_per_period = 500;
    const double dt = dw::floquet::TWO_PI_VAL / steps_per_period;
    const int transient_periods = 300, max_check_periods = 200;
    const double tol = 1e-6;

    std::vector<double> cos_t0(steps_per_period), cos_thalf(steps_per_period);
    for (int s = 0; s < steps_per_period; ++s) {
        cos_t0[s] = std::cos(s * dt);
        cos_thalf[s] = std::cos((s + 0.5) * dt);
    }

    // MODE A: Fixed ha -> (h0, kappa) grid for Figure 3 stability diagram
    if (fixed_ha >= 0.0) {
        std::cout << "========================================\n";
        std::cout << "Floquet Mode Stability Grid (Figure 3 Mode)\n";
        std::cout << "Fixed ha = " << fixed_ha << ", Omega = " << omega_val << ", Alpha = " << alpha << "\n";
        std::cout << "h0: [" << h0_min << ", " << h0_max << "] (" << num_h0 << " steps)\n";
        std::cout << "kappa: [" << k_min << ", " << k_max << "] (" << num_k << " steps)\n";
        std::cout << "========================================\n";

        std::vector<double> h0_vals(num_h0);
        for (int i = 0; i < num_h0; ++i) {
            h0_vals[i] = (num_h0 > 1) ? h0_min + i * (h0_max - h0_min) / (num_h0 - 1) : h0_min;
        }
        std::vector<double> k_vals(num_k);
        for (int i = 0; i < num_k; ++i) {
            k_vals[i] = (num_k > 1) ? k_min + i * (k_max - k_min) / (num_k - 1) : k_min;
        }

        std::vector<double> grid(num_h0 * num_k, -1.0); // -1.0 indicates unconverged

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_h0; ++i) {
            double h0 = h0_vals[i];
            double A = (alpha * h0) / omega_val;
            double B = alpha / omega_val;
            double C = (alpha * fixed_ha) / omega_val;

            double psi = 0.0;
            for (int p = 0; p < transient_periods; ++p) {
                for (int s = 0; s < steps_per_period; ++s) {
                    int inext = (s + 1) % steps_per_period;
                    double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                    double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                    double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                    double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                    psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                }
            }

            double prev_strobe = std::fmod(psi, dw::floquet::TWO_PI_VAL);
            if (prev_strobe < 0) prev_strobe += dw::floquet::TWO_PI_VAL;
            bool converged = false;

            for (int p = 0; p < max_check_periods; ++p) {
                for (int s = 0; s < steps_per_period; ++s) {
                    int inext = (s + 1) % steps_per_period;
                    double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                    double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                    double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                    double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                    psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                }
                double curr_strobe = std::fmod(psi, dw::floquet::TWO_PI_VAL);
                if (curr_strobe < 0) curr_strobe += dw::floquet::TWO_PI_VAL;
                if (dw::floquet::circle_distance(curr_strobe, prev_strobe) < tol) {
                    converged = true;
                    prev_strobe = curr_strobe;
                    break;
                }
                prev_strobe = curr_strobe;
            }

            if (converged) {
                for (int k_idx = 0; k_idx < num_k; ++k_idx) {
                    double kappa = k_vals[k_idx];
                    auto met = dw::floquet::eval_floquet_mode(
                        prev_strobe, A, B, C, kappa, omega_val, alpha, dt,
                        steps_per_period, cos_t0.data(), cos_thalf.data());
                    grid[i * num_k + k_idx] = met.mu;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Grid evaluated in " << elapsed << " s.\n";

        // Save text grid
        std::ofstream out(out_file);
        out << std::setprecision(8);
        out << "h0 kappa mu\n";
        for (int i = 0; i < num_h0; ++i) {
            for (int k_idx = 0; k_idx < num_k; ++k_idx) {
                out << h0_vals[i] << " " << k_vals[k_idx] << " " << grid[i * num_k + k_idx] << "\n";
            }
        }
        std::cout << "Saved: " << out_file << "\n";
        return 0;
    }

    // MODE B: Full (h0, ha) parameter sweep over modes for system size L
    std::cout << "========================================\n";
    std::cout << "Floquet Stability Parameter Sweep (CPU OpenMP)\n";
    std::cout << "System Size L = " << L << ", Omega = " << omega_val << ", Alpha = " << alpha << "\n";
    std::cout << "h0: [" << h0_min << ", " << h0_max << "] (" << num_h0 << ")\n";
    std::cout << "ha: [" << ha_min << ", " << ha_max << "] (" << num_ha << ")\n";
    std::cout << "Total combinations: " << num_h0 * num_ha << "\n";
    std::cout << "========================================\n";

    long long total = (long long)num_h0 * num_ha;
    std::vector<Result> results(total);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (long long flat = 0; flat < total; ++flat) {
        int i_h0 = flat / num_ha;
        int i_ha = flat % num_ha;

        double h0 = (num_h0 > 1) ? h0_min + i_h0 * (h0_max - h0_min) / (num_h0 - 1) : h0_min;
        double ha = (num_ha > 1) ? ha_min + i_ha * (ha_max - ha_min) / (num_ha - 1) : ha_min;

        Result res;
        res.h = h0;
        res.h_ac = ha;
        res.omega = omega_val;
        res.max_mu = 0.0;
        res.max_sigma = 0.0;
        res.max_k_mu = 0.0;
        res.max_k_sigma = 0.0;

        double A = (alpha * h0) / omega_val;
        double B = alpha / omega_val;
        double C = (alpha * ha) / omega_val;

        double psi = 0.0;
        for (int p = 0; p < transient_periods; ++p) {
            for (int s = 0; s < steps_per_period; ++s) {
                int inext = (s + 1) % steps_per_period;
                double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
        }

        double prev_strobe = std::fmod(psi, dw::floquet::TWO_PI_VAL);
        if (prev_strobe < 0) prev_strobe += dw::floquet::TWO_PI_VAL;
        bool converged = false;

        for (int p = 0; p < max_check_periods; ++p) {
            for (int s = 0; s < steps_per_period; ++s) {
                int inext = (s + 1) % steps_per_period;
                double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
            double curr_strobe = std::fmod(psi, dw::floquet::TWO_PI_VAL);
            if (curr_strobe < 0) curr_strobe += dw::floquet::TWO_PI_VAL;
            if (dw::floquet::circle_distance(curr_strobe, prev_strobe) < tol) {
                converged = true;
                prev_strobe = curr_strobe;
                break;
            }
            prev_strobe = curr_strobe;
        }

        res.x = prev_strobe;
        res.converged = converged;

        if (converged) {
            double fund_k = dw::floquet::TWO_PI_VAL / L;
            const double K_MAX = 5.0;
            long long n_max = static_cast<long long>(std::floor(K_MAX / fund_k));
            if (n_max < 1) n_max = 1;

            auto m0 = dw::floquet::eval_floquet_mode(
                prev_strobe, A, B, C, fund_k, omega_val, alpha, dt,
                steps_per_period, cos_t0.data(), cos_thalf.data());
            res.max_k_mu = fund_k;
            res.max_mu = m0.mu;
            res.max_k_sigma = fund_k;
            res.max_sigma = m0.sigma;

            for (long long n = 2; n <= n_max; ++n) {
                double test_k = n * fund_k;
                auto met = dw::floquet::eval_floquet_mode(
                    prev_strobe, A, B, C, test_k, omega_val, alpha, dt,
                    steps_per_period, cos_t0.data(), cos_thalf.data());
                if (met.mu > res.max_mu) {
                    res.max_mu = met.mu;
                    res.max_k_mu = test_k;
                }
                if (met.sigma > res.max_sigma) {
                    res.max_sigma = met.sigma;
                    res.max_k_sigma = test_k;
                }
            }
        }
        results[flat] = res;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Sweep completed in " << elapsed << " s.\n";

    // Write binary file
    std::ofstream out(out_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(results.data()), total * sizeof(Result));
    out.close();
    std::cout << "Saved: " << out_file << "\n";

    return 0;
}

