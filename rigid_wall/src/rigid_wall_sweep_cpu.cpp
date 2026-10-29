#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#include "rigid_wall.hpp"

struct Result {
    double h0;
    double ha;
    double omega;
    double psi_eq;
    double v_psi;
    double v_q;
    double a0_cos;
    bool converged;
};

int main(int argc, char** argv) {
    // Default parameters
    double alpha = 0.27;
    double omega = 1.0;
    double h0_min = 0.0, h0_max = 10.0;
    int num_h0 = 500;
    double ha_min = 0.0, ha_max = 10.0;
    int num_ha = 1;
    double fixed_ha = 5.0;
    std::string out_file = "rigid_wall_results.txt";

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--alpha" && i + 1 < argc) alpha = std::stod(argv[++i]);
        else if (arg == "--omega" && i + 1 < argc) omega = std::stod(argv[++i]);
        else if (arg == "--h0-min" && i + 1 < argc) h0_min = std::stod(argv[++i]);
        else if (arg == "--h0-max" && i + 1 < argc) h0_max = std::stod(argv[++i]);
        else if (arg == "--num-h0" && i + 1 < argc) num_h0 = std::stoi(argv[++i]);
        else if (arg == "--ha" && i + 1 < argc) { fixed_ha = std::stod(argv[++i]); ha_min = ha_max = fixed_ha; num_ha = 1; }
        else if (arg == "--ha-min" && i + 1 < argc) ha_min = std::stod(argv[++i]);
        else if (arg == "--ha-max" && i + 1 < argc) ha_max = std::stod(argv[++i]);
        else if (arg == "--num-ha" && i + 1 < argc) num_ha = std::stoi(argv[++i]);
        else if (arg == "-o" || arg == "--output") { if (i + 1 < argc) out_file = argv[++i]; }
    }

    std::cout << "========================================\n";
    std::cout << "Rigid Wall Analysis (CPU OpenMP)\n";
    std::cout << "Alpha: " << alpha << ", Omega: " << omega << "\n";
    std::cout << "h0: [" << h0_min << ", " << h0_max << "] (" << num_h0 << " steps)\n";
    std::cout << "ha: [" << ha_min << ", " << ha_max << "] (" << num_ha << " steps)\n";
    std::cout << "Total evaluations: " << num_h0 * num_ha << "\n";
    std::cout << "========================================\n";

    const int steps_per_cycle = 1000;
    const double dt = dw::rigid::TWO_PI_VAL / steps_per_cycle;
    const int trans_cycles = 250;
    const int eval_cycles = 250;
    const double tol = 1e-5;

    std::vector<double> cos_t0(steps_per_cycle);
    std::vector<double> cos_thalf(steps_per_cycle);
    for (int s = 0; s < steps_per_cycle; ++s) {
        cos_t0[s] = std::cos(s * dt);
        cos_thalf[s] = std::cos((s + 0.5) * dt);
    }

    int total_tasks = num_h0 * num_ha;
    std::vector<Result> results(total_tasks);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int task = 0; task < total_tasks; ++task) {
        int i_h0 = task / num_ha;
        int i_ha = task % num_ha;

        double h0 = (num_h0 > 1) ? h0_min + i_h0 * (h0_max - h0_min) / (num_h0 - 1) : h0_min;
        double ha = (num_ha > 1) ? ha_min + i_ha * (ha_max - ha_min) / (num_ha - 1) : ha_min;

        double A = (alpha * h0) / omega;
        double B = alpha / omega;
        double C = (alpha * ha) / omega;

        double psi = 0.0;

        // Transient phase
        for (int p = 0; p < trans_cycles; ++p) {
            for (int s = 0; s < steps_per_cycle; ++s) {
                int inext = (s + 1) % steps_per_cycle;
                double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
        }

        // Check periodic limit cycle convergence
        double prev_strobe = std::fmod(psi, dw::rigid::TWO_PI_VAL);
        if (prev_strobe < 0) prev_strobe += dw::rigid::TWO_PI_VAL;
        bool converged = false;

        for (int p = 0; p < 100; ++p) {
            for (int s = 0; s < steps_per_cycle; ++s) {
                int inext = (s + 1) % steps_per_cycle;
                double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
            double curr_strobe = std::fmod(psi, dw::rigid::TWO_PI_VAL);
            if (curr_strobe < 0) curr_strobe += dw::rigid::TWO_PI_VAL;
            if (dw::rigid::circle_distance(curr_strobe, prev_strobe) < tol) {
                converged = true;
                break;
            }
            prev_strobe = curr_strobe;
        }

        // Evaluation phase
        double psi_eval_start = psi;
        double a0_acc = 0.0;
        for (int p = 0; p < eval_cycles; ++p) {
            for (int s = 0; s < steps_per_cycle; ++s) {
                a0_acc += std::cos(psi);
                int inext = (s + 1) % steps_per_cycle;
                double k1 = dt * (A - B * std::sin(psi) + C * cos_t0[s]);
                double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * cos_thalf[s]);
                double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * cos_thalf[s]);
                double k4 = dt * (A - B * std::sin(psi + k3) + C * cos_t0[inext]);
                psi += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            }
        }

        double total_tau = eval_cycles * dw::rigid::TWO_PI_VAL;
        double v_psi = (psi - psi_eval_start) / total_tau;
        double v_q = ((alpha * alpha + 1.0) / (2.0 * omega)) * h0 - (1.0 / (2.0 * alpha)) * v_psi;
        double a0_cos = a0_acc / (eval_cycles * steps_per_cycle);

        results[task] = {h0, ha, omega, prev_strobe, v_psi, v_q, a0_cos, converged};
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Execution completed in " << elapsed << " s.\n";

    // Write output text file
    std::ofstream out(out_file);
    out << std::setprecision(10);
    out << "h0 ha omega psi_eq v_psi v_q a0_cos converged\n";
    for (const auto& r : results) {
        out << r.h0 << " " << r.ha << " " << r.omega << " " << r.psi_eq << " "
            << r.v_psi << " " << r.v_q << " " << r.a0_cos << " " << r.converged << "\n";
    }
    std::cout << "Results saved to: " << out_file << "\n";

    return 0;
}

