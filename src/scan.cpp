#include <deque>
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

template <typename Backend>
void scan_h0_space(std::size_t size, double L, double alpha, double h0_min, double h0_max, int n_h0,
                   double ha, double omega, double dt, int W, int N, double tolerance, double eps,
                   unsigned int seed, const std::string& output_file) {
    using Real = typename Backend::PrecisionType::Real;
    using ComplexVector =
        typename Backend::template Vector<typename Backend::PrecisionType::Complex>;

    std::ofstream out(output_file);
    out << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out << "h0 time_to_settle avg_u_dot avg_phi_dot var_u\n";

    double h0_step = (n_h0 > 1) ? (h0_max - h0_min) / (n_h0 - 1) : 0.0;

    for (int iter = 0; iter < n_h0; ++iter) {
        double h0 = h0_min + iter * h0_step;

        dw::DomainWall<Backend> sim(size, L, alpha, h0, ha, omega, dt, eps, seed);
        ComplexVector u_old(size);
        std::vector<Real> step_velocities(W);
        std::deque<Real> window_variances;

        Real current_time = 0.0;
        bool settled = false;

        Backend::synchronize();

        // Phase 1: Wait until settled
        while (!settled) {
            Real mean_velocity = 0.0;

            for (int i = 0; i < W; ++i) {
                Backend::copy(sim.get_state().u, u_old);
                sim.step(current_time);

                Real diff_sq = Backend::compute_diff_sq(u_old, sim.get_state().u);
                step_velocities[i] = diff_sq / static_cast<Real>(dt * dt);
                mean_velocity += step_velocities[i];
            }

            mean_velocity /= W;
            Real window_var = 0.0;
            for (Real v : step_velocities) {
                window_var += (v - mean_velocity) * (v - mean_velocity);
            }
            window_var /= W;

            window_variances.push_back(window_var);
            if (window_variances.size() > N) {
                window_variances.pop_front();
            }

            if (window_variances.size() == N) {
                Real var_mean = 0.0;
                for (Real v : window_variances) var_mean += v;
                var_mean /= N;

                Real var_std = 0.0;
                for (Real v : window_variances) var_std += (v - var_mean) * (v - var_mean);
                var_std = std::sqrt(var_std / N);

                if (var_std < tolerance) {
                    settled = true;
                }
            }
        }

        // Phase 2: Accumulate averages for output
        Real sum_u_dot = 0.0, sum_phi_dot = 0.0, sum_var_u = 0.0;
        Real N_grid = static_cast<Real>(size);

        for (int i = 0; i < W; ++i) {
            sim.step(current_time);
            auto obs = Backend::compute_observables(sim.get_state().u);

            Real mean_sin2phi = obs.sin2phi / N_grid;
            Real mean_u = obs.u / N_grid;
            Real mean_u2 = obs.u2 / N_grid;

            Real h_val = static_cast<Real>(h0) +
                         static_cast<Real>(ha) * std::cos(static_cast<Real>(omega) * current_time);
            Real r_alpha = static_cast<Real>(alpha);

            sum_u_dot += 0.5 * (r_alpha * r_alpha * h_val + mean_sin2phi);
            sum_phi_dot += 0.5 * (r_alpha * h_val - r_alpha * mean_sin2phi);
            sum_var_u += (mean_u2 - (mean_u * mean_u));
        }

        out << h0 << " " << current_time << " " << (sum_u_dot / W) << " " << (sum_phi_dot / W)
            << " " << (sum_var_u / W) << "\n";

        std::cout << "Completed h0 = " << h0 << " at t = " << current_time << "\n";
    }

    out.close();
}

int main(int argc, char** argv) {
    if (argc < 15) {
        std::cerr
            << "Usage: " << argv[0]
            << " <size> <L> <alpha> <ha> <omega> <dt> <h0_min> <h0_max> <n_h0> <W> <N> <tolerance> "
               "<backend> <output_file> [eps=0.05] [seed=42] [precision=double]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double L = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    double ha = std::stod(argv[4]);
    double omega = std::stod(argv[5]);
    double dt = std::stod(argv[6]);
    double h0_min = std::stod(argv[7]);
    double h0_max = std::stod(argv[8]);
    int n_h0 = std::stoi(argv[9]);
    int W = std::stoi(argv[10]);
    int N = std::stoi(argv[11]);
    double tolerance = std::stod(argv[12]);
    std::string backend = argv[13];
    std::string output_file = argv[14];

    double eps = (argc > 15) ? std::stod(argv[15]) : 0.05;
    unsigned int seed = (argc > 16) ? std::stoul(argv[16]) : 42;
    std::string precision = (argc > 17) ? argv[17] : "double";

    if (backend == "cpu") {
        if (precision == "single") {
            scan_h0_space<dw::CpuBackend<dw::FftwSinglePrecision>>(
                size, L, alpha, h0_min, h0_max, n_h0, ha, omega, dt, W, N, tolerance, eps, seed,
                output_file);
        } else if (precision == "double") {
            scan_h0_space<dw::CpuBackend<dw::FftwDoublePrecision>>(
                size, L, alpha, h0_min, h0_max, n_h0, ha, omega, dt, W, N, tolerance, eps, seed,
                output_file);
        } else {
            std::cerr << "Invalid precision for CPU.\n";
            return 1;
        }
    } else if (backend == "gpu") {
#ifdef HAS_CUDA
        if (precision == "single") {
            scan_h0_space<dw::GpuBackend<dw::CudaSinglePrecision>>(
                size, L, alpha, h0_min, h0_max, n_h0, ha, omega, dt, W, N, tolerance, eps, seed,
                output_file);
        } else if (precision == "double") {
            scan_h0_space<dw::GpuBackend<dw::CudaDoublePrecision>>(
                size, L, alpha, h0_min, h0_max, n_h0, ha, omega, dt, W, N, tolerance, eps, seed,
                output_file);
        } else {
            std::cerr << "Invalid precision for GPU.\n";
            return 1;
        }
#else
        std::cerr << "Binary was compiled without CUDA support.\n";
        return 1;
#endif
    } else {
        std::cerr << "Invalid backend.\n";
        return 1;
    }

    return 0;
}