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

template <typename Backend>
void run_window_analysis(std::size_t size, double L, double alpha, double h0, double ha,
                         double omega, double dt, int W, int N, double eps, unsigned int seed,
                         const std::string& output_file) {
    using Real = typename Backend::PrecisionType::Real;

    dw::DomainWall<Backend> sim(size, L, alpha, h0, ha, omega, dt, eps, seed);

    std::ofstream out(output_file);
    out << std::setprecision(std::numeric_limits<Real>::max_digits10);
    out << "window_index time mean_variance drift\n";

    Real current_time = 0.0;
    Backend::synchronize();

    Real previous_mean_var = -1.0;

    for (int n = 0; n < N; ++n) {
        Real sum_var_u = 0.0;

        for (int i = 0; i < W; ++i) {
            sim.step(current_time);

            auto obs = Backend::compute_observables(sim.get_state().u);
            Real N_grid = static_cast<Real>(sim.get_size());
            Real mean_u = obs.u / N_grid;
            Real mean_u2 = obs.u2 / N_grid;
            Real var_u = mean_u2 - (mean_u * mean_u);

            sum_var_u += var_u;
        }

        Real current_mean_var = sum_var_u / W;
        Real drift =
            (previous_mean_var >= 0.0) ? std::abs(current_mean_var - previous_mean_var) : 0.0;

        out << n << " " << current_time << " " << current_mean_var << " " << drift << "\n";

        previous_mean_var = current_mean_var;
    }
    out.close();
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <size> <L> <alpha> <ha> <omega> <dt> <h0> <W> <N> <backend> <output_file> "
                     "[eps=0.05] [seed=42] [precision=double]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double L = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    double ha = std::stod(argv[4]);
    double omega = std::stod(argv[5]);
    double dt = std::stod(argv[6]);
    double h0 = std::stod(argv[7]);
    int W = std::stoi(argv[8]);
    int N = std::stoi(argv[9]);
    std::string backend = argv[10];
    std::string output_file = argv[11];

    double eps = (argc > 12) ? std::stod(argv[12]) : 0.05;
    unsigned int seed = (argc > 13) ? std::stoul(argv[13]) : 42;
    std::string precision = (argc > 14) ? argv[14] : "double";

    if (backend == "cpu") {
        if (precision == "single") {
            run_window_analysis<dw::CpuBackend<dw::FftwSinglePrecision>>(
                size, L, alpha, h0, ha, omega, dt, W, N, eps, seed, output_file);
        } else if (precision == "double") {
            run_window_analysis<dw::CpuBackend<dw::FftwDoublePrecision>>(
                size, L, alpha, h0, ha, omega, dt, W, N, eps, seed, output_file);
        } else {
            std::cerr << "Invalid precision for CPU.\n";
            return 1;
        }
    } else if (backend == "gpu") {
#ifdef HAS_CUDA
        if (precision == "single") {
            run_window_analysis<dw::GpuBackend<dw::CudaSinglePrecision>>(
                size, L, alpha, h0, ha, omega, dt, W, N, eps, seed, output_file);
        } else if (precision == "double") {
            run_window_analysis<dw::GpuBackend<dw::CudaDoublePrecision>>(
                size, L, alpha, h0, ha, omega, dt, W, N, eps, seed, output_file);
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