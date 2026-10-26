#include <iostream>
#include <string>
#include <vector>

#include "cpu_backend.hpp"
#include "fftw_precision.hpp"
#ifdef HAS_CUDA
#include "cuda_precision.cuh"
#include "gpu_backend.cuh"
#endif
#include "domain_wall.cuh"
#include "sweep_observer.hpp"

template <typename Backend>
void run_sweep(std::size_t size, double L, double alpha, double omega, double dt,
               double transient_time, double eval_time, double h0_val, double ha_min,
               double ha_max, std::size_t ha_steps, const std::string& output_file,
               double eps, unsigned int seed) {
    std::vector<double> h0_batch(ha_steps, h0_val);
    std::vector<double> ha_batch(ha_steps);

    if (ha_steps == 1) {
        ha_batch[0] = ha_min;
    } else {
        double d_ha = (ha_max - ha_min) / static_cast<double>(ha_steps - 1);
        for (std::size_t i = 0; i < ha_steps; ++i) {
            ha_batch[i] = ha_min + i * d_ha;
        }
    }

    dw::DomainWall<Backend> sim(size, L, alpha, h0_batch, ha_batch, omega, dt, eps, seed);
    dw::SweepObserver<Backend> observer(sim, output_file);
    observer.run_sweep(transient_time, eval_time);
}

int main(int argc, char** argv) {
    if (argc < 13) {
        std::cerr
            << "Usage: " << argv[0] << " <size> <L> <alpha> <omega> <dt> "
            << "<transient_time> <eval_time> <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend=gpu]\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double L = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    double omega = std::stod(argv[4]);
    double dt = std::stod(argv[5]);

    double transient_time = std::stod(argv[6]);
    double eval_time = std::stod(argv[7]);

    double h0_val = std::stod(argv[8]);
    double ha_min = std::stod(argv[9]);
    double ha_max = std::stod(argv[10]);
    std::size_t ha_steps = std::stoull(argv[11]);
    std::string output_file = argv[12];
    std::string backend = (argc > 13) ? argv[13] : "gpu";

    double eps = 0.0005;
    unsigned int seed = 42;

    if (backend == "cpu") {
        run_sweep<dw::CpuBackend<dw::FftwDoublePrecision>>(
            size, L, alpha, omega, dt, transient_time, eval_time, h0_val, ha_min, ha_max,
            ha_steps, output_file, eps, seed);
    } else {
#ifdef HAS_CUDA
        run_sweep<dw::GpuBackend<dw::CudaDoublePrecision>>(
            size, L, alpha, omega, dt, transient_time, eval_time, h0_val, ha_min, ha_max,
            ha_steps, output_file, eps, seed);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
