#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cpu_backend.hpp"
#include "fftw_precision.hpp"
#ifdef HAS_CUDA
#include "cuda_memory.cuh"
#include "cuda_precision.cuh"
#include "gpu_backend.cuh"
#endif
#include "domain_wall.cuh"

template <typename Backend>
void run_saturation_check(std::size_t size, double L, double alpha, double dt, double omega, double eps, unsigned int seed) {
    std::vector<double> h0_batch = {4.0};
    std::vector<double> ha_batch = {0.0};

    dw::DomainWall<Backend> sim(size, L, alpha, h0_batch, ha_batch, omega, dt, eps, seed);

    std::ofstream out("saturation_rugosity.txt");
    out << std::setprecision(10);
    out << "time step running_avg_rugosity\n";

    double current_time = 0.0;
    unsigned long long step = 0;
    double next_checkpoint = 1.0;
    double max_time = std::pow(2.0, 20.0);

    // Allocate metric accumulators
    typename Backend::template Vector<double> d_acc_vu(1), d_acc_vphi(1), d_acc_rugosity(1);
    Backend::fill_zero(d_acc_vu.data(), 1);
    Backend::fill_zero(d_acc_vphi.data(), 1);
    Backend::fill_zero(d_acc_rugosity.data(), 1);

    std::cout << "Starting saturation check up to t = " << max_time << "...\n";

    while (current_time < max_time) {
        Backend::copy(sim.get_state().u.data(), sim.get_state().u_prev.data(), size);

        sim.step(current_time);
        step++;

        Backend::accumulate_sweep_metrics(sim.get_state().u.data(), sim.get_state().u_prev.data(),
                                          d_acc_vu.data(), d_acc_vphi.data(), d_acc_rugosity.data(),
                                          size, 1, dt);

        if (current_time >= next_checkpoint) {
            Backend::synchronize();

            auto h_rugosity = Backend::download_array(d_acc_rugosity.data(), 1);

            // Divide by total elapsed time steps
            double avg_rugosity = h_rugosity[0] / static_cast<double>(step);

            out << current_time << " " << step << " " << avg_rugosity << "\n";
            out.flush();

            std::cout << "t = " << current_time << " | Running Avg Rugosity = " << avg_rugosity
                      << std::endl;

            next_checkpoint *= 2.0;
        }
    }

    out.close();
    std::cout << "Completed. Data saved to saturation_rugosity.txt\n";
}

int main(int argc, char** argv) {
    // Default physics parameters (can be overridden via CLI if needed)
    std::size_t size = (argc > 1) ? std::stoull(argv[1]) : 1024;
    double L = (argc > 2) ? std::stod(argv[2]) : 50.0;
    double alpha = (argc > 3) ? std::stod(argv[3]) : 0.1;
    double dt = (argc > 4) ? std::stod(argv[4]) : 0.01;
    std::string backend = (argc > 5) ? argv[5] : "gpu";

    double omega = 1.0;
    double eps = 0.0005;
    unsigned int seed = 42;

    if (backend == "cpu") {
        run_saturation_check<dw::CpuBackend<dw::FftwDoublePrecision>>(size, L, alpha, dt, omega, eps, seed);
    } else {
#ifdef HAS_CUDA
        run_saturation_check<dw::GpuBackend<dw::CudaDoublePrecision>>(size, L, alpha, dt, omega, eps, seed);
#else
        std::cerr << "Error: GPU backend requested but CUDA is not enabled.\n";
        return 1;
#endif
    }

    return 0;
}
