#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Assuming your stepper and state classes are here
#include "state.hpp"
#include "stepper.hpp"

// Provided backend and precision headers
#include "cpu_backend.hpp"
#include "cuda_precision.cuh"
#include "fftw_precision.hpp"
#include "gpu_backend.cuh"

namespace dw {

template <typename Backend>
struct NonlinearEvaluator {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    Real alpha, h0, ha, omega, t;

    void operator()(const ComplexVector& z, ComplexVector& nl_out) const {
        Real h_val = h0 + ha * std::cos(omega * t);

        // N(z) = (alpha - i)/2 * (alpha*h + i*sin(2*phi))
        Complex prefactor(alpha / 2.0, -1.0 / 2.0);

        // Unified Memory allows this to run on host even for CudaVector.
        // For strict device-side execution, this loop would be mapped to a kernel.
        for (std::size_t j = 0; j < z.size(); ++j) {
            Real phi = -z[j].imag();  // z = u - i\phi
            Complex bracket(alpha * h_val, std::sin(2.0 * phi));
            nl_out[j] = prefactor * bracket;
        }
    }
};

template <typename Backend>
struct LinearStepper {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    Real alpha, dt;
    const Real* k_squared;

    void operator()(ComplexVector& z_hat, const ComplexVector& nl_hat, double dt_in) const {
        Complex dt_c(dt, 0.0);
        Complex one(1.0, 0.0);
        Complex half(0.5, 0.0);

        for (std::size_t j = 0; j < z_hat.size(); ++j) {
            Real k2 = k_squared[j];
            // L_k = -(alpha - i)/2 * k^2
            Complex L_k(-alpha * k2 / 2.0, k2 / 2.0);

            Complex half_dt_L_k = half * dt_c * L_k;
            Complex numerator = (one + half_dt_L_k) * z_hat[j] + dt_c * nl_hat[j];
            Complex denominator = one - half_dt_L_k;

            z_hat[j] = numerator / denominator;
        }
    }
};

template <typename Backend>
void run_simulation(std::size_t size, double alpha_in, double h0_in, double ha_in, double omega_in,
                    double dt_in, const std::string& output_file) {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;

    Real alpha = static_cast<Real>(alpha_in);
    Real h0 = static_cast<Real>(h0_in);
    Real ha = static_cast<Real>(ha_in);
    Real omega = static_cast<Real>(omega_in);
    Real dt = static_cast<Real>(dt_in);

    State<Backend> state(size);
    PseudospectralStepper<Backend> stepper(size);

    // Compute frequencies. Standard FFT layout: 0, 1, ..., N/2, -N/2+1, ..., -1
    typename Backend::template Vector<Real> k_squared(size);
    for (std::size_t i = 0; i < size; ++i) {
        long long k =
            (i <= size / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size);
        k_squared[i] = static_cast<Real>(k * k);
    }
    const Real* d_k_squared = k_squared.data();

    // Initial conditions: weak random perturbation
    std::mt19937 gen(42);
    std::uniform_real_distribution<Real> dist(-0.05, 0.05);
    for (std::size_t i = 0; i < size; ++i) {
        Real u = dist(gen);
        Real phi = dist(gen);
        state.u[i] = Complex(u, -phi);
    }

    Backend::synchronize();

    std::ofstream out(output_file);
    out << std::setprecision(std::numeric_limits<Real>::max_digits10);

    Real current_time = 0.0;
    unsigned long long step_count = 1;
    unsigned long long target_step = 2;
    int n = 1;
    const unsigned long long max_steps = 1ULL << 20;

    while (step_count <= max_steps) {
        NonlinearEvaluator<Backend> nonlin{alpha, h0, ha, omega, current_time};
        LinearStepper<Backend> lin{alpha, dt, d_k_squared};

        stepper.step(state, dt, lin, nonlin);
        current_time += dt;

        if (step_count == target_step) {
            Backend::synchronize();

            out << "n=" << n << " step=" << step_count << " t=" << current_time << "\n";
            for (std::size_t i = 0; i < size; ++i) {
                out << state.u[i].real() << " " << -state.u[i].imag() << "\n";
            }

            n++;
            target_step = 1ULL << n;
        }
        step_count++;
    }
    out.close();
}

}  // namespace dw

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <size> <alpha> <h0> <ha> <omega> <dt> <backend> <precision> <output_file>\n";
        return 1;
    }

    std::size_t size = std::stoull(argv[1]);
    double alpha = std::stod(argv[2]);
    double h0 = std::stod(argv[3]);
    double ha = std::stod(argv[4]);
    double omega = std::stod(argv[5]);
    double dt = std::stod(argv[6]);
    std::string backend = argv[7];
    std::string precision = argv[8];
    std::string output_file = argv[9];

    if (backend == "cpu") {
        if (precision == "single") {
            dw::run_simulation<dw::CpuBackend<dw::FftwSinglePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
        } else if (precision == "double") {
            dw::run_simulation<dw::CpuBackend<dw::FftwDoublePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
        } else {
            std::cerr << "Invalid precision. Use 'single' or 'double'.\n";
            return 1;
        }
    } else if (backend == "gpu") {
        if (precision == "single") {
            dw::run_simulation<dw::GpuBackend<dw::CudaSinglePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
        } else if (precision == "double") {
            dw::run_simulation<dw::GpuBackend<dw::CudaDoublePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
        } else {
            std::cerr << "Invalid precision. Use 'single' or 'double'.\n";
            return 1;
        }
    } else {
        std::cerr << "Invalid backend. Use 'cpu' or 'gpu'.\n";
        return 1;
    }

    return 0;
}