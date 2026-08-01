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

#include "cpu_backend.hpp"
#include "fftw_precision.hpp"
#include "state.hpp"
#include "stepper.hpp"

#ifdef HAS_CUDA
#include "cuda_precision.cuh"
#include "gpu_backend.cuh"
#endif

namespace dw {

// ============================================================================
// CPU Functors (Default)
// ============================================================================
template <typename Backend>
struct NonlinearEvaluator {
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    Real alpha, h0, ha, omega, t;

    void operator()(const ComplexVector& z, ComplexVector& nl_out) const {
        Real h_val = h0 + ha * std::cos(omega * t);
        Complex prefactor(alpha / Real(2.0), Real(-1.0) / Real(2.0));

#pragma omp parallel for
        for (std::size_t j = 0; j < z.size(); ++j) {
            Real phi = -z[j].imag();
            Complex bracket(alpha * h_val, std::sin(Real(2.0) * phi));
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
        Complex dt_c(static_cast<Real>(dt_in), Real(0.0));
        Complex one(Real(1.0), Real(0.0));
        Complex half(Real(0.5), Real(0.0));

        // FFT Normalization factor absorbed here
        Real norm_factor = Real(1.0) / static_cast<Real>(z_hat.size());

#pragma omp parallel for
        for (std::size_t j = 0; j < z_hat.size(); ++j) {
            Real k2 = k_squared[j];
            Complex L_k(-alpha * k2 / Real(2.0), k2 / Real(2.0));

            Complex half_dt_L_k = half * dt_c * L_k;
            Complex numerator = (one + half_dt_L_k) * z_hat[j] + dt_c * nl_hat[j];
            Complex denominator = one - half_dt_L_k;

            z_hat[j] = (numerator / denominator) * norm_factor;
        }
    }
};

// ============================================================================
// GPU Functors (Specialized for GpuBackend)
// ============================================================================
#ifdef HAS_CUDA

template <typename Real, typename Complex>
__global__ void nonlinear_kernel(const Complex* z, Complex* nl_out, Real alpha, Real h_val,
                                 std::size_t size) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
        Real phi = -z[j].imag();
        Complex prefactor(alpha / Real(2.0), Real(-1.0) / Real(2.0));
        Complex bracket(alpha * h_val, std::sin(Real(2.0) * phi));
        nl_out[j] = prefactor * bracket;
    }
}

template <typename Precision>
struct NonlinearEvaluator<GpuBackend<Precision>> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using ComplexVector = CudaVector<Complex>;

    Real alpha, h0, ha, omega, t;

    void operator()(const ComplexVector& z, ComplexVector& nl_out) const {
        Real h_val = h0 + ha * std::cos(omega * t);
        int blockSize = 256;
        int numBlocks = (z.size() + blockSize - 1) / blockSize;
        nonlinear_kernel<<<numBlocks, blockSize>>>(z.data(), nl_out.data(), alpha, h_val, z.size());
    }
};

template <typename Real, typename Complex>
__global__ void linear_kernel(Complex* z_hat, const Complex* nl_hat, const Real* k_squared,
                              Real alpha, Real dt, std::size_t size) {
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size) {
        Real k2 = k_squared[j];
        Complex L_k(-alpha * k2 / Real(2.0), k2 / Real(2.0));

        Complex dt_c(dt, Real(0.0));
        Complex one(Real(1.0), Real(0.0));
        Complex half(Real(0.5), Real(0.0));

        Complex half_dt_L_k = half * dt_c * L_k;
        Complex numerator = (one + half_dt_L_k) * z_hat[j] + dt_c * nl_hat[j];
        Complex denominator = one - half_dt_L_k;

        Real norm_factor = Real(1.0) / static_cast<Real>(size);
        z_hat[j] = (numerator / denominator) * norm_factor;
    }
}

template <typename Precision>
struct LinearStepper<GpuBackend<Precision>> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using ComplexVector = CudaVector<Complex>;

    Real alpha, dt;
    const Real* k_squared;

    void operator()(ComplexVector& z_hat, const ComplexVector& nl_hat, double dt_in) const {
        int blockSize = 256;
        int numBlocks = (z_hat.size() + blockSize - 1) / blockSize;
        linear_kernel<<<numBlocks, blockSize>>>(z_hat.data(), nl_hat.data(), k_squared, alpha,
                                                static_cast<Real>(dt_in), z_hat.size());
    }
};

#endif  // HAS_CUDA

// ============================================================================
// Simulation Execution
// ============================================================================
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

    typename Backend::template Vector<Real> k_squared(size);
    for (std::size_t i = 0; i < size; ++i) {
        long long k =
            (i <= size / 2) ? i : static_cast<long long>(i) - static_cast<long long>(size);
        k_squared[i] = static_cast<Real>(k * k);
    }

    // Explicit sync to device for Unified Memory pre-fetching
    Backend::synchronize();
    const Real* d_k_squared = k_squared.data();

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
            std::cerr << "Invalid precision for CPU.\n";
            return 1;
        }
    } else if (backend == "gpu") {
#ifdef HAS_CUDA
        if (precision == "single") {
            dw::run_simulation<dw::GpuBackend<dw::CudaSinglePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
        } else if (precision == "double") {
            dw::run_simulation<dw::GpuBackend<dw::CudaDoublePrecision>>(size, alpha, h0, ha, omega,
                                                                        dt, output_file);
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