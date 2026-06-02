#include "cu_pde_pseudospectral_solver.cuh"
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

// --- Slonczewski Physics Functors ---

struct SlonczewskiNonlinear
{
    double alpha, h, h0, Omega;

    // Stepper passes 't' automatically
    __host__ __device__
        thrust::complex<double>
        operator()(thrust::complex<double> z, double t) const
    {
        double current_h = h + h0 * cos(Omega * t);
        thrust::complex<double> prefix(alpha * 0.5, -0.5);

        // z.imag() represents the phase phi
        double term = alpha * current_h + sin(2.0 * z.imag());

        return prefix * term;
    }
};

struct SlonczewskiLinear
{
    double alpha;

    __host__ __device__
        thrust::complex<double>
        operator()(double k) const
    {
        // -0.5 * (alpha - i) * k^2
        return -0.5 * thrust::complex<double>(alpha, -1.0) * (k * k);
    }
};

// --- Buffer-based Initial Condition Functor ---

struct BufferInitialCondition
{
    const thrust::complex<double> *data_ptr;
    double dx;

    BufferInitialCondition(const thrust::complex<double> *d_ptr, double dx_val)
        : data_ptr(d_ptr), dx(dx_val) {}

    __host__ __device__
        thrust::complex<double>
        operator()(double x) const
    {
        // Map spatial coordinate x to buffer index i
        // Using +0.5 for rounding to nearest neighbor
        size_t i = static_cast<size_t>(x / dx + 0.5);
        return data_ptr[i];
    }
};

// --- Helpers ---

void save_binary_frame(std::ofstream &out, const thrust::device_vector<thrust::complex<double>> &d_vec)
{
    thrust::host_vector<thrust::complex<double>> h_vec = d_vec;
    out.write(reinterpret_cast<const char *>(h_vec.data()), h_vec.size() * sizeof(thrust::complex<double>));
}

// --- Main Simulation ---

int main()
{
    // 1. Simulation Setup
    const size_t n_modes = 10000;
    const double L = 10000.0;
    const double dt = 0.1;
    const double T_max = 1000.0;
    const int save_interval = 20;
    const double dx = L / n_modes;

    // 2. Physical Parameters
    double alpha = 0.27;
    double h = 4.0;
    double h0 = 0.5;
    double Omega = 2.0;

    // 3. Generate Random Initial Condition on the Host
    std::cout << "Generating initial condition on CPU..." << std::endl;
    thrust::host_vector<thrust::complex<double>> h_noise(n_modes);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.001, 0.001);

    for (size_t i = 0; i < n_modes; ++i)
    {
        h_noise[i] = thrust::complex<double>(dist(gen), dist(gen));
    }

    // 4. Transfer IC to Device
    // Important: d_noise must stay in scope until the stepper is constructed
    thrust::device_vector<thrust::complex<double>> d_noise = h_noise;

    // 5. Build Functors
    SlonczewskiNonlinear nonlin{alpha, h, h0, Omega};
    SlonczewskiLinear lin{alpha};
    BufferInitialCondition init(thrust::raw_pointer_cast(d_noise.data()), dx);

    // 6. Instantiate Stepper
    cu_PDE_pseudospectral_stepper<SlonczewskiNonlinear, BufferInitialCondition, SlonczewskiLinear>
        stepper(n_modes, L, dt, nonlin, init, lin);

    // 7. Output File
    std::ofstream outfile("slonczewski_evolution.bin", std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }

    // 8. Main Loop
    int total_steps = static_cast<int>(T_max / dt);
    std::cout << "Starting simulation on GPU (" << total_steps << " steps)..." << std::endl;

    for (int i = 0; i <= total_steps; ++i)
    {
        stepper.step(); // Time and z are updated internally

        if (i % save_interval == 0)
        {
            save_binary_frame(outfile, stepper.get_z());

            if (i % 1000 == 0)
            {
                std::cout << "Time: " << stepper.get_time()
                          << " | Progress: " << (100.0 * i / total_steps) << "%" << std::endl;
            }
        }
    }

    outfile.close();
    std::cout << "Simulation complete. Data written to slonczewski_evolution.bin" << std::endl;

    return 0;
}