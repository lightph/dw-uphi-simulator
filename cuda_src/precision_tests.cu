#include "cu_domain_wall.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

void run_benchmark_and_save(const std::string &output_filename, cuda_math::REAL system_size, cuda_math::REAL h0)
{
    cuda_math::REAL alpha = 0.27;
    cuda_math::REAL h = 0.5;
    cuda_math::REAL Omega = 2.0;

    cuda_math::REAL min_resolution = 0.05;
    size_t steps_per_block = 1000;

    cuda_math::REAL T_phys = std::min(2.0 * M_PI / Omega,
                                      (h > 1.01) ? 2.0 * M_PI / (alpha * 0.5 * std::sqrt(h * h - 1.0)) : 1e9);

    cuda_math::REAL T_cfl = (min_resolution * min_resolution) / (alpha + 1e-6);

    cuda_math::REAL dt = std::min(T_phys / 50.0, T_cfl * 0.5);

    size_t n_samples = CufftUtils::getOptimalSize(static_cast<size_t>(std::ceil(system_size / min_resolution)));
    std::vector<cuda_math::COMPLEX> h_ic(n_samples, cuda_math::COMPLEX(1.0, 0.0));
    cuda_math::COMPLEX *d_ic_ptr;
    CUDA_CHECK(cudaMalloc(&d_ic_ptr, n_samples * sizeof(cuda_math::COMPLEX)));
    CUDA_CHECK(cudaMemcpy(d_ic_ptr, h_ic.data(), n_samples * sizeof(cuda_math::COMPLEX), cudaMemcpyHostToDevice));

    RandomInitialCondition ic(d_ic_ptr, system_size / n_samples);

    // Instantiate Solver
    cu_domain_wall<RandomInitialCondition> solver(
        alpha, h, h0, Omega, dt, system_size, min_resolution, ic);

    // Warm-up and Graph Creation
    solver.create_step_graph(10);
    solver.run_block(10);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Performance Measurement
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    solver.run_block(steps_per_block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Executed " << steps_per_block << " steps in " << milliseconds << " ms." << std::endl;
    std::cout << "Average time per step: " << milliseconds / steps_per_block << " ms." << std::endl;

    // Save Results for Error Analysis
    const auto &d_z = solver.get_z();
    std::vector<cuda_math::COMPLEX> h_z(n_samples);
    CUDA_CHECK(cudaMemcpy(h_z.data(), thrust::raw_pointer_cast(d_z.data()), n_samples * sizeof(cuda_math::COMPLEX), cudaMemcpyDeviceToHost));

    std::ofstream out_file(output_filename, std::ios::binary);
    out_file.write(reinterpret_cast<const char *>(h_z.data()), n_samples * sizeof(cuda_math::COMPLEX));
    out_file.close();

    CUDA_CHECK(cudaFree(d_ic_ptr));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <system_size> <h0>" << std::endl;
        return EXIT_FAILURE;
    }

    cuda_math::REAL system_size = static_cast<cuda_math::REAL>(std::stod(argv[1]));
    cuda_math::REAL h0 = static_cast<cuda_math::REAL>(std::stod(argv[2]));

    // Format h0 to avoid long decimal strings in the filename
    std::ostringstream h0_stream;
    h0_stream << std::fixed << std::setprecision(2) << h0;

    // Construct unique filename based on precision, L, and h0
#ifdef DOUBLE_PRECISION
    std::string filename = "output_dp_L" + std::to_string(static_cast<int>(system_size)) + "_h0_" + h0_stream.str() + ".bin";
    std::cout << "Running benchmark in DOUBLE precision..." << std::endl;
#else
    std::string filename = "output_sp_L" + std::to_string(static_cast<int>(system_size)) + "_h0_" + h0_stream.str() + ".bin";
    std::cout << "Running benchmark in SINGLE precision..." << std::endl;
#endif

    std::cout << "System size: " << system_size << ", h0: " << h0 << std::endl;
    std::cout << "Saving output to: " << filename << std::endl;

    run_benchmark_and_save(filename, system_size, h0);
    return 0;
}