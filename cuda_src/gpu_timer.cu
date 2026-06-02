#include "cu_domain_wall_experiments.cuh"
#include "cu_domain_wall.cuh"
#include "types.cuh"
#include "cufftHandler.cuh"
#include <chrono>
#include <iostream>
#include <cmath>
int main(int argc, char *argv[])
{
    if (argc < 14)
    {
        std::cerr << "Usage: " << argv[0]
                  << " system_size single_run_steps Omega h0 hmin hmax hnumber "
                  << "alpha dt min_resolution max_steps periodsphi periods_in_block\n";
        return 1;
    }

    const cuda_math::REAL system_size = static_cast<cuda_math::REAL>(std::stod(argv[1]));
    const size_t single_run_steps = std::stoull(argv[2]);
    const cuda_math::REAL Omega = static_cast<cuda_math::REAL>(std::stod(argv[3]));
    const cuda_math::REAL h0 = static_cast<cuda_math::REAL>(std::stod(argv[4]));
    const cuda_math::REAL hmin = static_cast<cuda_math::REAL>(std::stod(argv[5]));
    const cuda_math::REAL hmax = static_cast<cuda_math::REAL>(std::stod(argv[6]));
    const int hnumber = std::stoi(argv[7]);
    const cuda_math::REAL alpha = static_cast<cuda_math::REAL>(std::stod(argv[8]));
    const cuda_math::REAL dt = static_cast<cuda_math::REAL>(std::stod(argv[9]));
    const cuda_math::REAL min_resolution = static_cast<cuda_math::REAL>(std::stod(argv[10]));
    const size_t max_steps = std::stoull(argv[11]);
    const size_t periodsphi = std::stoull(argv[12]);
    const size_t periods_in_block = std::stoull(argv[13]);

    const cuda_math::REAL h_step = (hnumber > 1) ? (hmax - hmin) / (hnumber - 1) : 0.0;
    const cuda_math::REAL PI_VAL = std::acos(-1.0);

    RandomInitialCondition initial_ic(0.02, 0);
    cu_domain_wall<RandomInitialCondition> wall(alpha, hmin, h0, Omega, dt, system_size, min_resolution, initial_ic);

    cudaDeviceSynchronize();
    auto sweep_start_time = std::chrono::steady_clock::now();

    for (int k = 0; k < hnumber; ++k)
    {
        cuda_math::REAL h = hmin + (k * h_step);
        cuda_math::REAL T_phys = std::min(2.0 * PI_VAL / Omega, (h > 1.01) ? 2.0 * PI_VAL / (alpha * 0.5 * std::sqrt(h * h - 1.0)) : 1e9);
        cuda_math::REAL target_dt = std::min(T_phys / 50.0, (min_resolution * min_resolution) / (alpha + 1e-6) * 0.5);

        wall.set_dt(target_dt);
        wall.set_h(h);
        wall.reset(RandomInitialCondition(0.02, k));
        wall.set_time(0.0);

        SweepMetricsExperiment exp;
        run_simulation(wall, exp, target_dt, max_steps, periodsphi, periods_in_block);

        if (std::isnan(exp.final_speed_mean_phi))
        {
            std::cerr << "[Timer Warning] H=" << h << " failed to converge.\n";
        }
    }

    cudaDeviceSynchronize();
    auto sweep_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> sweep_elapsed = sweep_end_time - sweep_start_time;

    // --- Single Performance Run ---
    wall.set_h(1.0);
    wall.set_dt(0.01);
    wall.reset(RandomInitialCondition(0.02, 999));
    wall.set_time(0.0);
    cudaDeviceSynchronize();

    auto single_start_time = std::chrono::steady_clock::now();

    // Chunk the runs to prevent CUDA Graph memory exhaustion
    const size_t chunk_size = 1000;
    size_t num_chunks = single_run_steps / chunk_size;
    size_t remainder = single_run_steps % chunk_size;

    for (size_t i = 0; i < num_chunks; ++i)
    {
        wall.run_block(chunk_size);
    }
    if (remainder > 0)
    {
        wall.run_block(remainder);
    }

    cudaDeviceSynchronize();
    auto single_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> single_elapsed = single_end_time - single_start_time;

    std::cout << sweep_elapsed.count() << " " << single_elapsed.count() << std::endl;
    return 0;
}