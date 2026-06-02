#include "FftwHandler.h"
#include "slonczewski_experiments.h"
#include "slonczewski_wall.h"
#include <chrono>
#include <iostream>
#include <cmath>

int main(int argc, char *argv[])
{
    fftw_init_threads();

    if (argc < 14)
    {
        std::cerr << "Usage: " << argv[0]
                  << " system_size single_run_steps Omega h0 hmin hmax hnumber "
                  << "alpha dt min_resolution max_steps periodsphi periods_in_block\n";
        return 1;
    }

    const FFTW_REAL system_size = static_cast<FFTW_REAL>(std::stod(argv[1]));
    const size_t single_run_steps = std::stoull(argv[2]);
    const FFTW_REAL Omega = static_cast<FFTW_REAL>(std::stod(argv[3]));
    const FFTW_REAL h0 = static_cast<FFTW_REAL>(std::stod(argv[4]));
    const FFTW_REAL hmin = static_cast<FFTW_REAL>(std::stod(argv[5]));
    const FFTW_REAL hmax = static_cast<FFTW_REAL>(std::stod(argv[6]));
    const int hnumber = std::stoi(argv[7]);
    const FFTW_REAL alpha = static_cast<FFTW_REAL>(std::stod(argv[8]));
    const FFTW_REAL dt = static_cast<FFTW_REAL>(std::stod(argv[9]));
    const FFTW_REAL min_resolution = static_cast<FFTW_REAL>(std::stod(argv[10]));
    const size_t max_steps = std::stoull(argv[11]);
    const size_t periodsphi = std::stoull(argv[12]);
    const size_t periods_in_block = std::stoull(argv[13]);

    const FFTW_REAL h_step = (hnumber > 1) ? (hmax - hmin) / (hnumber - 1) : 0.0;
    const FFTW_REAL PI_VAL = std::acos(-1.0);

    Slonczewski_Wall<RandomInitialCondition> wall(alpha, hmin, h0, Omega, dt, system_size, min_resolution, RandomInitialCondition(0.02, 0));

    auto sweep_start_time = std::chrono::steady_clock::now();

    for (int k = 0; k < hnumber; ++k)
    {
        FFTW_REAL h = hmin + (k * h_step);
        FFTW_REAL T_phys = std::min(static_cast<FFTW_REAL>(2.0 * PI_VAL / Omega), (h > 1.01) ? static_cast<FFTW_REAL>(2.0 * PI_VAL / (alpha * 0.5 * std::sqrt(h * h - 1.0))) : static_cast<FFTW_REAL>(1e9));
        FFTW_REAL target_dt = std::min(static_cast<FFTW_REAL>(T_phys / 50.0), static_cast<FFTW_REAL>((min_resolution * min_resolution) / (alpha + 1e-6) * 0.5));

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

    auto sweep_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> sweep_elapsed = sweep_end_time - sweep_start_time;

    // --- Single Performance Run ---
    wall.set_h(1.0);
    wall.set_dt(0.01);
    wall.reset(RandomInitialCondition(0.02, 999));
    wall.set_time(0.0);

    auto single_start_time = std::chrono::steady_clock::now();

    // Chunked for consistency with GPU execution
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

    auto single_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> single_elapsed = single_end_time - single_start_time;

    std::cout << sweep_elapsed.count() << " " << single_elapsed.count() << std::endl;

    fftw_cleanup_threads();
    fftw_cleanup();
    return 0;
}