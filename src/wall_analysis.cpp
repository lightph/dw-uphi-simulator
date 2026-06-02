#include "FftwHandler.h"
#include "filesaver.h"
#include "slonczewski_experiments.h"
#include "slonczewski_wall.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cmath>

int main(int argc, char *argv[])
{
    fftw_init_threads();

    // Check for mandatory arguments
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " Omega h0 system_size hmin hmax hnumber "
                  << "[alpha dt min_resolution max_steps periodsphi periods_in_block]\n";
        return 1;
    }

    // Mandatory parameters
    const FFTW_REAL Omega = static_cast<FFTW_REAL>(std::stod(argv[1]));
    const FFTW_REAL h0 = static_cast<FFTW_REAL>(std::stod(argv[2]));
    const FFTW_REAL system_size = static_cast<FFTW_REAL>(std::stod(argv[3]));
    const FFTW_REAL hmin = static_cast<FFTW_REAL>(std::stod(argv[4]));
    const FFTW_REAL hmax = static_cast<FFTW_REAL>(std::stod(argv[5]));
    const int hnumber = std::stoi(argv[6]);

    // Optional parameters with default fallbacks
    const FFTW_REAL alpha = (argc > 7) ? static_cast<FFTW_REAL>(std::stod(argv[7])) : 0.27;
    FFTW_REAL dt = (argc > 8) ? static_cast<FFTW_REAL>(std::stod(argv[8])) : 0.1;
    const FFTW_REAL min_resolution = (argc > 9) ? static_cast<FFTW_REAL>(std::stod(argv[9])) : 0.1;

    // Block-based stroboscopic parameters
    const size_t max_steps = (argc > 10) ? std::stoull(argv[10]) : 500000;
    const size_t periodsphi = (argc > 11) ? std::stoull(argv[11]) : 100;
    const size_t periods_in_block = (argc > 12) ? std::stoull(argv[12]) : 1;

    // Calculate step size (avoid division by zero if hnumber is 1)
    const FFTW_REAL h_step = (hnumber > 1) ? (hmax - hmin) / (hnumber - 1) : 0.0;

    // Build the comprehensive JSON Log map
    std::map<std::string, std::string> sweep_params = {
        {"Type", "Single_Parameter_Sweep"},

        // Fixed Physics & Numerical Parameters
        {"Alpha", std::to_string(alpha)},
        {"TimeStep_dt", std::to_string(dt)},
        {"SystemSize", std::to_string(system_size)},
        {"SpatialRes_min", std::to_string(min_resolution)},
        {"MaxSteps", std::to_string(max_steps)},
        {"PeriodsPhi", std::to_string(periodsphi)},
        {"PeriodsInBlock", std::to_string(periods_in_block)},

        // Fixed Sweep Parameters
        {"Omega", std::to_string(Omega)},
        {"h0", std::to_string(h0)},

        // Sweep Boundaries: h
        {"h_Min", std::to_string(hmin)},
        {"h_Max", std::to_string(hmax)},
        {"h_Points", std::to_string(hnumber)},
    };

    BinarySaver saver("../output", "cpu_h_sweep", sweep_params);

    auto sweep_start_time = std::chrono::steady_clock::now();

    for (int k = 0; k < hnumber; ++k)
    {
        FFTW_REAL h = hmin + (k * h_step);
        int seed = k;

        RandomInitialCondition ic(0.02, seed);

        Slonczewski_Wall<RandomInitialCondition> wall(
            alpha, h, h0, Omega, dt, system_size, min_resolution, ic);

        // Inside the CPU sweep loop
        FFTW_REAL T_phys = std::min(static_cast<FFTW_REAL>(2.0 * std::acos(-1.0) / Omega),
                                    (h > 1.01) ? static_cast<FFTW_REAL>(2.0 * std::acos(-1.0) / (alpha * 0.5 * std::sqrt(h * h - 1.0))) : static_cast<FFTW_REAL>(1e9));
        FFTW_REAL T_cfl = (min_resolution * min_resolution) / (alpha + 1e-6);
        FFTW_REAL target_dt = std::min(T_phys / 50.0, T_cfl * 1.0);

        dt = target_dt;
        wall.set_dt(target_dt);
        SweepMetricsExperiment exp;

        auto run_start_time = std::chrono::steady_clock::now();

        run_simulation(wall, exp, dt, max_steps, periodsphi, periods_in_block);

        auto run_end_time = std::chrono::steady_clock::now();

        if (std::isnan(exp.final_speed_mean_phi))
        {
            std::cerr << "\n[Warning] Run " << (k + 1) << " (H=" << h << ") failed to converge. Saving NaNs to output.\n";
        }

        std::chrono::duration<double> run_duration = run_end_time - run_start_time;

        std::vector<double> row_data = {
            static_cast<double>(alpha),
            static_cast<double>(h),
            static_cast<double>(h0),
            static_cast<double>(Omega),
            static_cast<double>(exp.final_time_avg_rugosity),
            static_cast<double>(exp.final_speed_mean_u),
            static_cast<double>(exp.final_speed_mean_phi),
            static_cast<double>(run_duration.count())};

        saver.saveRow(DataRow(std::move(row_data)));

        // Force write to disk every 10 steps to prevent data loss on crash
        if ((k + 1) % 10 == 0 || k == hnumber - 1)
        {
            saver.flush();
        }

        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds =
            current_time - sweep_start_time;

        int current_run = k + 1;
        double avg_time_per_run = elapsed_seconds.count() / current_run;
        int runs_left = hnumber - current_run;
        double estimated_remaining_seconds = avg_time_per_run * runs_left;

        int hours = static_cast<int>(estimated_remaining_seconds) / 3600;
        int minutes = (static_cast<int>(estimated_remaining_seconds) % 3600) / 60;
        int seconds = static_cast<int>(estimated_remaining_seconds) % 60;

        std::cout << "Running " << current_run << "/" << hnumber << " (h=" << h
                  << ", h0=" << h0 << ", Omega=" << Omega
                  << ") | ETA: " << std::setfill('0') << std::setw(2) << hours
                  << ":" << std::setfill('0') << std::setw(2) << minutes << ":"
                  << std::setfill('0') << std::setw(2) << seconds << std::endl;
    }

    std::cout << "Parameter sweep complete. Data saved to binary file.\n";

    // Clean up FFTW global resources
    fftw_cleanup_threads();
    fftw_cleanup();

    return 0;
}