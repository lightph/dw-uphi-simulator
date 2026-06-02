#include "filesaver.h"
#include "DomainWallSimulator.h"
#include "domain_wall_experiments.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

int main(int argc, char *argv[])
{
    bool request_gpu = true;
    std::vector<std::string> args;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--cpu")
        {
            request_gpu = false;
        }
        else if (arg == "--gpu")
        {
            request_gpu = true;
        }
        else
        {
            args.push_back(arg);
        }
    }

    if (args.size() < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [--cpu | --gpu] Omega h0 system_size hmin hmax hnumber "
                  << "[alpha dt min_resolution max_steps periodsphi periods_in_block]\n";
        return 1;
    }

    const SIM_REAL Omega = static_cast<SIM_REAL>(std::stod(args[0]));
    const SIM_REAL h0 = static_cast<SIM_REAL>(std::stod(args[1]));
    const SIM_REAL system_size = static_cast<SIM_REAL>(std::stod(args[2]));
    const SIM_REAL hmin = static_cast<SIM_REAL>(std::stod(args[3]));
    const SIM_REAL hmax = static_cast<SIM_REAL>(std::stod(args[4]));
    const int hnumber = std::stoi(args[5]);

    const SIM_REAL alpha = (args.size() > 6) ? static_cast<SIM_REAL>(std::stod(args[6])) : 0.27;
    SIM_REAL dt = (args.size() > 7) ? static_cast<SIM_REAL>(std::stod(args[7])) : 0.1;
    const SIM_REAL min_resolution = (args.size() > 8) ? static_cast<SIM_REAL>(std::stod(args[8])) : 0.1;

    const size_t max_steps = (args.size() > 9) ? std::stoull(args[9]) : 500000;
    const size_t periodsphi = (args.size() > 10) ? std::stoull(args[10]) : 100;
    const size_t periods_in_block = (args.size() > 11) ? std::stoull(args[11]) : 1;

    const SIM_REAL h_step = (hnumber > 1) ? (hmax - hmin) / (hnumber - 1) : 0.0;

    std::map<std::string, std::string> sweep_params = {
        {"Backend", request_gpu ? "GPU" : "CPU"},
        {"Type", "Single_Parameter_Sweep"},
        {"Alpha", std::to_string(alpha)},
        {"TimeStep_dt", std::to_string(dt)},
        {"SystemSize", std::to_string(system_size)},
        {"SpatialRes_min", std::to_string(min_resolution)},
        {"MaxSteps", std::to_string(max_steps)},
        {"PeriodsPhi", std::to_string(periodsphi)},
        {"PeriodsInBlock", std::to_string(periods_in_block)},
        {"Omega", std::to_string(Omega)},
        {"h0", std::to_string(h0)},
        {"h_Min", std::to_string(hmin)},
        {"h_Max", std::to_string(hmax)},
        {"h_Points", std::to_string(hnumber)},
    };

    BinarySaver saver("output", "unified_h_sweep", sweep_params);
    auto sweep_start_time = std::chrono::steady_clock::now();

    SimulatorConfig config;
    config.backend = request_gpu ? ComputeBackend::GPU : ComputeBackend::CPU;
    config.alpha = alpha;
    config.h0 = h0;
    config.Omega = Omega;
    config.system_size = system_size;
    config.min_res = min_resolution;
    config.ic_amplitude = 0.02;

    std::cout << "[INFO] Initializing simulator on requested backend: "
              << (request_gpu ? "GPU" : "CPU") << "...\n";

    auto wall = create_simulator(config);

    for (int k = 0; k < hnumber; ++k)
    {
        SIM_REAL h = hmin + (k * h_step);
        int seed = k;

        SIM_REAL T_phys = std::min(static_cast<SIM_REAL>(2.0 * std::acos(-1.0) / Omega),
                                   (h > 1.01) ? static_cast<SIM_REAL>(2.0 * std::acos(-1.0) / (alpha * 0.5 * std::sqrt(h * h - 1.0))) : static_cast<SIM_REAL>(1e9));

        SIM_REAL T_cfl = (min_resolution * min_resolution) / (alpha + 1e-6);
        SIM_REAL target_dt = std::min(T_phys / 50.0, T_cfl * 1.0);

        wall->set_h(h);
        wall->set_dt(std::min(target_dt, dt));
        wall->reset(0.02, seed);
        wall->set_time(0.0);

        SweepMetricsExperiment exp;

        auto run_start_time = std::chrono::steady_clock::now();

        run_simulation(*wall, exp, wall->get_dt(), max_steps, periodsphi, periods_in_block);

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

        if ((k + 1) % 10 == 0 || k == hnumber - 1)
        {
            saver.flush();
        }

        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - sweep_start_time;

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
    return 0;
}