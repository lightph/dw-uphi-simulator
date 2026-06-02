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

#include <locale>

int main(int argc, char *argv[])
{
    bool request_gpu = true;
    std::vector<std::string> args;
    std::locale::global(std::locale("C"));

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--cpu")
            request_gpu = false;
        else if (arg == "--gpu")
            request_gpu = true;
        else
            args.push_back(arg);
    }

    if (args.size() < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [--cpu | --gpu] Omega h0 h system_size "
                  << "[alpha dt min_res max_steps periodsphi sample_rate]\n";
        return 1;
    }

    // Parameters
    const SIM_REAL Omega = static_cast<SIM_REAL>(std::stod(args[0]));
    const SIM_REAL h0 = static_cast<SIM_REAL>(std::stod(args[1]));
    const SIM_REAL h = static_cast<SIM_REAL>(std::stod(args[2]));
    const SIM_REAL system_size = static_cast<SIM_REAL>(std::stod(args[3]));

    const SIM_REAL alpha = (args.size() > 4) ? static_cast<SIM_REAL>(std::stod(args[4])) : 0.27;
    const SIM_REAL dt = (args.size() > 5) ? static_cast<SIM_REAL>(std::stod(args[5])) : 0.1;
    const SIM_REAL min_res = (args.size() > 6) ? static_cast<SIM_REAL>(std::stod(args[6])) : 0.1;
    const size_t max_steps = (args.size() > 7) ? std::stoull(args[7]) : 1000000;
    const size_t periodsphi = (args.size() > 8) ? std::stoull(args[8]) : 100;
    const size_t sample_rate = (args.size() > 9) ? std::stoull(args[9]) : 1; // Sample every N steps

    std::map<std::string, std::string> params = {
        {"Backend", request_gpu ? "GPU" : "CPU"},
        {"h", std::to_string(h)},
        {"h0", std::to_string(h0)},
        {"Omega", std::to_string(Omega)},
        {"Alpha", std::to_string(alpha)},
        {"SystemSize", std::to_string(system_size)},
        {"SampleRate", std::to_string(sample_rate)}};

    // Initialize Saver
    BinarySaver summary_saver("output", "single_run_summary", params);
    BinarySaver phidot_saver("output", "single_run_phidot_series", params);
    BinarySaver spectrum_saver("output", "single_run_u_spectrum", params);

    // Setup Simulator
    SimulatorConfig config;
    config.backend = request_gpu ? ComputeBackend::GPU : ComputeBackend::CPU;
    config.alpha = alpha;
    config.h = h;
    config.h0 = h0;
    config.Omega = Omega;
    config.dt = dt;
    config.system_size = system_size;
    config.min_res = min_res;
    config.ic_amplitude = 0.02;
    config.ic_seed = 42;

    auto wall = create_simulator(config);

    // Use the new experiment class
    SpectralVelocityExperiment exp(sample_rate);

    std::cout << "[INFO] Starting single run (h=" << h << ") on " << (request_gpu ? "GPU" : "CPU") << "...\n";

    auto start_t = std::chrono::steady_clock::now();

    // Execute
    run_simulation(*wall, exp, dt, max_steps, periodsphi, 1);

    auto end_t = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end_t - start_t;

    // --- SAVE SUMMARY METRICS ---
    std::vector<double> summary = {
        static_cast<double>(alpha),
        static_cast<double>(h),
        static_cast<double>(h0),
        static_cast<double>(Omega),
        static_cast<double>(exp.final_time_avg_rugosity),
        static_cast<double>(exp.final_speed_mean_u),
        static_cast<double>(exp.final_speed_mean_phi),
        duration.count()};
    summary_saver.saveRow(DataRow(std::move(summary)));

    // --- SAVE PHI_DOT TIME SERIES ---
    // We save this as one giant row of doubles
    std::vector<double> phidot_data;
    phidot_data.reserve(exp.fine_phi_dot_history.size());
    for (auto val : exp.fine_phi_dot_history)
        phidot_data.push_back(static_cast<double>(val));
    phidot_saver.saveRow(DataRow(std::move(phidot_data)));

    // --- SAVE U POWER SPECTRUM ---
    std::vector<double> spectrum_data;
    spectrum_data.reserve(exp.final_u_power_spectrum.size());
    for (auto val : exp.final_u_power_spectrum)
        spectrum_data.push_back(static_cast<double>(val));
    spectrum_saver.saveRow(DataRow(std::move(spectrum_data)));

    std::cout << "[SUCCESS] Run complete.\n"
              << " - Rugosity: " << exp.final_time_avg_rugosity << "\n"
              << " - V_phi:    " << exp.final_speed_mean_phi << "\n"
              << " - Samples:  " << exp.fine_phi_dot_history.size() << "\n"
              << " - Time:     " << duration.count() << "s\n";

    return 0;
}