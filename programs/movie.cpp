#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <cmath>
#include <string>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "DomainWallSimulator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void print_usage(const char *prog_name)
{
    std::cerr << "Usage: " << prog_name << " [options]\n\n"
              << "Mandatory options:\n"
              << "  --size <val>           System size\n"
              << "  --h <val>              Parameter h\n"
              << "  --h0 <val>             Parameter h0\n\n"
              << "Animation options:\n"
              << "  --wait_time <val>      Time to simulate before recording starts (default: 1000.0)\n"
              << "  --periods <val>        Number of periods to record (default: 5)\n"
              << "  --samples <val>        Samples per period (framerate per cycle) (default: 32)\n\n"
              << "Optional parameters:\n"
              << "  --backend <cpu|gpu>    (default: cpu)\n"
              << "  --alpha <val>          (default: 0.27)\n"
              << "  --Omega <val>          (default: 1.0)\n"
              << "  --dt <val>             (default: 2*PI/100 ≈ 0.06283)\n"
              << "  --ic_amp <val>         (default: 0.1)\n"
              << "  --ic_seed <val>        (default: 42)\n";
}

int main(int argc, char **argv)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    SimulatorConfig config;
    std::string backend_arg = "cpu";

    config.alpha = 0.27;
    config.Omega = 1.0;
    config.dt = 2.0 * M_PI / 100.0;
    config.min_res = 0.1;
    config.ic_amplitude = 0.1;
    config.ic_seed = 42;

    bool has_size = false;
    bool has_h = false;
    bool has_h0 = false;

    SIM_REAL wait_time = 1000.0;
    int num_periods = 5;
    int samples_per_period = 32;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            else if (arg == "--size" && i + 1 < argc)
            {
                config.system_size = static_cast<SIM_REAL>(std::stod(argv[++i]));
                has_size = true;
            }
            else if (arg == "--h" && i + 1 < argc)
            {
                config.h = static_cast<SIM_REAL>(std::stod(argv[++i]));
                has_h = true;
            }
            else if (arg == "--h0" && i + 1 < argc)
            {
                config.h0 = static_cast<SIM_REAL>(std::stod(argv[++i]));
                has_h0 = true;
            }
            else if (arg == "--backend" && i + 1 < argc)
            {
                backend_arg = argv[++i];
            }
            else if (arg == "--alpha" && i + 1 < argc)
            {
                config.alpha = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--Omega" && i + 1 < argc)
            {
                config.Omega = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--dt" && i + 1 < argc)
            {
                config.dt = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--wait_time" && i + 1 < argc)
            {
                wait_time = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--periods" && i + 1 < argc)
            {
                num_periods = std::stoi(argv[++i]);
            }
            else if (arg == "--samples" && i + 1 < argc)
            {
                samples_per_period = std::stoi(argv[++i]);
            }
            else if (arg == "--ic_amp" && i + 1 < argc)
            {
                config.ic_amplitude = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--ic_seed" && i + 1 < argc)
            {
                config.ic_seed = std::stoi(argv[++i]);
            }
            else
            {
                std::cerr << "Unknown or incomplete argument: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return 1;
    }

    if (!has_size || !has_h || !has_h0)
    {
        std::cerr << "Error: Missing mandatory arguments (--size, --h, --h0).\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::transform(backend_arg.begin(), backend_arg.end(), backend_arg.begin(), ::tolower);
    if (backend_arg == "gpu")
        config.backend = ComputeBackend::GPU;
    else if (backend_arg == "cpu")
        config.backend = ComputeBackend::CPU;
    else
    {
        std::cerr << "Error: Invalid backend specified.\n";
        return 1;
    }

    const SIM_REAL period = (config.Omega > 1e-12) ? (4.0 * M_PI / config.Omega) : 1e9;

    std::cout << "Initializing Steady State Animator...\n"
              << "Backend: " << backend_arg << " | Size: " << config.system_size
              << " | h: " << config.h << " | h0: " << config.h0 << "\n"
              << "Wait Time: " << wait_time << " | Periods: " << num_periods
              << " | Samples/Period: " << samples_per_period << "\n"
              << "Effective Period: " << period << "\n"
              << std::endl;

    try
    {
        auto sim = create_simulator(config);
        const size_t max_fast_forward_block = 1000;

        sim->reset(config.ic_amplitude, config.ic_seed);
        std::cout << "System initialized. Fast forwarding to wait time..." << std::endl;

        while (sim->get_time() < wait_time)
        {
            SIM_REAL time_remaining = wait_time - sim->get_time();
            size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
            size_t steps_to_run = std::min(steps_remaining, max_fast_forward_block);
            sim->run_block(steps_to_run);
        }

        std::cout << "Reached wait time (T = " << sim->get_time() << "). Starting recording phase." << std::endl;

        SIM_REAL dt_sample = period / static_cast<SIM_REAL>(samples_per_period);
        SIM_REAL start_record_time = sim->get_time();
        int frame_count = 0;

        for (int p = 0; p < num_periods; ++p)
        {
            for (int s = 0; s < samples_per_period; ++s)
            {
                SIM_REAL target_time = start_record_time + (p * period) + (s * dt_sample);

                while (sim->get_time() < target_time - 1e-9)
                {
                    SIM_REAL time_remaining = target_time - sim->get_time();
                    size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
                    size_t steps_to_run = std::min(steps_remaining, max_fast_forward_block);
                    sim->run_block(steps_to_run);
                }

                std::vector<SIM_COMPLEX> z_host;
                sim->get_z_host(z_host);

                std::stringstream ss_filename;
                ss_filename << "anim_frame_" << std::setw(4) << std::setfill('0') << frame_count << ".bin";

                std::ofstream outfile(ss_filename.str(), std::ios::binary | std::ios::out);
                if (outfile)
                {
                    outfile.write(reinterpret_cast<const char *>(z_host.data()), z_host.size() * sizeof(SIM_COMPLEX));
                    outfile.close();
                }

                if (frame_count % 10 == 0)
                {
                    std::cout << "Saved frame " << frame_count << " (T = " << sim->get_time() << ")" << std::endl;
                }

                frame_count++;
            }
        }

        std::cout << "Animation generation completed successfully. Saved " << frame_count << " frames." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n========================================\n"
                  << "FATAL SIMULATOR EXCEPTION CAUGHT:\n"
                  << e.what()
                  << "\n========================================\n"
                  << std::endl;
        return 1;
    }

    return 0;
}