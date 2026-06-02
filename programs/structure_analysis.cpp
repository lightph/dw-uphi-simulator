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
#include <numeric>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "DomainWallSimulator.h"
#include "FftwHandler.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void print_usage(const char *prog_name)
{
    std::cerr << "Usage: " << prog_name << " [options]\n\n"
              << "Mandatory options:\n"
              << "  --size <val>         System size\n"
              << "  --h <val>            Parameter h\n"
              << "  --h0 <val>           Parameter h0\n\n"
              << "Optional parameters:\n"
              << "  --backend <cpu|gpu>  (default: cpu)\n"
              << "  --alpha <val>        (default: 0.27)\n"
              << "  --Omega <val>        (default: 1.0)\n"
              << "  --dt <val>           (default: 2*PI/100 ≈ 0.06283)\n"
              << "  --min_res <val>      (default: 0.5)\n"
              << "  --ic_amp <val>       (default: 0.1)\n"
              << "  --ic_seed <val>      (default: 42)\n"
              << "  --start_p <val>      (default: 2)\n"
              << "  --end_p <val>        (default: 20)\n"
              << "  --runs <val>         (default: 200)\n";
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
    config.min_res = 0.5;
    config.ic_amplitude = 0.1;
    config.ic_seed = 42;

    bool has_size = false;
    bool has_h = false;
    bool has_h0 = false;

    int start_p = 2;
    int end_p = 20;
    int num_realizations = 200;

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
            else if (arg == "--min_res" && i + 1 < argc)
            {
                config.min_res = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--ic_amp" && i + 1 < argc)
            {
                config.ic_amplitude = static_cast<SIM_REAL>(std::stod(argv[++i]));
            }
            else if (arg == "--ic_seed" && i + 1 < argc)
            {
                config.ic_seed = std::stoi(argv[++i]);
            }
            else if (arg == "--start_p" && i + 1 < argc)
            {
                start_p = std::stoi(argv[++i]);
            }
            else if (arg == "--end_p" && i + 1 < argc)
            {
                end_p = std::stoi(argv[++i]);
            }
            else if (arg == "--runs" && i + 1 < argc)
            {
                num_realizations = std::stoi(argv[++i]);
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
    {
        config.backend = ComputeBackend::GPU;
    }
    else if (backend_arg == "cpu")
    {
        config.backend = ComputeBackend::CPU;
    }
    else
    {
        std::cerr << "Error: Invalid backend specified.\n";
        return 1;
    }

    const SIM_REAL period = (config.Omega > 1e-12) ? (4.0 * M_PI / config.Omega) : 1e9;
    std::cout << "Initializing Simulator Ensemble...\n"
              << "Backend: " << backend_arg << " | Size: " << config.system_size << " | h: " << config.h << " | h0: " << config.h0 << "\n"
              << "Realizations: " << num_realizations << " | Power of 2 Range: [" << start_p << ", " << end_p << "]\n"
              << " | Effective Period: " << period << "\n"
              << std::endl;

    try
    {
        auto sim = create_simulator(config);
        FftwHandlerR2C fftw_r2c;

        const size_t steps_between_samples = 10;
        const size_t max_fast_forward_block = 1000;
        const int num_p = end_p - start_p + 1;

        std::vector<std::vector<SIM_REAL>> ensemble_sf(num_p);
        std::vector<std::vector<SIM_REAL>> ensemble_phidot_ps(num_p);
        std::vector<SIM_REAL> ensemble_avg_phidot(num_p, 0.0);
        std::vector<SIM_REAL> ensemble_avg_rug(num_p, 0.0);

        std::string base_name = "_L" + std::to_string(config.system_size) + "_h" + std::to_string(config.h) + "_h0" + std::to_string(config.h0) + ".bin";
        std::ofstream scalar_out;

        for (int r = 0; r < num_realizations; ++r)
        {
            int current_seed = config.ic_seed + r;
            std::cout << "Starting realization " << r + 1 << "/" << num_realizations << " (Seed: " << current_seed << ")" << std::endl;

            sim->reset(config.ic_amplitude, current_seed);

            if (r == num_realizations - 1)
            {
                scalar_out.open("ensemble_scalars_L" + std::to_string(config.system_size) + "_h" + std::to_string(config.h) + ".txt");
                if (scalar_out)
                    scalar_out << "pow2\tavg_phidot\tavg_rugosity\n";
            }

            for (int p = start_p; p <= end_p; ++p)
            {
                int p_idx = p - start_p;
                SIM_REAL target_time = std::pow(2.0, static_cast<SIM_REAL>(p));

                if (sim->get_time() < target_time)
                {
                    while (sim->get_time() < target_time)
                    {
                        SIM_REAL time_remaining = target_time - sim->get_time();
                        size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
                        size_t steps_to_run = std::min(steps_remaining, max_fast_forward_block);
                        sim->run_block(steps_to_run);
                    }
                }

                sim->set_fine_tracking(true, 1);
                sim->reset_spectrum_accumulator();

                SIM_REAL end_averaging_time = sim->get_time() + period * 30;

                while (sim->get_time() < end_averaging_time - 1e-9)
                {
                    SIM_REAL time_remaining = end_averaging_time - sim->get_time();
                    size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
                    size_t steps_to_run = std::min(steps_between_samples, steps_remaining);

                    if (steps_to_run == 0)
                        break;

                    sim->run_block(steps_to_run);
                    sim->accumulate_u_power_spectrum();
                }

                std::vector<SIM_REAL> sf_averaged = sim->get_averaged_u_power_spectrum();
                std::vector<SIM_REAL> phidot_history = sim->get_and_clear_fine_phi_dot_history();
                std::vector<SIM_REAL> rugosity_history = sim->get_and_clear_fine_rugosity_history();

                sim->set_fine_tracking(false);

                if (ensemble_sf[p_idx].empty())
                {
                    ensemble_sf[p_idx] = sf_averaged;
                }
                else
                {
                    for (size_t i = 0; i < sf_averaged.size(); ++i)
                        ensemble_sf[p_idx][i] += sf_averaged[i];
                }

                if (!phidot_history.empty())
                {
                    SIM_REAL sum_phidot = std::accumulate(phidot_history.begin(), phidot_history.end(), static_cast<SIM_REAL>(0.0));
                    SIM_REAL avg_phidot = sum_phidot / static_cast<SIM_REAL>(phidot_history.size());
                    ensemble_avg_phidot[p_idx] += avg_phidot;
                }

                if (!rugosity_history.empty())
                {
                    SIM_REAL sum_rug = std::accumulate(rugosity_history.begin(), rugosity_history.end(), static_cast<SIM_REAL>(0.0));
                    SIM_REAL avg_rug = sum_rug / static_cast<SIM_REAL>(rugosity_history.size());
                    ensemble_avg_rug[p_idx] += avg_rug;
                }

                if (!phidot_history.empty())
                {
                    AlignedRealVector in_fftw(phidot_history.begin(), phidot_history.end());
                    AlignedComplexVector out_fftw;
                    fftw_r2c.do_fft(in_fftw, out_fftw, false);

                    size_t n_ts = phidot_history.size();
                    std::vector<SIM_REAL> ps(out_fftw.size());
                    for (size_t i = 0; i < out_fftw.size(); ++i)
                    {
                        SIM_REAL re = std::real(out_fftw[i]);
                        SIM_REAL im = std::imag(out_fftw[i]);
                        ps[i] = (re * re + im * im) / static_cast<SIM_REAL>(n_ts * n_ts);
                    }

                    if (ensemble_phidot_ps[p_idx].empty())
                    {
                        ensemble_phidot_ps[p_idx] = ps;
                    }
                    else
                    {
                        for (size_t i = 0; i < ps.size(); ++i)
                            ensemble_phidot_ps[p_idx][i] += ps[i];
                    }
                }

                if (r == num_realizations - 1)
                {
                    ensemble_avg_phidot[p_idx] /= num_realizations;
                    ensemble_avg_rug[p_idx] /= num_realizations;

                    if (scalar_out)
                    {
                        scalar_out << p << "\t" << ensemble_avg_phidot[p_idx] << "\t" << ensemble_avg_rug[p_idx] << "\n";
                        scalar_out.flush();
                    }

                    if (!ensemble_sf[p_idx].empty())
                    {
                        for (auto &val : ensemble_sf[p_idx])
                            val /= num_realizations;
                        std::stringstream ss_sf;
                        ss_sf << "ensemble_sf" << "_T2pow" << p << base_name;
                        std::ofstream out_sf(ss_sf.str(), std::ios::binary | std::ios::out);
                        if (out_sf)
                            out_sf.write(reinterpret_cast<const char *>(ensemble_sf[p_idx].data()), ensemble_sf[p_idx].size() * sizeof(SIM_REAL));
                    }

                    if (!ensemble_phidot_ps[p_idx].empty())
                    {
                        for (auto &val : ensemble_phidot_ps[p_idx])
                            val /= num_realizations;
                        std::stringstream ss_ps;
                        ss_ps << "ensemble_phidot_ps" << "_T2pow" << p << base_name;
                        std::ofstream out_ps(ss_ps.str(), std::ios::binary | std::ios::out);
                        if (out_ps)
                            out_ps.write(reinterpret_cast<const char *>(ensemble_phidot_ps[p_idx].data()), ensemble_phidot_ps[p_idx].size() * sizeof(SIM_REAL));
                    }

                    std::cout << "  -> Saved ensemble averages to disk for T=2^" << p << std::endl;
                }
            }

            if (r == num_realizations - 1 && scalar_out)
            {
                scalar_out.close();
            }

            if (r >= num_realizations - 5)
            {
                SIM_REAL current_time = sim->get_time();
                SIM_REAL next_period_start = std::ceil(current_time / period) * period;

                if (next_period_start - current_time < 1e-9)
                {
                    next_period_start += period;
                }

                while (sim->get_time() < next_period_start - 1e-9)
                {
                    SIM_REAL time_remaining = next_period_start - sim->get_time();
                    size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
                    size_t steps_to_run = std::min(steps_remaining, max_fast_forward_block);
                    sim->run_block(steps_to_run);
                }

                SIM_REAL eighth_period = period / 8.0;
                for (int k = 0; k < 8; ++k)
                {
                    std::vector<SIM_COMPLEX> z_host;
                    sim->get_z_host(z_host);

                    std::stringstream ss_z;
                    ss_z << "z_snapshot_L" << config.system_size << "_h" << config.h
                         << "_R" << r << "_snap" << k << ".bin";

                    std::ofstream outfile(ss_z.str(), std::ios::binary | std::ios::out);
                    if (outfile)
                    {
                        outfile.write(reinterpret_cast<const char *>(z_host.data()), z_host.size() * sizeof(SIM_COMPLEX));
                        outfile.close();
                    }

                    if (k < 7)
                    {
                        SIM_REAL next_target = next_period_start + (k + 1) * eighth_period;

                        while (sim->get_time() < next_target - 1e-9)
                        {
                            SIM_REAL time_remaining = next_target - sim->get_time();
                            size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
                            size_t steps_to_run = std::min(steps_remaining, max_fast_forward_block);
                            sim->run_block(steps_to_run);
                        }
                    }
                }
            }
        }

        std::cout << "Simulation ensemble completed successfully." << std::endl;
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