#include <iostream>
#include <algorithm>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <numeric>
#include <chrono>
#include <map>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "DomainWallSimulator.h"
#include "FftwHandler.h"
#include "filesaver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void print_usage(const char *prog_name)
{
    std::cerr << "Usage: " << prog_name << " --h <val> --h0 <val> --out_dir <dir>\n\n"
              << "Options:\n"
              << "  --out_dir <dir>      (default: output)\n"
              << "  --wait_time <val>    (default: 1000.0)\n"
              << "  --record_time <val>  (default: 500.0)\n"
              << "  --dt_sample <val>    (default: 0.1)\n"
              << "  --backend <cpu|gpu>  (default: cpu)\n";
}

int main(int argc, char **argv)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    SimulatorConfig config;
    std::string backend_arg = "cpu";
    std::string out_dir = "output";

    config.alpha = 0.27;
    config.Omega = 1.0;
    config.dt = 2.0 * M_PI / 100.0;
    config.min_res = 0.5;
    config.ic_amplitude = 0.1;
    config.ic_seed = 42;

    bool has_h = false, has_h0 = false;
    SIM_REAL wait_time = 1000.0;
    SIM_REAL record_time = 500.0;
    SIM_REAL dt_sample = 0.1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--h" && i + 1 < argc)
        {
            config.h = std::stod(argv[++i]);
            has_h = true;
        }
        else if (arg == "--h0" && i + 1 < argc)
        {
            config.h0 = std::stod(argv[++i]);
            has_h0 = true;
        }
        else if (arg == "--out_dir" && i + 1 < argc)
        {
            out_dir = argv[++i];
        }
        else if (arg == "--wait_time" && i + 1 < argc)
        {
            wait_time = std::stod(argv[++i]);
        }
        else if (arg == "--record_time" && i + 1 < argc)
        {
            record_time = std::stod(argv[++i]);
        }
        else if (arg == "--dt_sample" && i + 1 < argc)
        {
            dt_sample = std::stod(argv[++i]);
        }
        else if (arg == "--backend" && i + 1 < argc)
        {
            backend_arg = argv[++i];
        }
    }

    if (!has_h || !has_h0)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::transform(backend_arg.begin(), backend_arg.end(), backend_arg.begin(), ::tolower);
    config.backend = (backend_arg == "gpu") ? ComputeBackend::GPU : ComputeBackend::CPU;

    std::map<std::string, std::string> exp_params = {
        {"Backend", backend_arg},
        {"h", std::to_string(config.h)},
        {"h0", std::to_string(config.h0)},
        {"Alpha", std::to_string(config.alpha)},
        {"Omega", std::to_string(config.Omega)}};

    std::vector<std::string> headers = {"L", "Avg_Variance", "Peak_k", "Peak_Height"};
    CsvSaver csv_saver(out_dir, "size_scaling_u", exp_params, headers);

    size_t steps_per_sample = std::max((size_t)1, (size_t)std::round(dt_sample / config.dt));

    for (int p = 0; p <= 12; ++p)
    {
        SIM_REAL L = std::pow(2.0, p);
        config.system_size = L;

        std::cout << "\n========================================\n"
                  << "Processing System Size L = " << L << "\n"
                  << "========================================\n";

        auto sim = create_simulator(config);
        sim->reset(config.ic_amplitude, config.ic_seed);

        std::cout << "Fast-forwarding to T=" << wait_time << "...\n";
        while (sim->get_time() < wait_time)
        {
            size_t steps_remaining = std::ceil((wait_time - sim->get_time()) / config.dt);
            sim->run_block(std::min(steps_remaining, (size_t)1000));
        }

        std::cout << "Accumulating data for " << record_time << " time units...\n";

        SIM_REAL period = (config.Omega > 1e-12) ? (4.0 * M_PI / config.Omega) : 1e9;
        size_t num_periods = std::max(static_cast<size_t>(1), static_cast<size_t>(std::round(record_time / period)));
        SIM_REAL exact_record_time = num_periods * period;

        std::cout << "Accumulating data for " << num_periods << " periods (" << exact_record_time << " time units)...\n";

        FftwHandlerR2C fftw_r2c;
        std::vector<SIM_REAL> Sk_avg;
        SIM_REAL accumulated_var = 0.0;
        uint64_t sample_count = 0;

        SIM_REAL end_time = wait_time + exact_record_time;

        while (sim->get_time() < end_time - 1e-9)
        {
            // Prevent overshooting the exact end time boundary
            SIM_REAL time_remaining = end_time - sim->get_time();
            size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
            size_t steps_to_run = std::min(steps_per_sample, steps_remaining);

            if (steps_to_run == 0)
                break;
            sim->run_block(steps_to_run);

            std::vector<SIM_COMPLEX> z;
            sim->get_z_host(z);
            size_t N = z.size();

            AlignedRealVector u_arr(N);
            SIM_REAL u_sum = 0;

            for (size_t i = 0; i < N; ++i)
            {
                u_arr[i] = std::real(z[i]);
                u_sum += u_arr[i];
            }

            SIM_REAL u_mean = u_sum / N;
            SIM_REAL current_var = 0;

            for (size_t i = 0; i < N; ++i)
            {
                current_var += (u_arr[i] - u_mean) * (u_arr[i] - u_mean);
            }
            accumulated_var += (current_var / N);

            AlignedComplexVector u_fk;
            fftw_r2c.do_fft(u_arr, u_fk, false);

            if (Sk_avg.empty())
            {
                Sk_avg.resize(u_fk.size(), 0.0);
            }

            for (size_t i = 0; i < u_fk.size(); ++i)
            {
                Sk_avg[i] += std::norm(u_fk[i]) / (N * N);
            }

            sample_count++;
        }

        SIM_REAL avg_variance = accumulated_var / sample_count;

        SIM_REAL peak_height = 0.0;
        size_t peak_idx = 0;

        // Start from index 1 to ignore the mean (k=0 DC component)
        for (size_t i = 1; i < Sk_avg.size(); ++i)
        {
            SIM_REAL current_S = Sk_avg[i] / sample_count;
            if (current_S > peak_height)
            {
                peak_height = current_S;
                peak_idx = i;
            }
        }

        // Convert index to physical wavenumber k = 2 * pi * n / L
        SIM_REAL peak_k = 2.0 * M_PI * peak_idx / L;

        std::vector<double> row = {
            static_cast<double>(L),
            static_cast<double>(avg_variance),
            static_cast<double>(peak_k),
            static_cast<double>(peak_height)};

        csv_saver.saveRow(DataRow(std::move(row)));

        std::cout << "Results for L=" << L << ":\n"
                  << "  Avg Var:  " << avg_variance << "\n"
                  << "  Peak k:   " << peak_k << " (idx " << peak_idx << ")\n"
                  << "  Peak S(k):" << peak_height << "\n";
    }

    std::cout << "\nScaling analysis complete. Data saved to CSV.\n";
    return 0;
}