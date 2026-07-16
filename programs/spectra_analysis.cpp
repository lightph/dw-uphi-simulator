#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
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
    std::cerr << "Usage: " << prog_name << " --size <val> --h <val> --h0 <val> --out_dir <dir>\n\n"
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
    config.min_res = 0.2;
    config.ic_amplitude = 0.1;
    config.ic_seed = 42;

    bool has_size = false, has_h = false, has_h0 = false;
    SIM_REAL wait_time = 1000.0;
    SIM_REAL record_time = 500.0;
    SIM_REAL dt_sample = 0.1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc)
        {
            config.system_size = std::stod(argv[++i]);
            has_size = true;
        }
        else if (arg == "--h" && i + 1 < argc)
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

    if (!has_size || !has_h || !has_h0)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::transform(backend_arg.begin(), backend_arg.end(), backend_arg.begin(), ::tolower);
    config.backend = (backend_arg == "gpu") ? ComputeBackend::GPU : ComputeBackend::CPU;

    std::map<std::string, std::string> exp_params = {
        {"Backend", backend_arg},
        {"SystemSize", std::to_string(config.system_size)},
        {"h", std::to_string(config.h)},
        {"h0", std::to_string(config.h0)},
        {"Alpha", std::to_string(config.alpha)},
        {"Omega", std::to_string(config.Omega)},
        {"dt_sample", std::to_string(dt_sample)}};

    BinarySaver spatial_u_saver(out_dir, "spatial_sf_u", exp_params);
    BinarySaver spatial_v_saver(out_dir, "spatial_sf_v", exp_params);
    BinarySaver temporal_u_saver(out_dir, "temporal_ps_u", exp_params);
    BinarySaver temporal_v_saver(out_dir, "temporal_ps_v", exp_params);
    BinarySaver raw_u_dev_saver(out_dir, "raw_u_deviations", exp_params);

    auto sim = create_simulator(config);
    sim->reset(config.ic_amplitude, config.ic_seed);

    std::cout << "Fast-forwarding to T=" << wait_time << "...\n";
    while (sim->get_time() < wait_time)
    {
        size_t steps_remaining = std::ceil((wait_time - sim->get_time()) / config.dt);
        sim->run_block(std::min(steps_remaining, (size_t)1000));
    }

    SIM_REAL period = (config.Omega > 1e-12) ? (4.0 * M_PI / config.Omega) : 1e9;
    size_t num_periods = std::max(static_cast<size_t>(1), static_cast<size_t>(std::round(record_time / period)));
    SIM_REAL exact_record_time = num_periods * period;

    std::cout << "Accumulating data for " << num_periods << " periods (" << exact_record_time << " time units)...\n";

    FftwHandlerR2C fftw_r2c;
    std::vector<SIM_REAL> Sk_u_avg, Sk_v_avg;
    uint64_t sample_count = 0;

    SIM_REAL end_time = wait_time + exact_record_time;
    size_t steps_per_sample = std::max((size_t)1, (size_t)std::round(dt_sample / config.dt));

    // Enable internal simulator fine tracking to record phidot every sample interval
    sim->set_fine_tracking(true, steps_per_sample);

    while (sim->get_time() < end_time - 1e-9)
    {
        SIM_REAL time_remaining = end_time - sim->get_time();
        size_t steps_remaining = static_cast<size_t>(std::ceil(time_remaining / config.dt));
        size_t steps_to_run = std::min(steps_per_sample, steps_remaining);

        if (steps_to_run == 0)
            break;
        sim->run_block(steps_to_run);

        std::vector<SIM_COMPLEX> z;
        sim->get_z_host(z);
        size_t N = z.size();

        AlignedRealVector u_arr(N), v_arr(N);
        SIM_REAL u_sum = 0;

        for (size_t i = 0; i < N; ++i)
        {
            u_arr[i] = std::real(z[i]);
            v_arr[i] = std::imag(z[i]);
            u_sum += u_arr[i];
        }

        SIM_REAL u_mean = u_sum / N;

        std::vector<double> dev_row;
        dev_row.reserve(N);
        for (size_t i = 0; i < N; ++i)
        {
            dev_row.push_back(static_cast<double>(u_arr[i] - u_mean));
        }
        raw_u_dev_saver.saveRow(DataRow(std::move(dev_row)));

        AlignedComplexVector u_fk, v_fk;
        fftw_r2c.do_fft(u_arr, u_fk, false);
        fftw_r2c.do_fft(v_arr, v_fk, false);

        if (Sk_u_avg.empty())
        {
            Sk_u_avg.resize(u_fk.size(), 0.0);
            Sk_v_avg.resize(v_fk.size(), 0.0);
        }

        for (size_t i = 0; i < u_fk.size(); ++i)
        {
            Sk_u_avg[i] += std::norm(u_fk[i]) / (N * N);
            Sk_v_avg[i] += std::norm(v_fk[i]) / (N * N);
        }
        sample_count++;
    }

    std::vector<double> final_Sk_u, final_Sk_v;
    final_Sk_u.reserve(Sk_u_avg.size());
    final_Sk_v.reserve(Sk_v_avg.size());

    for (size_t i = 0; i < Sk_u_avg.size(); ++i)
    {
        final_Sk_u.push_back(static_cast<double>(Sk_u_avg[i] / sample_count));
        final_Sk_v.push_back(static_cast<double>(Sk_v_avg[i] / sample_count));
    }

    spatial_u_saver.saveRow(DataRow(std::move(final_Sk_u)));
    spatial_v_saver.saveRow(DataRow(std::move(final_Sk_v)));

    // Retrieve analytically exact phidot array natively recorded by the simulator
    std::vector<SIM_REAL> phidot_ts = sim->get_and_clear_fine_phi_dot_history();
    sim->set_fine_tracking(false);

    // Reconstruct the exact u_dot time series using the analytical identity:
    // u_dot = 0.5 * (1 + alpha^2) * h(t) - (1 / alpha) * phi_dot
    std::vector<SIM_REAL> u_dot_ts;
    u_dot_ts.reserve(phidot_ts.size());

    SIM_REAL alpha_sq = config.alpha * config.alpha;
    SIM_REAL prefactor_h = 0.5 * (1.0 + alpha_sq);
    SIM_REAL inv_alpha = 1.0 / config.alpha;
    SIM_REAL dt_actual = steps_per_sample * config.dt;

    for (size_t i = 0; i < phidot_ts.size(); ++i)
    {
        // Calculate the exact simulation time for this recorded sample
        SIM_REAL t = wait_time + (i + 1) * dt_actual;
        SIM_REAL h_t = config.h + config.h0 * std::cos(config.Omega * t);

        SIM_REAL u_dot = prefactor_h * h_t - inv_alpha * phidot_ts[i];
        u_dot_ts.push_back(u_dot);
    }

    AlignedRealVector phidot_aligned(phidot_ts.begin(), phidot_ts.end());
    AlignedRealVector u_dot_aligned(u_dot_ts.begin(), u_dot_ts.end());

    AlignedComplexVector phidot_fw, u_dot_fw;
    fftw_r2c.do_fft(phidot_aligned, phidot_fw, false);
    fftw_r2c.do_fft(u_dot_aligned, u_dot_fw, false);

    size_t Nt = phidot_ts.size();
    std::vector<double> u_mean_ps, v_mean_ps;
    u_mean_ps.reserve(u_dot_fw.size());
    v_mean_ps.reserve(phidot_fw.size());

    for (size_t i = 0; i < phidot_fw.size(); ++i)
    {
        v_mean_ps.push_back(static_cast<double>(std::norm(phidot_fw[i]) / (Nt * Nt)));
        u_mean_ps.push_back(static_cast<double>(std::norm(u_dot_fw[i]) / (Nt * Nt)));
    }

    temporal_u_saver.saveRow(DataRow(std::move(u_mean_ps)));
    temporal_v_saver.saveRow(DataRow(std::move(v_mean_ps)));

    std::cout << "Data saved successfully via BinarySaver. Total samples: " << sample_count << "\n";
    return 0;
}