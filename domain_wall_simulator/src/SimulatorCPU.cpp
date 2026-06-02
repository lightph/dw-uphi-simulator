#include "DomainWallSimulator.h"
#include "cpu_domain_wall.h"
#include <vector>
#include <cmath>
#include <algorithm>

class SimulatorCPU : public IDomainWallSimulator
{
    cpu_domain_wall<RandomInitialCondition> sim;

    bool fine_tracking_enabled = false;
    size_t fine_sample_rate = 1;
    size_t fine_step_counter = 0;
    std::vector<SIM_REAL> host_fine_phi_dot_history;
    std::vector<SIM_REAL> host_fine_rugosity_history;

    std::vector<SIM_REAL> u_power_spectrum_acc;
    size_t power_spectrum_count = 0;

public:
    SimulatorCPU(const SimulatorConfig &cfg)
        : sim(cfg.alpha, cfg.h, cfg.h0, cfg.Omega, cfg.dt, cfg.system_size, cfg.min_res,
              RandomInitialCondition(cfg.ic_amplitude, cfg.ic_seed)) {}

    void step() override { sim.step(); }

    void run_block(size_t steps) override
    {
        if (!fine_tracking_enabled)
        {
            sim.run_block(steps);
            return;
        }

        size_t steps_taken = 0;
        while (steps_taken < steps)
        {
            size_t next_sample_dist = fine_sample_rate - (fine_step_counter % fine_sample_rate);
            size_t chunk = std::min(next_sample_dist, steps - steps_taken);

            if (chunk > 0)
            {
                if (chunk > 1)
                {
                    sim.run_block(chunk);
                }
                else
                {
                    sim.step();
                }
                steps_taken += chunk;
                fine_step_counter += chunk;
            }

            if (fine_step_counter % fine_sample_rate == 0)
            {
                SpatialMetrics m = get_spatial_metrics();
                host_fine_phi_dot_history.push_back(m.mean_phidot);
                host_fine_rugosity_history.push_back(m.rugosity);
            }
        }
    }

    void create_step_graph(size_t steps) override { sim.create_step_graph(steps); }

    void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) override
    {
        fine_tracking_enabled = enable;
        fine_sample_rate = sample_every_n_steps;
        if (!enable)
        {
            host_fine_phi_dot_history.clear();
            host_fine_rugosity_history.clear();
            fine_step_counter = 0;
        }
    }

    std::vector<SIM_REAL> get_and_clear_fine_phi_dot_history() override
    {
        std::vector<SIM_REAL> result = std::move(host_fine_phi_dot_history);
        host_fine_phi_dot_history.clear();
        return result;
    }

    std::vector<SIM_REAL> get_and_clear_fine_rugosity_history() override
    {
        std::vector<SIM_REAL> result = std::move(host_fine_rugosity_history);
        host_fine_rugosity_history.clear();
        return result;
    }

    void accumulate_u_power_spectrum() override
    {
        const auto &zhat = sim.get_zhat();
        size_t N = zhat.size();

        if (u_power_spectrum_acc.size() != N)
        {
            u_power_spectrum_acc.assign(N, static_cast<SIM_REAL>(0.0));
            power_spectrum_count = 0;
        }

        const FFTW_COMPLEX_STD *data = zhat.data();

#pragma omp parallel for
        for (size_t k = 0; k < N; ++k)
        {
            size_t minus_k = (k == 0) ? 0 : N - k;

            SIM_REAL zk_re = data[k].real();
            SIM_REAL zk_im = data[k].imag();
            SIM_REAL z_minus_k_re = data[minus_k].real();
            SIM_REAL z_minus_k_im = data[minus_k].imag();

            SIM_REAL Uk_re = static_cast<SIM_REAL>(0.5) * (zk_re + z_minus_k_re);
            SIM_REAL Uk_im = static_cast<SIM_REAL>(0.5) * (zk_im - z_minus_k_im);

            SIM_REAL power = Uk_re * Uk_re + Uk_im * Uk_im;
            u_power_spectrum_acc[k] += power;
        }

        power_spectrum_count++;
    }

    std::vector<SIM_REAL> get_averaged_u_power_spectrum() const override
    {
        std::vector<SIM_REAL> avg(u_power_spectrum_acc.size(), static_cast<SIM_REAL>(0.0));

        if (power_spectrum_count > 0)
        {
            SIM_REAL inv_count = static_cast<SIM_REAL>(1.0) / static_cast<SIM_REAL>(power_spectrum_count);
            for (size_t i = 0; i < avg.size(); ++i)
            {
                avg[i] = u_power_spectrum_acc[i] * inv_count;
            }
        }
        return avg;
    }

    void reset_spectrum_accumulator() override
    {
        std::fill(u_power_spectrum_acc.begin(), u_power_spectrum_acc.end(), static_cast<SIM_REAL>(0.0));
        power_spectrum_count = 0;
    }

    void get_z_host(std::vector<SIM_COMPLEX> &out) const override
    {
        const auto &z = sim.get_z();
        out.assign(z.begin(), z.end());
    }

    void get_zhat_host(std::vector<SIM_COMPLEX> &out) const override
    {
        const auto &zhat = sim.get_zhat();
        out.assign(zhat.begin(), zhat.end());
    }

    void set_h(SIM_REAL h) override { sim.set_h(h); }
    void set_alpha(SIM_REAL alpha) override { sim.set_alpha(alpha); }
    void set_h0(SIM_REAL h0) override { sim.set_h0(h0); }
    void set_Omega(SIM_REAL Omega) override { sim.set_Omega(Omega); }
    void set_dt(SIM_REAL dt) override { sim.set_dt(dt); }

    SIM_REAL get_time() const override { return sim.get_time(); }
    void set_time(SIM_REAL t) override { sim.set_time(t); }
    SIM_REAL get_dt() const override { return sim.get_dt(); }
    SIM_REAL get_Omega() const override { return sim.get_Omega(); }

    SIM_REAL get_h() const override { return static_cast<SIM_REAL>(sim.get_h()); }
    SIM_REAL get_h0() const override { return static_cast<SIM_REAL>(sim.get_h0()); }
    SIM_REAL get_alpha() const override { return static_cast<SIM_REAL>(sim.get_alpha()); }
    SIM_REAL get_system_size() const override { return static_cast<SIM_REAL>(sim.get_system_size()); }
    size_t get_n_samples() const override { return sim.get_n_samples(); }

    void reset(SIM_REAL ic_amplitude, int ic_seed) override
    {
        sim.reset(RandomInitialCondition(ic_amplitude, ic_seed));
    }

    SpatialMetrics get_spatial_metrics() const override
    {
        const auto &z_buf = sim.get_z();
        size_t n = z_buf.size();
        if (n == 0)
            return {static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0),
                    static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0)};

        const FFTW_COMPLEX_STD *data = z_buf.data();

        SIM_REAL sum_u = static_cast<SIM_REAL>(0.0);
        SIM_REAL sum_phi = static_cast<SIM_REAL>(0.0);
        SIM_REAL sum_phi_dot = static_cast<SIM_REAL>(0.0);

        SIM_REAL t = sim.get_time();
        SIM_REAL alpha = sim.get_alpha();
        SIM_REAL h_dc = sim.get_h();
        SIM_REAL h_ac = sim.get_h0();
        SIM_REAL Omega = sim.get_Omega();

        SIM_REAL h_t = h_dc + h_ac * std::cos(Omega * t);
        SIM_REAL prefactor = static_cast<SIM_REAL>(0.5) * alpha;

#pragma omp simd reduction(+ : sum_u, sum_phi, sum_phi_dot)
        for (size_t i = 0; i < n; ++i)
        {
            SIM_REAL phi = data[i].imag();
            sum_u += data[i].real();
            sum_phi += phi;

            sum_phi_dot += prefactor * (h_t - std::sin(static_cast<SIM_REAL>(2.0) * phi));
        }

        SIM_REAL inv_n = static_cast<SIM_REAL>(1.0) / static_cast<SIM_REAL>(n);
        SIM_REAL mean_u = sum_u * inv_n;
        SIM_REAL mean_phi = static_cast<SIM_REAL>(-1.0) * (sum_phi * inv_n);
        SIM_REAL mean_phi_dot = sum_phi_dot * inv_n;

        SIM_REAL sum_sq_diff = static_cast<SIM_REAL>(0.0);
#pragma omp simd reduction(+ : sum_sq_diff)
        for (size_t i = 0; i < n; ++i)
        {
            SIM_REAL diff = data[i].real() - mean_u;
            sum_sq_diff += diff * diff;
        }

        SIM_REAL rugosity = sum_sq_diff * inv_n;

        return {mean_u, mean_phi, mean_phi_dot, rugosity};
    }
};

std::unique_ptr<IDomainWallSimulator> create_cpu_simulator(const SimulatorConfig &cfg)
{
    return std::make_unique<SimulatorCPU>(cfg);
}