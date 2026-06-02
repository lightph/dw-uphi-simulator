#include "DomainWallSimulator.h"
#include "gpu_domain_wall.cuh"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

bool is_cuda_available()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        return false;
    }
    return deviceCount > 0;
}

struct AccumulateUPowerSpectrum
{
    const cuda_math::COMPLEX *zhat;
    SIM_REAL *power_acc;
    size_t N;

    __device__ void operator()(size_t k) const
    {
        size_t minus_k = (k == 0) ? 0 : N - k;

        cuda_math::COMPLEX zk = zhat[k];
        cuda_math::COMPLEX z_minus_k = zhat[minus_k];

        SIM_REAL Uk_re = static_cast<SIM_REAL>(0.5) * (zk.real() + z_minus_k.real());
        SIM_REAL Uk_im = static_cast<SIM_REAL>(0.5) * (zk.imag() - z_minus_k.imag());

        SIM_REAL power = Uk_re * Uk_re + Uk_im * Uk_im;
        power_acc[k] += power;
    }
};

class SimulatorGPU : public IDomainWallSimulator
{
    gpu_domain_wall<RandomInitialCondition> sim;

    bool fine_tracking_enabled = false;
    size_t fine_sample_rate = 1;
    size_t fine_step_counter = 0;
    std::vector<SIM_REAL> host_fine_phi_dot_history;
    std::vector<SIM_REAL> host_fine_rugosity_history;

    thrust::device_vector<SIM_REAL> d_power_spectrum_acc;
    size_t power_spectrum_count = 0;

public:
    SimulatorGPU(const SimulatorConfig &cfg)
        : sim(cfg.alpha, cfg.h, cfg.h0, cfg.Omega, cfg.dt, cfg.system_size, cfg.min_res,
              RandomInitialCondition(cfg.ic_amplitude, cfg.ic_seed)) {}

    void step() override
    {
        sim.step();

        if (fine_tracking_enabled)
        {
            fine_step_counter++;
            if (fine_step_counter % fine_sample_rate == 0)
            {
                SpatialMetrics m = get_spatial_metrics();
                host_fine_phi_dot_history.push_back(m.mean_phidot);
                host_fine_rugosity_history.push_back(m.rugosity);
            }
        }
    }

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

    void reset(SIM_REAL ic_amplitude, int ic_seed) override
    {
        sim.reset(RandomInitialCondition(ic_amplitude, ic_seed));
    }

    void get_z_host(std::vector<SIM_COMPLEX> &out) const override
    {
        const auto &z = sim.get_z();
        std::vector<cuda_math::COMPLEX> h_z(z.size());
        thrust::copy(z.begin(), z.end(), h_z.begin());

        out.resize(h_z.size());
        std::memcpy(out.data(), h_z.data(), out.size() * sizeof(SIM_COMPLEX));
    }

    void get_zhat_host(std::vector<SIM_COMPLEX> &out) const override
    {
        const auto &zhat = sim.get_zhat();
        std::vector<cuda_math::COMPLEX> h_zhat(zhat.size());
        thrust::copy(zhat.begin(), zhat.end(), h_zhat.begin());

        out.resize(h_zhat.size());
        std::memcpy(out.data(), h_zhat.data(), out.size() * sizeof(SIM_COMPLEX));
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

    SpatialMetrics get_spatial_metrics() const override
    {
        const auto &z_buf = sim.get_z();
        size_t n = z_buf.size();
        if (n == 0)
            return {static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0),
                    static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0)};

        using SumTuple = thrust::tuple<SIM_REAL, SIM_REAL, SIM_REAL>;
        SumTuple init_sum = thrust::make_tuple(static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0), static_cast<SIM_REAL>(0.0));

        SIM_REAL t = sim.get_time();
        SIM_REAL alpha = sim.get_alpha();
        SIM_REAL h_dc = sim.get_h();
        SIM_REAL h_ac = sim.get_h0();
        SIM_REAL Omega = sim.get_Omega();

        SIM_REAL h_t = h_dc + h_ac * std::cos(Omega * t);
        SIM_REAL prefactor = static_cast<SIM_REAL>(0.5) * alpha;

        auto to_sum_tuple = [prefactor, h_t] __device__(const cuda_math::COMPLEX &z) -> SumTuple
        {
            SIM_REAL phi = z.imag();
            SIM_REAL phi_dot = prefactor * (h_t - sin(static_cast<SIM_REAL>(2.0) * phi));
            return thrust::make_tuple(z.real(), phi, phi_dot);
        };

        auto reduce_sum_tuple = [] __device__(const SumTuple &a, const SumTuple &b) -> SumTuple
        {
            return thrust::make_tuple(
                thrust::get<0>(a) + thrust::get<0>(b),
                thrust::get<1>(a) + thrust::get<1>(b),
                thrust::get<2>(a) + thrust::get<2>(b));
        };

        SumTuple total = thrust::transform_reduce(
            thrust::device, z_buf.begin(), z_buf.end(),
            to_sum_tuple, init_sum, reduce_sum_tuple);

        SIM_REAL inv_n = static_cast<SIM_REAL>(1.0) / static_cast<SIM_REAL>(n);
        SIM_REAL mean_u = thrust::get<0>(total) * inv_n;
        SIM_REAL mean_phi = static_cast<SIM_REAL>(-1.0) * (thrust::get<1>(total) * inv_n);
        SIM_REAL mean_phidot = thrust::get<2>(total) * inv_n;

        auto to_variance = [mean_u] __device__(const cuda_math::COMPLEX &z) -> SIM_REAL
        {
            SIM_REAL diff = z.real() - mean_u;
            return diff * diff;
        };

        SIM_REAL sum_sq_diff = thrust::transform_reduce(
            thrust::device, z_buf.begin(), z_buf.end(),
            to_variance, static_cast<SIM_REAL>(0.0), thrust::plus<SIM_REAL>());

        SIM_REAL rugosity = sum_sq_diff * inv_n;

        return {mean_u, mean_phi, mean_phidot, rugosity};
    }
};

std::unique_ptr<IDomainWallSimulator> create_gpu_simulator(const SimulatorConfig &cfg)
{
    return std::make_unique<SimulatorGPU>(cfg);
}