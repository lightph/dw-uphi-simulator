#pragma once

#include "DomainWallSimulator.h"
#include <cmath>
#include <algorithm>
#include <limits>

class DomainWallExperiment
{
public:
    virtual ~DomainWallExperiment() = default;

    virtual void on_start(IDomainWallSimulator &wall) {}
    virtual void on_transient_block(IDomainWallSimulator &wall, size_t block_count, SIM_REAL current_time) {}
    virtual void on_measurement_block(IDomainWallSimulator &wall, size_t block_count, SIM_REAL current_time) {}
    virtual void on_fine_measurement_step(IDomainWallSimulator &wall, SIM_REAL current_time) {}
    virtual void on_finish(IDomainWallSimulator &wall) {}

    SIM_REAL final_time_avg_rugosity = (SIM_REAL)0.0;
    SIM_REAL final_speed_mean_u = (SIM_REAL)0.0;
    SIM_REAL final_speed_mean_phi = (SIM_REAL)0.0;
};

class SweepMetricsExperiment : public DomainWallExperiment
{
public:
    SIM_REAL accumulated_rugosity = (SIM_REAL)0.0;
    int measurement_blocks = 0;

    void on_start(IDomainWallSimulator &wall) override
    {
        accumulated_rugosity = (SIM_REAL)0.0;
        measurement_blocks = 0;
    }

    void on_measurement_block(IDomainWallSimulator &wall,
                              size_t block_count, SIM_REAL current_time) override
    {
        SpatialMetrics current = wall.get_spatial_metrics();
        accumulated_rugosity += current.rugosity;
        measurement_blocks++;
    }

    void on_finish(IDomainWallSimulator &wall) override
    {
        if (measurement_blocks > 0)
        {
            final_time_avg_rugosity = accumulated_rugosity / measurement_blocks;
        }
    }
};

class SpectralVelocityExperiment : public SweepMetricsExperiment
{
public:
    std::vector<SIM_REAL> fine_time_history;
    std::vector<SIM_REAL> fine_phi_dot_history;

    std::vector<SIM_REAL> final_u_power_spectrum;

    size_t sample_every_n_steps;

    SpectralVelocityExperiment(size_t sample_rate = 1)
        : sample_every_n_steps(sample_rate) {}

    void on_start(IDomainWallSimulator &wall) override
    {
        SweepMetricsExperiment::on_start(wall);

        fine_time_history.clear();
        fine_phi_dot_history.clear();
        final_u_power_spectrum.clear();

        wall.set_fine_tracking(true, sample_every_n_steps);
        wall.reset_spectrum_accumulator();
    }

    void on_measurement_block(IDomainWallSimulator &wall,
                              size_t block_count, SIM_REAL current_time) override
    {
        SweepMetricsExperiment::on_measurement_block(wall, block_count, current_time);

        wall.accumulate_u_power_spectrum();

        std::vector<SIM_REAL> block_phi_dot = wall.get_and_clear_fine_phi_dot_history();

        if (!block_phi_dot.empty())
        {
            SIM_REAL dt = wall.get_dt();
            SIM_REAL sample_dt = dt * static_cast<SIM_REAL>(sample_every_n_steps);

            SIM_REAL start_t = current_time - (static_cast<SIM_REAL>(block_phi_dot.size() - 1) * sample_dt);

            for (size_t i = 0; i < block_phi_dot.size(); ++i)
            {
                fine_time_history.push_back(start_t + i * sample_dt);
                fine_phi_dot_history.push_back(block_phi_dot[i]);
            }
        }
    }

    void on_finish(IDomainWallSimulator &wall) override
    {
        SweepMetricsExperiment::on_finish(wall);

        final_u_power_spectrum = wall.get_averaged_u_power_spectrum();

        wall.set_fine_tracking(false);
    }
};

inline void run_simulation(IDomainWallSimulator &wall,
                           DomainWallExperiment &experiment,
                           SIM_REAL dt, size_t max_steps, size_t periodsphi,
                           size_t periods_in_block, SIM_REAL fourier_dt = (SIM_REAL)0.0)
{
    SIM_REAL Omega = wall.get_Omega();

    size_t steps_per_block = 100 * periods_in_block;
    if (Omega > (SIM_REAL)1e-12)
    {
        SIM_REAL T = (SIM_REAL)2.0 * std::acos((SIM_REAL)-1.0) / Omega;
        steps_per_block = static_cast<size_t>(std::round(T * (SIM_REAL)periods_in_block / dt));
    }

    size_t steps_in_fourier_dt = steps_per_block;
    size_t chunks_per_period = 1;

    if (fourier_dt > dt)
    {
        steps_in_fourier_dt = static_cast<size_t>(std::round(fourier_dt / dt));
        chunks_per_period = std::max<size_t>((size_t)1, steps_per_block / steps_in_fourier_dt);
        steps_per_block = chunks_per_period * steps_in_fourier_dt;
    }

    SIM_REAL target_distance = (SIM_REAL)2.0 * std::acos((SIM_REAL)-1.0) * (SIM_REAL)periodsphi;

    experiment.on_start(wall);
    SpatialMetrics initial_metrics = wall.get_spatial_metrics();
    SIM_REAL initial_phi = initial_metrics.mean_phi;

    bool equilibrated = false;
    SpatialMetrics middle_metrics;
    SIM_REAL middle_time = (SIM_REAL)0.0;

    SpatialMetrics fallback_metrics;
    SIM_REAL fallback_time = (SIM_REAL)0.0;
    bool fallback_recorded = false;

    size_t n_step = 0;
    SpatialMetrics current_metrics = initial_metrics;

    while (std::abs(current_metrics.mean_phi - initial_phi) < target_distance && n_step < max_steps)
    {
        SIM_REAL distance_traveled = std::abs(current_metrics.mean_phi - initial_phi);

        if (distance_traveled > target_distance * (SIM_REAL)0.5 || n_step > max_steps / 2)
        {
            if (chunks_per_period > 1)
            {
                for (size_t i = 0; i < chunks_per_period; ++i)
                {
                    wall.run_block(steps_in_fourier_dt);
                    n_step += steps_in_fourier_dt;
                    experiment.on_fine_measurement_step(wall, wall.get_time());
                }
            }
            else
            {
                wall.run_block(steps_per_block);
                n_step += steps_per_block;
            }
            experiment.on_measurement_block(wall, n_step / steps_per_block, wall.get_time());
        }
        else
        {
            wall.run_block(steps_per_block);
            n_step += steps_per_block;
            experiment.on_transient_block(wall, n_step / steps_per_block, wall.get_time());
        }

        current_metrics = wall.get_spatial_metrics();
        distance_traveled = std::abs(current_metrics.mean_phi - initial_phi);

        if (distance_traveled > target_distance * (SIM_REAL)0.9 && !equilibrated)
        {
            equilibrated = true;
            middle_metrics = current_metrics;
            middle_time = wall.get_time();
        }

        if (n_step >= static_cast<size_t>((SIM_REAL)max_steps * (SIM_REAL)0.8) && !fallback_recorded)
        {
            fallback_metrics = current_metrics;
            fallback_time = wall.get_time();
            fallback_recorded = true;
        }
    }

    SIM_REAL delta_phi = (SIM_REAL)0.0;
    SIM_REAL delta_u = (SIM_REAL)0.0;
    SIM_REAL elapsed_time = (SIM_REAL)0.0;

    if (equilibrated)
    {
        delta_phi = current_metrics.mean_phi - middle_metrics.mean_phi;
        delta_u = current_metrics.mean_u - middle_metrics.mean_u;
        elapsed_time = wall.get_time() - middle_time;
    }
    else if (fallback_recorded)
    {
        delta_phi = current_metrics.mean_phi - fallback_metrics.mean_phi;
        delta_u = current_metrics.mean_u - fallback_metrics.mean_u;
        elapsed_time = wall.get_time() - fallback_time;
    }

    if (elapsed_time > (SIM_REAL)0.0)
    {
        experiment.final_speed_mean_phi = delta_phi / elapsed_time;
        experiment.final_speed_mean_u = delta_u / elapsed_time;
    }
    else
    {
        experiment.final_speed_mean_phi = std::numeric_limits<SIM_REAL>::quiet_NaN();
        experiment.final_speed_mean_u = std::numeric_limits<SIM_REAL>::quiet_NaN();
    }

    experiment.on_finish(wall);
}