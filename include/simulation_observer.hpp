#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cuda_memory.cuh"
#include "domain_wall.cuh"

namespace dw {

template <typename Backend>
class SimulationObserver {
   public:
    using Real = typename Backend::PrecisionType::Real;

    SimulationObserver(DomainWall<Backend>& sim, const std::string& output_file)
        : sim_(sim), out_(output_file), current_time_(0.0) {
        out_ << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ << "batch_idx time avg_u_dot avg_phi_dot var_u\n";
    }

    ~SimulationObserver() {
        if (out_.is_open()) out_.close();
    }

    void run(int total_steps, int output_interval) {
        Backend::synchronize();

        std::size_t batches = sim_.get_batch();

        // Allocate device buffers to hold interval history without stalling the CPU
        CudaVector<Real> d_mean_u(batches * output_interval);
        CudaVector<Real> d_mean_u2(batches * output_interval);
        CudaVector<Real> d_mean_sin(batches * output_interval);

        std::vector<Real> h_val_history;
        h_val_history.reserve(output_interval);

        for (int step = 1; step <= total_steps; ++step) {
            int local_step = (step - 1) % output_interval;

            if (local_step == 0) {
                // Pre-zero the arrays asynchronously at the start of every interval
                Backend::fill_zero(d_mean_u.data(), d_mean_u.size());
                Backend::fill_zero(d_mean_u2.data(), d_mean_u2.size());
                Backend::fill_zero(d_mean_sin.data(), d_mean_sin.size());
                h_val_history.clear();
            }

            // Calculate field at t before step increments it
            Real h_val = sim_.get_h0() + sim_.get_ha() * std::cos(sim_.get_omega() * current_time_);
            h_val_history.push_back(h_val);

            // GPU processes the entire batched state asynchronously
            sim_.step(current_time_);

            // Queue up reduction kernels for each batch
            for (std::size_t b = 0; b < batches; ++b) {
                std::size_t step_idx = b * output_interval + local_step;
                Backend::record_observables_async(sim_.get_state().get_u(b), sim_.get_size(),
                                                  d_mean_u.data(), d_mean_u2.data(),
                                                  d_mean_sin.data(), step_idx);
            }

            // Sync and evaluate only when the interval is full
            if (step % output_interval == 0) {
                Backend::synchronize();

                auto h_mean_u = Backend::download_array(d_mean_u.data(), d_mean_u.size());
                auto h_mean_u2 = Backend::download_array(d_mean_u2.data(), d_mean_u2.size());
                auto h_mean_sin = Backend::download_array(d_mean_sin.data(), d_mean_sin.size());

                write_state(h_mean_u, h_mean_u2, h_mean_sin, h_val_history, output_interval);
            }
        }
        Backend::synchronize();
    }

   private:
    DomainWall<Backend>& sim_;
    std::ofstream out_;
    Real current_time_;

    void write_state(const std::vector<Real>& h_mean_u, const std::vector<Real>& h_mean_u2,
                     const std::vector<Real>& h_mean_sin, const std::vector<Real>& h_val_history,
                     int output_interval) {
        std::size_t batches = sim_.get_batch();
        Real alpha = sim_.get_alpha();

        for (std::size_t b = 0; b < batches; ++b) {
            Real sum_u_dot = 0.0;
            Real sum_phi_dot = 0.0;
            Real sum_var_u = 0.0;

            // Compute true interval averages on the CPU using the downloaded GPU history
            for (int s = 0; s < output_interval; ++s) {
                std::size_t idx = b * output_interval + s;

                Real mean_u = h_mean_u[idx];
                Real mean_u2 = h_mean_u2[idx];
                Real mean_sin2phi = h_mean_sin[idx];
                Real h_val = h_val_history[s];

                Real var_u = mean_u2 - (mean_u * mean_u);
                Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
                Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);

                sum_u_dot += u_dot;
                sum_phi_dot += phi_dot;
                sum_var_u += var_u;
            }

            out_ << b << " " << current_time_ << " "
                 << (sum_u_dot / static_cast<Real>(output_interval)) << " "
                 << (sum_phi_dot / static_cast<Real>(output_interval)) << " "
                 << (sum_var_u / static_cast<Real>(output_interval)) << "\n";
        }
        out_.flush();
    }
};

}  // namespace dw