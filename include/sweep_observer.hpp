#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "domain_wall.cuh"
#include "sim_utils.hpp"

namespace dw {

template <typename Backend>
class SweepObserver {
   public:
    using Real = typename Backend::PrecisionType::Real;
    using Complex = typename Backend::PrecisionType::Complex;
    using VectorReal = typename Backend::template Vector<Real>;
    using VectorComplex = typename Backend::template Vector<Complex>;

    SweepObserver(DomainWall<Backend>& sim, const std::string& output_file)
        : sim_(sim), out_(output_file) {
        out_ << std::setprecision(10);
        // Output strictly raw continuous variables. Classification happens in Python.
        out_ << "h0 ha mean_var_strobo rugosity v_u v_phi max_q\n";
    }

    ~SweepObserver() {
        if (out_.is_open()) out_.close();
    }

    void run_sweep(double transient_time, double eval_time) {
        std::size_t size = sim_.get_size();
        std::size_t batch = sim_.get_batch();
        Real dt = sim_.get_dt();

        int transient_steps = static_cast<int>(std::round(transient_time / dt));
        int eval_steps = static_cast<int>(std::round(eval_time / dt));

        Real current_time = 0.0;

        std::cout << "Running transient for " << transient_steps << " steps..." << std::endl;
        dw::utils::ProgressPrinter transient_progress(transient_steps, 20);
        for (int step = 1; step <= transient_steps; ++step) {
            sim_.step(current_time);
            transient_progress.print("Sweep Transient", step);
        }

        // Standard spatial metrics
        VectorReal d_acc_vu(batch);
        VectorReal d_acc_vphi(batch);
        VectorReal d_acc_rugosity(batch);

        // Stroboscopic temporal metrics
        std::size_t total_elements = size * batch;
        VectorReal d_acc_step(total_elements);
        VectorReal d_acc_step2(total_elements);
        VectorComplex d_u_strobo_prev(total_elements);

        Backend::fill_zero(d_acc_vu.data(), batch);
        Backend::fill_zero(d_acc_vphi.data(), batch);
        Backend::fill_zero(d_acc_rugosity.data(), batch);
        Backend::fill_zero(d_acc_step.data(), total_elements);
        Backend::fill_zero(d_acc_step2.data(), total_elements);

        // Store the initial state for the first stroboscopic step reference
        Backend::copy(sim_.get_state().u.data(), d_u_strobo_prev.data(), total_elements);

        Real T_lock = 4.0 * M_PI / sim_.get_omega();
        int steps_per_strobo = static_cast<int>(std::round(T_lock / dt));
        int num_strobos = 0;

        std::cout << "Evaluating metrics for " << eval_steps << " steps..." << std::endl;
        dw::utils::ProgressPrinter eval_progress(eval_steps, 20);
        for (int step = 1; step <= eval_steps; ++step) {
            Backend::copy(sim_.get_state().u.data(), sim_.get_state().u_prev.data(),
                          total_elements);
            sim_.step(current_time);

            Backend::accumulate_sweep_metrics(
                sim_.get_state().u.data(), sim_.get_state().u_prev.data(), d_acc_vu.data(),
                d_acc_vphi.data(), d_acc_rugosity.data(), size, batch, dt);

            if (step % steps_per_strobo == 0) {
                Backend::accumulate_strobo_metrics(sim_.get_state().u.data(),
                                                   d_u_strobo_prev.data(), d_acc_step.data(),
                                                   d_acc_step2.data(), size, batch);
                // Update the stroboscopic reference frame for the next interval
                Backend::copy(sim_.get_state().u.data(), d_u_strobo_prev.data(), total_elements);
                num_strobos++;
            }
            eval_progress.print("Sweep Eval", step);
        }

        Backend::synchronize();
        process_and_save(eval_steps, num_strobos, d_acc_vu, d_acc_vphi, d_acc_rugosity, d_acc_step,
                         d_acc_step2);
    }

   private:
    DomainWall<Backend>& sim_;
    std::ofstream out_;

    void process_and_save(int eval_steps, int num_strobos, const VectorReal& d_acc_vu,
                          const VectorReal& d_acc_vphi,
                          const VectorReal& d_acc_rugosity,
                          const VectorReal& d_acc_step, const VectorReal& d_acc_step2) {
        std::size_t size = sim_.get_size();
        std::size_t batch = sim_.get_batch();
        Real L = sim_.get_L();  // Fixed: Fetch dynamic length

        auto h_vu = Backend::download_array(d_acc_vu.data(), batch);
        auto h_vphi = Backend::download_array(d_acc_vphi.data(), batch);
        auto h_rugosity = Backend::download_array(d_acc_rugosity.data(), batch);

        auto h_step = Backend::download_array(d_acc_step.data(), size * batch);
        auto h_step2 = Backend::download_array(d_acc_step2.data(), size * batch);

        auto h0_batch = sim_.get_h0_batch();
        auto ha_batch = sim_.get_ha_batch();

        // Fixed: Allocate device memory once outside the batch loop
        VectorReal d_ps(size);

        for (std::size_t b = 0; b < batch; ++b) {
            Real avg_vu = h_vu[b] / static_cast<Real>(eval_steps);
            Real avg_vphi = h_vphi[b] / static_cast<Real>(eval_steps);
            Real rugosity = h_rugosity[b] / static_cast<Real>(eval_steps);

            Real mean_var_strobo = 0.0;
            if (num_strobos > 0) {
                Real inv_N = Real(1.0) / static_cast<Real>(num_strobos);
                Real sum_var = 0.0;

                for (std::size_t i = 0; i < size; ++i) {
                    Real mu = h_step[b * size + i] * inv_N;
                    Real mu2 = h_step2[b * size + i] * inv_N;
                    Real var = std::max(Real(0.0), mu2 - (mu * mu));
                    sum_var += var;
                }
                mean_var_strobo = sum_var / static_cast<Real>(size);
            }

            // Reuse the same device allocation
            Backend::fill_zero(d_ps.data(), size);
            Backend::accumulate_power_spectrum(sim_.get_state().get_u_hat(b), d_ps.data(), size);
            auto h_ps = Backend::download_array(d_ps.data(), size);

            Real max_ps = -1.0;
            std::size_t max_idx = 0;
            for (std::size_t i = 1; i <= size / 2; ++i) {
                if (h_ps[i] > max_ps) {
                    max_ps = h_ps[i];
                    max_idx = i;
                }
            }
            Real max_q = max_idx * (2.0 * M_PI / L);

            out_ << h0_batch[b] << " " << ha_batch[b] << " " << mean_var_strobo << " " << rugosity
                 << " " << avg_vu << " " << avg_vphi << " " << max_q << "\n";
        }
        out_.flush();
    }
};

}  // namespace dw
