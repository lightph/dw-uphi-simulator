#pragma once

#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "domain_wall.hpp"

namespace dw {

template <typename Backend>
class SimulationObserver {
   public:
    using Real = typename Backend::PrecisionType::Real;
    using ComplexVector =
        typename Backend::template Vector<typename Backend::PrecisionType::Complex>;

    SimulationObserver(DomainWall<Backend>& sim, const std::string& output_file)
        : sim_(sim), out_(output_file), current_time_(0.0), u_old_(sim.get_size()) {
        out_ << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ << "time avg_u_dot avg_phi_dot var_u phase_variance\n";
    }

    ~SimulationObserver() {
        if (out_.is_open()) out_.close();
    }

    void run_until_settled(int W, int N, double tolerance) {
        std::vector<Real> step_velocities(W);
        std::deque<Real> window_variances;
        Real dt = sim_.get_dt();

        Backend::synchronize();

        while (true) {
            Real mean_velocity = 0.0;

            for (int i = 0; i < W; ++i) {
                Backend::copy(sim_.get_state().u, u_old_);
                sim_.step(current_time_);

                Real diff_sq = Backend::compute_diff_sq(u_old_, sim_.get_state().u);
                step_velocities[i] = diff_sq / (dt * dt);
                mean_velocity += step_velocities[i];

                accumulate_observables();
            }

            mean_velocity /= W;
            Real window_var = 0.0;
            for (Real v : step_velocities) {
                window_var += (v - mean_velocity) * (v - mean_velocity);
            }
            window_var /= W;

            window_variances.push_back(window_var);
            if (window_variances.size() > N) {
                window_variances.pop_front();
            }

            write_state(window_var);

            if (window_variances.size() == N) {
                Real var_mean = 0.0;
                for (Real v : window_variances) var_mean += v;
                var_mean /= N;

                Real var_std = 0.0;
                for (Real v : window_variances) var_std += (v - var_mean) * (v - var_mean);
                var_std = std::sqrt(var_std / N);

                if (var_std < tolerance) {
                    std::cout << "System settled at t = " << current_time_
                              << " with std(variances) = " << var_std << "\n";
                    break;
                }
            }
        }
    }

   private:
    DomainWall<Backend>& sim_;
    std::ofstream out_;
    Real current_time_;
    ComplexVector u_old_;

    Real sum_avg_u_dot_ = 0.0;
    Real sum_avg_phi_dot_ = 0.0;
    Real sum_var_u_ = 0.0;
    unsigned long long sample_count_ = 0;

    void accumulate_observables() {
        auto obs = Backend::compute_observables(sim_.get_state().u);

        Real N_grid = static_cast<Real>(sim_.get_size());
        Real mean_sin2phi = obs.sin2phi / N_grid;
        Real mean_u = obs.u / N_grid;
        Real mean_u2 = obs.u2 / N_grid;

        Real alpha = sim_.get_alpha();
        Real h_val = sim_.get_h0() + sim_.get_ha() * std::cos(sim_.get_omega() * current_time_);

        Real u_dot = 0.5 * (alpha * alpha * h_val + mean_sin2phi);
        Real phi_dot = 0.5 * (alpha * h_val - alpha * mean_sin2phi);
        Real var_u = mean_u2 - (mean_u * mean_u);

        sum_avg_u_dot_ += u_dot;
        sum_avg_phi_dot_ += phi_dot;
        sum_var_u_ += var_u;
        sample_count_++;
    }

    void write_state(Real current_variance) {
        out_ << current_time_ << " " << (sum_avg_u_dot_ / sample_count_) << " "
             << (sum_avg_phi_dot_ / sample_count_) << " " << (sum_var_u_ / sample_count_) << " "
             << current_variance << "\n";

        sum_avg_u_dot_ = 0.0;
        sum_avg_phi_dot_ = 0.0;
        sum_var_u_ = 0.0;
        sample_count_ = 0;
    }
};

}  // namespace dw