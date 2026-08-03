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
        : sim_(sim), out_(output_file), current_time_(0.0) {
        out_ << std::setprecision(std::numeric_limits<Real>::max_digits10);
        out_ << "time avg_u_dot avg_phi_dot var_u phase_variance\n";
    }

    ~SimulationObserver() {
        if (out_.is_open()) out_.close();
    }

    void run_until_settled(int W, double tolerance) {
        Backend::synchronize();

        Real previous_mean_var = -1.0;

        while (true) {
            for (int i = 0; i < W; ++i) {
                sim_.step(current_time_);
                accumulate_observables();
            }

            Real current_mean_var = sum_var_u_ / sample_count_;

            if (previous_mean_var >= 0.0) {
                Real drift = std::abs(current_mean_var - previous_mean_var);

                if (drift < tolerance) {
                    std::cout << "System settled at t = " << current_time_
                              << " with variance drift = " << drift << "\n";

                    // Write the final state before breaking out
                    write_state(current_mean_var);
                    break;
                }
            }

            write_state(current_mean_var);
            previous_mean_var = current_mean_var;
        }
    }

   private:
    DomainWall<Backend>& sim_;
    std::ofstream out_;
    Real current_time_;

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