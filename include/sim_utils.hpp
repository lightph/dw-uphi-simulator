#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace dw {
namespace utils {

class ProgressPrinter {
    unsigned long long total_steps_;
    unsigned long long print_interval_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

   public:
    ProgressPrinter(unsigned long long total_steps, unsigned long long num_prints = 20)
        : total_steps_(total_steps) {
        print_interval_ = std::max<unsigned long long>(1, total_steps_ / num_prints);
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void print(const std::string& phase, unsigned long long current_step) {
        if (current_step % print_interval_ == 0 || current_step == total_steps_) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start_time_;
            double steps_per_sec = current_step / elapsed.count();
            double eta_sec = (total_steps_ - current_step) / steps_per_sec;

            int eta_h = static_cast<int>(eta_sec) / 3600;
            int eta_m = (static_cast<int>(eta_sec) % 3600) / 60;
            int eta_s = static_cast<int>(eta_sec) % 60;

            std::cout << "[" << phase << "] Step " << current_step << " / " << total_steps_
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * current_step / total_steps_) << "%) "
                      << "| Elapsed: " << std::setprecision(1) << elapsed.count() << "s "
                      << "| ETA: " << std::setfill('0') << std::setw(2) << eta_h << ":"
                      << std::setw(2) << eta_m << ":" << std::setw(2) << eta_s << " ("
                      << std::setprecision(0) << steps_per_sec << " steps/s)" << std::endl;
        }
    }
};

inline std::vector<unsigned long long> generate_log_checkpoints(unsigned long long total_steps,
                                                                int num_points = 20,
                                                                unsigned long long start_step = 1) {
    std::vector<unsigned long long> cps;
    if (total_steps > 0 && num_points > 0 && total_steps >= start_step) {
        double log_min = std::log10(static_cast<double>(start_step));
        double log_max = std::log10(static_cast<double>(total_steps));
        for (int i = 0; i < num_points; ++i) {
            double divisor = (num_points > 1) ? static_cast<double>(num_points - 1) : 1.0;
            double log_val = log_min + i * (log_max - log_min) / divisor;
            unsigned long long cp =
                static_cast<unsigned long long>(std::round(std::pow(10.0, log_val)));

            if (cp < start_step) cp = start_step;
            if (cp > total_steps) cp = total_steps;
            cps.push_back(cp);
        }
        std::sort(cps.begin(), cps.end());
        cps.erase(std::unique(cps.begin(), cps.end()), cps.end());
    }
    return cps;
}

}  // namespace utils
}  // namespace dw
