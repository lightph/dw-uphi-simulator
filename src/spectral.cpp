#include "pde_pseudospectral_solver.h"
#include <chrono>
#include <cmath>
#include <complex>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// Helper function to replace std::bit_ceil for C++17
// Rounds up to the nearest power of 2
size_t next_pow2(size_t n) {
  if (n <= 1)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32; // Ensures it works for 64-bit size_t
  return n + 1;
}

struct NonlinearPart {
  double h;
  double h0;
  double Omega;
  const double *t_ptr;
  double alpha;

  std::complex<double> operator()(std::complex<double> z) const {
    double current_h = h + h0 * std::cos(Omega * (*t_ptr));
    return std::complex<double>(alpha / 2.0, -0.5) *
           std::complex<double>(alpha * current_h, std::sin(2.0 * z.imag()));
  }
};

struct LinearPart {
  double alpha;
  std::complex<double> operator()(double k) const {
    return -0.5 * std::complex<double>(alpha, -1.0) * (k * k);
  }
};

struct InitialCondition {
  double eps;
  mutable std::mt19937 gen;
  mutable std::uniform_real_distribution<double> dist;

  InitialCondition(double eps_val, int seed_offset)
      : eps(eps_val), gen(std::random_device{}() + seed_offset),
        dist(0.0, eps_val) {}

  std::complex<double> operator()(double /*x*/) const {
    return std::complex<double>(dist(gen), -dist(gen));
  }
};

int main() {
  const double alpha = 0.27;
  const double delta_t = 0.02;
  const double time_transient = 1000;
  const double time_measure = 1000;
  const int n_transient = std::round(time_transient / delta_t);
  const int n_measure = std::round(time_measure / delta_t);

  const int num_h_values = 200;
  const int num_h0_values = 101;
  const double PI = std::acos(-1.0);

  for (int step_L = 1; step_L <= 32; ++step_L) {
    double sys_L = step_L * (PI / 4.0);
    double L = sys_L;

    double min_slots = std::ceil(sys_L / 0.1);
    // Replaced std::__bit_ceil with our C++17 compatible helper
    size_t vec_size = next_pow2(static_cast<size_t>(min_slots));

    double k1 = 2.0 * PI / sys_L;

    std::ostringstream filename;
    filename << "output/pseudospectral/velocity_vs_h_h0_v3_k1_" << std::fixed
             << std::setprecision(4) << k1 << ".bin";

    std::ofstream out_file(filename.str(), std::ios::binary);

    for (int step_h = 0; step_h < num_h_values; ++step_h) {
      double h = 1.0 + step_h * (7. / (num_h_values - 1));
      double Omega = 1.;
      for (int step_h0 = 0; step_h0 < num_h0_values; ++step_h0) {
        double h0 = 1.0 + step_h0 * (99.0 / (num_h0_values - 1));

        double current_t = 0.0;

        NonlinearPart nonlin{h, h0, Omega, &current_t, alpha};
        LinearPart lin{alpha};
        InitialCondition init_cond(0.02, L * num_h_values * num_h0_values +
                                             step_h * num_h0_values + step_h0);

        PDE_pseudospectral_stepper<NonlinearPart, InitialCondition, LinearPart>
            stepper(vec_size, sys_L, delta_t, nonlin, init_cond, lin);

        for (int n = 0; n < n_transient; ++n) {
          stepper.step();
          current_t += delta_t;
        }

        double u_start =
            stepper.get_zhat().data()[0].real() / static_cast<double>(vec_size);

        for (int n = 0; n < n_measure; ++n) {
          stepper.step();
          current_t += delta_t;
        }

        double u_end =
            stepper.get_zhat().data()[0].real() / static_cast<double>(vec_size);

        double time_measured = n_measure * delta_t;
        double v_avg = (u_end - u_start) / time_measured;

        double data[3] = {h, h0, v_avg};
        if (out_file.is_open()) {
          out_file.write(reinterpret_cast<const char *>(data), sizeof(data));
          out_file.flush();
        }

        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);

        std::cout << std::put_time(std::localtime(&now_time), "%H:%M:%S")
                  << " | "
                  << "L: " << std::fixed << std::setprecision(2) << std::setw(5)
                  << sys_L << " | "
                  << "h: " << std::fixed << std::setprecision(2) << std::setw(4)
                  << h << " | "
                  << "h0: " << std::setw(6) << h0 << " | "
                  << "v_avg = " << std::setprecision(6) << std::setw(10)
                  << v_avg << "\n";
      }
    }
  }

  return 0;
}
