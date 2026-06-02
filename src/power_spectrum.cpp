#include "FftwHandler.h"
#include "pde_pseudospectral_solver.h"
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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
  const double delta_t = 0.1;
  const double time_transient = 10000;
  const double time_measure = 10000;
  const int n_transient = std::round(time_transient / delta_t);
  const int n_measure = std::round(time_measure / delta_t);
  const int num_h0_values = 101;
  const double PI = std::acos(-1.0);

  const double h_fixed = 2.5;
  const double Omega = 0.5 * std::sqrt(h_fixed * h_fixed - 1.0);

  double L = 4 * M_PI;
  double sys_L = L;
  double min_slots = std::ceil(sys_L / 0.01);
  size_t vec_size = std::__bit_ceil(static_cast<size_t>(min_slots));

  double k1 = 2.0 * PI / sys_L;

  std::ostringstream filename;
  filename << "output/pseudospectral/power_spectrum_vs_h0_v2_k1_" << std::fixed
           << std::setprecision(4) << k1 << ".bin";

  std::ofstream out_file(filename.str(), std::ios::binary);

  fftw_ctc_handler fft_handler;
  fftw_buffer fft_in(n_measure);
  fftw_buffer fft_out(n_measure);

  size_t n_save = n_measure / 2 + 1;
  std::vector<double> power_spectrum(n_save);

  for (int step = 0; step < num_h0_values; ++step) {
    // h0 values swept between 0 and 10
    double h0 = step * (100.0 / (num_h0_values - 1));

    // Time tracking variable to pass into the nonlinear part
    double current_t = 0.0;

    NonlinearPart nonlin{h_fixed, h0, Omega, &current_t, alpha};
    LinearPart lin{alpha};
    InitialCondition init_cond(0.02, L * num_h0_values + step);

    PDE_pseudospectral_stepper<NonlinearPart, InitialCondition, LinearPart>
        stepper(vec_size, sys_L, delta_t, nonlin, init_cond, lin);

    for (int n = 0; n < n_transient; ++n) {
      stepper.step();
      current_t += delta_t; // Advance time
    }

    for (int n = 0; n < n_measure; ++n) {
      stepper.step();
      current_t += delta_t; // Advance time
      double u_val =
          stepper.get_zhat().data()[0].real() / static_cast<double>(vec_size);
      fft_in.data()[n] = std::complex<double>(u_val, 0.0);
    }

    fft_handler.do_fft(fft_in, fft_out);

    for (size_t n = 0; n < n_save; ++n) {
      double re = fft_out.data()[n].real();
      double im = fft_out.data()[n].imag();
      power_spectrum[n] = (re * re + im * im);
    }

    if (out_file.is_open()) {
      out_file.write(reinterpret_cast<const char *>(&h0), sizeof(h0));
      out_file.write(reinterpret_cast<const char *>(power_spectrum.data()),
                     n_save * sizeof(double));
      out_file.flush();
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::cout << std::put_time(std::localtime(&now_time), "%H:%M:%S") << " | "
              << "L: " << std::setw(3) << L << " | "
              << "Step: " << std::setw(2) << step << " / "
              << (num_h0_values - 1) << " | h0 = " << std::fixed
              << std::setprecision(4) << std::setw(7) << h0 << "\n";
  }

  return 0;
}
