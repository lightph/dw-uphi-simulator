#include "PhiSweeper.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

void PhiSweeper::getAvgVelRange3D(DrivenODPendulum &solver,
                                  PendulumAnalyser &analyser, double hmin,
                                  double hmax, size_t nh, double B_min,
                                  double B_max, size_t B_n, double h0min,
                                  double h0max, size_t nh0, DataSaver *saver,
                                  std::vector<std::array<double, 4>> *output) {

  nh = nh == 0 ? 1 : nh;
  B_n = B_n == 0 ? 1 : B_n;
  nh0 = nh0 == 0 ? 1 : nh0;

  size_t total_points = nh * nh0 * B_n;
  if (output) {
    output->resize(total_points);
  }

  double h_step = nh > 1 ? (hmax - hmin) / static_cast<double>(nh - 1) : 0.;
  double B_step = B_n > 1 ? (B_max - B_min) / static_cast<double>(B_n - 1) : 0.;
  double h0_step =
      nh0 > 1 ? (h0max - h0min) / static_cast<double>(nh0 - 1) : 0.;

  auto start_time = std::chrono::steady_clock::now();
  size_t points_processed = 0;
  size_t step_1_percent = std::max<size_t>(1, total_points / 100);
  size_t next_threshold = step_1_percent;

  for (size_t i = 0; i < nh; ++i) {
    double h_curr = hmin + i * h_step;
    for (size_t j = 0; j < B_n; ++j) {
      double B_curr = B_min + j * B_step;

      double Omega_curr = 1.0 / B_curr;

      double A = B_curr * h_curr;

      for (size_t k = 0; k < nh0; ++k) {
        double h0_curr = h0min + k * h0_step;

        double C = B_curr * h0_curr;

        solver.setA(A);
        solver.setB(B_curr);
        solver.setC(C);

        auto solver_output = solver.solve();

        double vel = analyser.getAvgVel(solver_output) * Omega_curr / 2.0;

        // Save the physical variables: h, B (alpha/Omega ratio), h0, and
        // velocity
        if (output) {
          (*output)[i * (B_n * nh0) + j * nh0 + k] = {h_curr, B_curr, h0_curr,
                                                      vel};
        }

        if (saver) {
          std::vector<double> row = {h_curr, B_curr, h0_curr, vel};
          saver->saveRow(row);
        }

        points_processed++;
        if (points_processed >= next_threshold ||
            points_processed == total_points) {
          auto current_time = std::chrono::steady_clock::now();
          std::chrono::duration<double> elapsed = current_time - start_time;

          double percent = static_cast<double>(points_processed) / total_points;
          double eta = (elapsed.count() / percent) - elapsed.count();

          std::cout << "\rProgress: " << static_cast<int>(percent * 100) << "% "
                    << std::fixed << std::setprecision(1)
                    << "| Elapsed: " << elapsed.count() << "s "
                    << "| ETA: " << eta << "s   " << std::flush;

          next_threshold += step_1_percent;
        }
      }
    }
  }
  std::cout << std::endl;
}

void measureFFTsRange3D(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                        double Amin, double Amax, int nA, double Bmin,
                        double Bmax, int nB, double Cmin, double Cmax, int nC,
                        int powerof2, DataSaver *saver, double maxfreq = -1,
                        int downsample = 1,
                        std::vector<std::vector<double>> *output = nullptr);
