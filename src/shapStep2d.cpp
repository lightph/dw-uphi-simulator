#include "ODPendulum.h"
#include "pendulumSweeper.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

void runAndSaveShapiroSweep(const std::string &outDir, int nstep = 0,
                            double Bmin = 0., double Bmax = 1., int nB = 100,
                            double Cmin = 0., double Cmax = 100., int nC = 1000,
                            double Atol = 1e-6, double vtol = 1e-4,
                            double Aleft = 0., double Aright = 10.) {

  int waitNPeriods = 100;
  int averageNPeriods = 100;

  double h = 100 * 2 * M_PI / static_cast<double>(1 << 13);
  DrivenODPendulum solver(1., -1., 1., h, 0., 0.01, (2 << 14) + 50);
  PendulumAnalyser analyser(waitNPeriods, averageNPeriods);
  PendulumSweeper sweeper;

  auto result = sweeper.measureShapiroStepBoundsRange2D(
      solver, analyser, nstep, Cmin, Cmax, nC, Bmin, Bmax, nB, Atol, Atol,
      Aleft, Aright);

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

  std::stringstream filename_ss;
  filename_ss << "shapiro_step" << nstep << "_B" << Bmin << "-" << Bmax
              << "_pts" << nB << "_C" << Cmin << "-" << Cmax << "_pts" << nC
              << "_" << time_ss.str() << ".csv";

  std::filesystem::path dir(outDir);
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::filesystem::path filepath = dir / filename_ss.str();

  std::ofstream file(filepath);
  file << "B,C,stepBegin,stepEnd\n";

  for (size_t i = 0; i < result.sweepVar.size(); ++i) {
    double B = result.sweepVar[i];
    auto result2 = result.result[i];
    for (size_t j = 0; j < result2.sweepVar.size(); ++j) {
      double C = result2.sweepVar[j];
      auto bounds = result2.result[j];

      file << B << ',' << C << ',' << bounds.first << ',' << bounds.second
           << '\n';
    }
  }
}

int main(int argc, char **argv) {
  std::string outDir = "output/shapiro2d";
  double Bmin = 0., Bmax = 100.;
  double Cmin = 0., Cmax = 100.;
  int nB = 1000, nC = 1000;

  auto future1 =
      std::async(std::launch::async, runAndSaveShapiroSweep, outDir, 0, Bmin,
                 Bmax, nB, Cmin, Cmax, nC, 1e-6, 1e-6, 0., 10.);
  auto future2 =
      std::async(std::launch::async, runAndSaveShapiroSweep, outDir, 2, Bmin,
                 Bmax, nB, Cmin, Cmax, nC, 1e-6, 1e-4, 1., 20.);
  auto future3 =
      std::async(std::launch::async, runAndSaveShapiroSweep, outDir, 1, Bmin,
                 Bmax, nB, Cmin, Cmax, nC, 1e-6, 1e-4, 0., 10.);
  auto future4 =
      std::async(std::launch::async, runAndSaveShapiroSweep, outDir, 3, Bmin,
                 Bmax, nB, Cmin, Cmax, nC, 1e-6, 1e-4, 2., 20.);
  auto future5 =
      std::async(std::launch::async, runAndSaveShapiroSweep, outDir, 4, Bmin,
                 Bmax, nB, Cmin, Cmax, nC, 1e-6, 1e-4, 3., 20.);

  future1.wait();
  future2.wait();
  future3.wait();
  future4.wait();
  future5.wait();

  return 0;
}
