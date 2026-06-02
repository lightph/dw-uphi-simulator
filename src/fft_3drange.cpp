#include "ODPendulum.h"
#include "filesaver.h"
#include "pendulumSweeper.h"
#include <cmath>
#include <iostream>
#include <string>

void runAndSaveFFTSweep3D(const std::string &outDir, double Amin, double Amax,
                          int nA, double Bmin, double Bmax, int nB, double Cmin,
                          double Cmax, int nC) {

  int waitNPeriods = 200;
  int averageNPeriods = 50;
  int powerof2 = 15;
  int downsample = 1;
  double maxfreq = 15.0 / (2.0 * M_PI);

  double T_total = averageNPeriods * 2.0 * M_PI;
  double df = 1.0 / T_total;
  std::cout << "Frequency resolution: " << df << " (standard) | "
            << df * 2.0 * M_PI << " (angular)\n";

  double h = averageNPeriods * 2 * M_PI / static_cast<double>(1ULL << powerof2);
  DrivenODPendulum solver(1., -0.5, 25., h, 0., 0.01,
                          (2 << 15) + 2 * M_PI * waitNPeriods / h + 5);
  PendulumAnalyser analyser(waitNPeriods, averageNPeriods);
  PendulumSweeper sweeper;

  std::map<std::string, std::string> params = {
      {"Amin", std::to_string(Amin)}, {"Amax", std::to_string(Amax)},
      {"nA", std::to_string(nA)},     {"Bmin", std::to_string(Bmin)},
      {"Bmax", std::to_string(Bmax)}, {"nB", std::to_string(nB)},
      {"Cmin", std::to_string(Cmin)}, {"Cmax", std::to_string(Cmax)},
      {"nC", std::to_string(nC)}};

  std::string expName = "fft_sweep_3D";

  BinarySaver saver(outDir, expName, params);

  sweeper.measureFFTsRange3D(solver, analyser, Amin, Amax, nA, Bmin, Bmax, nB,
                             Cmin, Cmax, nC, powerof2, &saver, maxfreq,
                             downsample);
}

int main(int argc, char **argv) {
  std::string outDir = "output/fft3d";

  runAndSaveFFTSweep3D(outDir, 0.0, 10.0, 1000, 0.0, 10.0, 10, 0.0, 10.0, 10);

  return 0;
}
