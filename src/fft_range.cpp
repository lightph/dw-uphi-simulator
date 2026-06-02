#include "ODPendulum.h"
#include "filesaver.h"
#include "pendulumSweeper.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

void runAndSaveFFTSweepA(const std::string &outDir, double Amin = 0.,
                         double Amax = 2., int nA = 100) {

  int waitNPeriods = 100;
  int averageNPeriods = 1000;
  int powerof2 = 15;
  int downsample = 1;
  double maxfreq = 1 / (2 * M_PI);

  double T_total = averageNPeriods * 2.0 * M_PI;
  double df = 1.0 / T_total;
  std::cout << "Frequency resolution: " << df << " (standard) | "
            << df * 2.0 * M_PI << " (angular)" << '\n';

  double h = averageNPeriods * 2 * M_PI / static_cast<double>(1ULL << powerof2);
  DrivenODPendulum solver(1., -0.5, 25., h, 0., 0.01, (2 << 15) + 50);
  PendulumAnalyser analyser(waitNPeriods, averageNPeriods);
  PendulumSweeper sweeper;

  std::map<std::string, std::string> params = {{"Amin", std::to_string(Amin)},
                                               {"Amax", std::to_string(Amax)},
                                               {"nA", std::to_string(nA)}};

  std::string expName = "fft_sweep";

  std::vector<std::string> headers = {"A", "B", "C", "Frequency",
                                      "PowerSpectrum"};
  CsvSaver saver(outDir, expName, params, headers);

  sweeper.measureFFTsARange(solver, analyser, Amin, Amax, nA, powerof2, &saver,
                            maxfreq, downsample);
}

int main(int argc, char **argv) {
  std::string outDir = "output/fft1d";

  runAndSaveFFTSweepA(outDir, 0.0, 10, 10000);

  return 0;
}
