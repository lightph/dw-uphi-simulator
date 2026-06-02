#pragma once

#include "pendulumSweeper.h"
#include <array>
#include <cstddef>
#include <vector>

class PhiSweeper : PendulumSweeper {
private:
  double getA(double h, double Omega) { return h / Omega; }
  double getB(double Omega) { return 1 / Omega; }
  double getC(double h0, double Omega) { return h0 / Omega; }

public:
  void getAvgVelRange3D(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                        double hmin, double hmax, size_t nh, double B_min,
                        double B_max, size_t B_n, double h0min, double h0max,
                        size_t nh0, DataSaver *saver = nullptr,
                        std::vector<std::array<double, 4>> *output = nullptr);

  void measureFFTsRange3D(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                          double Amin, double Amax, int nA, double Bmin,
                          double Bmax, int nB, double Cmin, double Cmax, int nC,
                          int powerof2, DataSaver *saver, double maxfreq = -1,
                          int downsample = 1,
                          std::vector<std::vector<double>> *output = nullptr);
};
