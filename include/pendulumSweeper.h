#pragma once

#include "ODPendulum.h"
#include "filesaver.h"
#include "pendulumAnalyser.h"
#include <algorithm>
#include <utility>
#include <vector>

template <typename T> struct SweepOutput {
  std::vector<double> sweepVar;
  std::vector<T> result;

  SweepOutput() = default;

  SweepOutput(const SweepOutput &other)
      : sweepVar(other.sweepVar), result(other.result) {}

  SweepOutput &operator=(SweepOutput other) {
    swap(*this, other);
    return *this;
  }

  SweepOutput(SweepOutput &&other) noexcept
      : sweepVar(std::move(other.sweepVar)), result(std::move(other.result)) {}

  friend void swap(SweepOutput &first, SweepOutput &second) noexcept {
    using std::swap;
    swap(first.sweepVar, second.sweepVar);
    swap(first.result, second.result);
  }
};

class PendulumSweeper {
private:
public:
  SweepOutput<double> getAvgVelRange(DrivenODPendulum &solver,
                                     PendulumAnalyser &analyser, double Amin,
                                     double Amax, size_t n,
                                     DataSaver *saver = nullptr);

  SweepOutput<SweepOutput<SweepOutput<double>>>
  getAvgVelRange3D(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                   double Amin, double Amax, int nA, double Bmin, double Bmax,
                   int nB, double Cmin, double Cmax, int nC,
                   DataSaver *saver = nullptr);

  std::pair<double, double> measureShapiroStepBounds(DrivenODPendulum &solver,
                                                     PendulumAnalyser &analyser,
                                                     int nstep, double Atol,
                                                     double vtol, double Aleft,
                                                     double Aright);

  double measureShapiroStep(DrivenODPendulum &solver,
                            PendulumAnalyser &analyser, int nstep, double Atol,
                            double vtol, double Aleft, double Aright);

  SweepOutput<std::pair<double, double>>
  measureShapiroStepBoundsRange(DrivenODPendulum &solver,
                                PendulumAnalyser &analyser, int nstep,
                                double Cmin, double Cmax, int n, double Atol,
                                double vtol, double Aleft, double Aright);

  SweepOutput<double> measureShapiroStepRange(DrivenODPendulum &solver,
                                              PendulumAnalyser &analyser,
                                              int nstep, double Cmin,
                                              double Cmax, int n, double Atol,
                                              double vtol, double Aleft,
                                              double Aright);

  SweepOutput<SweepOutput<std::pair<double, double>>>
  measureShapiroStepBoundsRange2D(DrivenODPendulum &solver,
                                  PendulumAnalyser &analyser, int nstep,
                                  double Cmin, double Cmax, int nC, double Bmin,
                                  double Bmax, int nB, double Atol, double vtol,
                                  double Aleft, double Aright);

  SweepOutput<SweepOutput<double>> measureShapiroStepRange2D(
      DrivenODPendulum &solver, PendulumAnalyser &analyser, int nstep,
      double Cmin, double Cmax, int nC, double Bmin, double Bmax, int nB,
      double Atol, double vtol, double Aleft, double Aright);

  SweepOutput<PendulumFourierResult>
  measureFFTsARange(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                    double Amin, double Amax, int nA, int powerof2,
                    DataSaver *saver, double maxfreq = -1, int downsample = 1);

  SweepOutput<SweepOutput<PendulumFourierResult>>
  measureFFTsACRange(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                     double Amin, double Amax, int nA, double Cmin, double Cmax,
                     int nC, int powerof2, DataSaver *saver,
                     double maxfreq = -1, int downsample = 1);

  SweepOutput<SweepOutput<SweepOutput<PendulumFourierResult>>>
  measureFFTsRange3D(DrivenODPendulum &solver, PendulumAnalyser &analyser,
                     double Amin, double Amax, int nA, double Bmin, double Bmax,
                     int nB, double Cmin, double Cmax, int nC, int powerof2,
                     DataSaver *saver, double maxfreq = -1, int downsample = 1);
};
