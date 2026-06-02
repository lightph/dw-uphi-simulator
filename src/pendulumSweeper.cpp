#include "pendulumSweeper.h"

#include "ODPendulum.h"
#include "filesaver.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

SweepOutput<double> PendulumSweeper::getAvgVelRange(DrivenODPendulum &solver,
                                                    PendulumAnalyser &analyser,
                                                    double Amin, double Amax,
                                                    size_t n,
                                                    DataSaver *saver) {
  SweepOutput<double> output;
  output.sweepVar.resize(n + 1);
  output.result.resize(n + 1);

  double Astep = (Amax - Amin) / static_cast<double>(n);

  for (size_t i = 0; i < output.sweepVar.size(); ++i) {
    solver.setA(Amin + i * Astep);
    PendulumResult runResult = solver.solve();
    output.sweepVar[i] = solver.getA();
    output.result[i] = analyser.getAvgVel(runResult);

    if (saver) {
      saver->saveRow(DataRow(
          {solver.getA(), solver.getB(), solver.getC(), output.result[i]}));
    }
  }

  return output;
}

SweepOutput<SweepOutput<SweepOutput<double>>> PendulumSweeper::getAvgVelRange3D(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, double Amin,
    double Amax, int nA, double Bmin, double Bmax, int nB, double Cmin,
    double Cmax, int nC, DataSaver *saver) {
  SweepOutput<SweepOutput<SweepOutput<double>>> output;
  output.sweepVar.resize(nA + 1);
  output.result.resize(nA + 1);

  double Astep = (Amax - Amin) / static_cast<double>(nA);
  double Bstep = (Bmax - Bmin) / static_cast<double>(nB);
  double Cstep = (Cmax - Cmin) / static_cast<double>(nC);

  long long totalPoints = static_cast<long long>(nA + 1) * (nB + 1) * (nC + 1);
  long long completedPoints = 0;
  int lastPercentage = -1;

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setA(Amin + i * Astep);
    output.sweepVar[i] = solver.getA();

    SweepOutput<SweepOutput<double>> &outputB = output.result[i];
    outputB.sweepVar.resize(nB + 1);
    outputB.result.resize(nB + 1);

    for (int j = 0; j < outputB.sweepVar.size(); ++j) {
      solver.setB(Bmin + j * Bstep);
      outputB.sweepVar[j] = solver.getB();

      SweepOutput<double> &outputC = outputB.result[j];
      outputC.sweepVar.resize(nC + 1);
      outputC.result.resize(nC + 1);

      for (int k = 0; k < outputC.sweepVar.size(); ++k) {
        solver.setC(Cmin + k * Cstep);
        PendulumResult runResult = solver.solve();
        outputC.sweepVar[k] = solver.getC();
        outputC.result[k] = analyser.getAvgVel(runResult);

        if (saver) {
          saver->saveRow(DataRow({solver.getA(), solver.getB(), solver.getC(),
                                  outputC.result[k]}));
        }
        completedPoints++;
      }
    }

    int currentPercentage =
        static_cast<int>((completedPoints * 100) / totalPoints);
    if (currentPercentage > lastPercentage) {
      std::cout << "\rProgress: " << currentPercentage << "% ("
                << completedPoints << "/" << totalPoints << " points computed)"
                << std::flush;
      lastPercentage = currentPercentage;
    }
  }

  std::cout << "\n";

  return output;
}

std::pair<double, double> PendulumSweeper::measureShapiroStepBounds(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, int nstep,
    double Atol, double vtol, double Aleft, double Aright) {
  double Amid = 0., stepBegin = Aleft, stepEnd = Aright;

  double left = Aleft, right = Aright;
  while (right - left >= Atol) {
    Amid = 0.5 * (left + right);
    solver.setA(Amid);
    PendulumResult runResult = solver.solve();
    if (analyser.getAvgVel(runResult) >= static_cast<double>(nstep) - vtol) {
      right = Amid;
      continue;
    }
    left = Amid;
  }
  stepBegin = Amid;

  left = stepBegin;
  right = Aright;

  while (right - left >= Atol) {
    Amid = 0.5 * (left + right);
    solver.setA(Amid);
    PendulumResult runResult = solver.solve();
    if (analyser.getAvgVel(runResult) > static_cast<double>(nstep) + vtol) {
      right = Amid;
      continue;
    }
    left = Amid;
  }
  stepEnd = Amid;

  return std::make_pair(stepBegin, stepEnd);
}

double PendulumSweeper::measureShapiroStep(DrivenODPendulum &solver,
                                           PendulumAnalyser &analyser,
                                           int nstep, double Atol, double vtol,
                                           double Aleft, double Aright) {
  std::pair<double, double> bounds = measureShapiroStepBounds(
      solver, analyser, nstep, Atol, vtol, Aleft, Aright);
  return bounds.second - bounds.first;
}

SweepOutput<std::pair<double, double>>
PendulumSweeper::measureShapiroStepBoundsRange(DrivenODPendulum &solver,
                                               PendulumAnalyser &analyser,
                                               int nstep, double Cmin,
                                               double Cmax, int n, double Atol,
                                               double vtol, double Aleft,
                                               double Aright) {
  SweepOutput<std::pair<double, double>> output;

  output.sweepVar.resize(n + 1);
  output.result.resize(n + 1);

  double Cstep = (Cmax - Cmin) / static_cast<double>(n);

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setC(Cmin + i * Cstep);
    output.sweepVar[i] = solver.getC();
    output.result[i] = measureShapiroStepBounds(solver, analyser, nstep, Atol,
                                                vtol, Aleft, Aright);
  }
  return output;
}

SweepOutput<double> PendulumSweeper::measureShapiroStepRange(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, int nstep,
    double Cmin, double Cmax, int n, double Atol, double vtol, double Aleft,
    double Aright) {
  SweepOutput<double> output;

  output.sweepVar.resize(n + 1);
  output.result.resize(n + 1);

  double Cstep = (Cmax - Cmin) / static_cast<double>(n);

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setC(Cmin + i * Cstep);
    double step =
        measureShapiroStep(solver, analyser, nstep, Atol, vtol, Aleft, Aright);
    output.sweepVar[i] = solver.getC();
    output.result[i] = step;
  }
  return output;
}

SweepOutput<SweepOutput<std::pair<double, double>>>
PendulumSweeper::measureShapiroStepBoundsRange2D(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, int nstep,
    double Cmin, double Cmax, int nC, double Bmin, double Bmax, int nB,
    double Atol, double vtol, double Aleft, double Aright) {

  SweepOutput<SweepOutput<std::pair<double, double>>> output;
  output.result.resize(nB + 1);
  output.sweepVar.resize(nB + 1);

  double Bstep = (Bmax - Bmin) / static_cast<double>(nB);

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setB(Bmin + i * Bstep);
    output.sweepVar[i] = solver.getB();
    output.result[i] = measureShapiroStepBoundsRange(
        solver, analyser, nstep, Cmin, Cmax, nC, Atol, vtol, Aleft, Aright);
  }

  return output;
}

SweepOutput<SweepOutput<double>> PendulumSweeper::measureShapiroStepRange2D(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, int nstep,
    double Cmin, double Cmax, int nC, double Bmin, double Bmax, int nB,
    double Atol, double vtol, double Aleft, double Aright) {

  SweepOutput<SweepOutput<double>> output;
  output.result.resize(nB + 1);
  output.sweepVar.resize(nB + 1);

  double Bstep = (Bmax - Bmin) / static_cast<double>(nB);
  double Cstep = (Cmax - Cmin) / static_cast<double>(nC);

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setB(Bmin + i * Bstep);
    SweepOutput<double> BRunOutput = measureShapiroStepRange(
        solver, analyser, nstep, Cmin, Cmax, nC, Atol, vtol, Aleft, Aright);
    output.sweepVar[i] = solver.getB();
    output.result[i] = BRunOutput;
  }

  return output;
}

SweepOutput<PendulumFourierResult> PendulumSweeper::measureFFTsARange(
    DrivenODPendulum &solver, PendulumAnalyser &analyser, double Amin,
    double Amax, int nA, int powerof2, DataSaver *saver, double maxfreq,
    int downsample) {
  SweepOutput<PendulumFourierResult> output;
  output.result.resize(nA + 1);
  output.sweepVar.resize(nA + 1);

  double Astep = (Amax - Amin) / static_cast<double>(nA);
  double dt = (2.0 * M_PI * analyser.getAverageNPeriods()) /
              static_cast<double>((1ULL << powerof2) * downsample);
  solver.seth(dt);

  for (size_t i = 0; i < output.sweepVar.size(); ++i) {
    solver.setA(Amin + i * Astep);
    PendulumResult result = solver.solve();
    PendulumFourierResult fftresult =
        analyser.computeFftV(result, downsample, true);

    output.sweepVar[i] = solver.getA();

    if (maxfreq > 0) {
      auto max_freq_it = std::upper_bound(fftresult.freqs.begin(),
                                          fftresult.freqs.end(), maxfreq);
      size_t newsize = std::distance(fftresult.freqs.begin(), max_freq_it);
      fftresult.freqs.resize(newsize);
      fftresult.fourier.resize(newsize);
    }

    output.result[i] = fftresult;
    if (!saver) {
      output.result[i] = fftresult;
    }

    if (saver) {
      std::vector<double> PS = fftresult.computePowerSpectrum();
      for (size_t j = 0; j < fftresult.freqs.size(); ++j) {
        saver->saveRow(DataRow({solver.getA(), solver.getB(), solver.getC(),
                                fftresult.freqs[j], PS[j]}));
      }
    }
  }

  return output;
}

SweepOutput<SweepOutput<PendulumFourierResult>>
PendulumSweeper::measureFFTsACRange(DrivenODPendulum &solver,
                                    PendulumAnalyser &analyser, double Amin,
                                    double Amax, int nA, double Cmin,
                                    double Cmax, int nC, int powerof2,
                                    DataSaver *saver, double maxfreq,
                                    int downsample) {

  SweepOutput<SweepOutput<PendulumFourierResult>> allOutputs;
  allOutputs.result.resize(nC + 1);
  allOutputs.sweepVar.resize(nC + 1);

  double Astep = (Amax - Amin) / static_cast<double>(nA);
  double Cstep = (Cmax - Cmin) / static_cast<double>(nC);
  double dt = (2.0 * M_PI * analyser.getAverageNPeriods()) /
              static_cast<double>((1ULL << powerof2) * downsample);
  solver.seth(dt);

  for (int c_idx = 0; c_idx <= nC; ++c_idx) {
    solver.setC(Cmin + c_idx * Cstep);

    SweepOutput<PendulumFourierResult> &output = allOutputs.result[c_idx];
    output.result.resize(nA + 1);
    output.sweepVar.resize(nA + 1);

    for (size_t i = 0; i < output.sweepVar.size(); ++i) {
      solver.setA(Amin + i * Astep);
      PendulumResult result = solver.solve();
      PendulumFourierResult fftresult =
          analyser.computeFftV(result, downsample, true);

      output.sweepVar[i] = solver.getA();

      if (maxfreq > 0) {
        auto max_freq_it = std::upper_bound(fftresult.freqs.begin(),
                                            fftresult.freqs.end(), maxfreq);
        size_t newsize = std::distance(fftresult.freqs.begin(), max_freq_it);
        fftresult.freqs.resize(newsize);
        fftresult.fourier.resize(newsize);
      }

      output.result[i] = fftresult;
      if (!saver) {
        output.result[i] = fftresult;
      }

      if (saver) {
        std::vector<double> PS = fftresult.computePowerSpectrum();
        for (size_t j = 0; j < fftresult.freqs.size(); ++j) {
          saver->saveRow(DataRow({solver.getA(), solver.getB(), solver.getC(),
                                  fftresult.freqs[j], PS[j]}));
        }
      }
    }
  }

  return allOutputs;
}

SweepOutput<SweepOutput<SweepOutput<PendulumFourierResult>>>
PendulumSweeper::measureFFTsRange3D(DrivenODPendulum &solver,
                                    PendulumAnalyser &analyser, double Amin,
                                    double Amax, int nA, double Bmin,
                                    double Bmax, int nB, double Cmin,
                                    double Cmax, int nC, int powerof2,
                                    DataSaver *saver, double maxfreq,
                                    int downsample) {

  SweepOutput<SweepOutput<SweepOutput<PendulumFourierResult>>> output;
  output.sweepVar.resize(nA + 1);
  output.result.resize(nA + 1);

  double Astep = (Amax - Amin) / static_cast<double>(nA);
  double Bstep = (Bmax - Bmin) / static_cast<double>(nB);
  double Cstep = (Cmax - Cmin) / static_cast<double>(nC);

  double dt = (2.0 * M_PI * analyser.getAverageNPeriods()) /
              static_cast<double>((1ULL << powerof2) * downsample);
  solver.seth(dt);

  long long totalPoints = static_cast<long long>(nA + 1) * (nB + 1) * (nC + 1);
  long long completedPoints = 0;
  int lastPercentage = -1;

  for (int i = 0; i < output.sweepVar.size(); ++i) {
    solver.setA(Amin + i * Astep);
    output.sweepVar[i] = solver.getA();

    SweepOutput<SweepOutput<PendulumFourierResult>> &outputB = output.result[i];
    outputB.sweepVar.resize(nB + 1);
    outputB.result.resize(nB + 1);

    for (int j = 0; j < outputB.sweepVar.size(); ++j) {
      solver.setB(Bmin + j * Bstep);
      outputB.sweepVar[j] = solver.getB();

      SweepOutput<PendulumFourierResult> &outputC = outputB.result[j];
      outputC.sweepVar.resize(nC + 1);
      outputC.result.resize(nC + 1);

      for (int k = 0; k < outputC.sweepVar.size(); ++k) {
        solver.setC(Cmin + k * Cstep);
        PendulumResult runResult = solver.solve();
        PendulumFourierResult fftresult =
            analyser.computeFftV(runResult, downsample, true);

        outputC.sweepVar[k] = solver.getC();

        if (maxfreq > 0) {
          auto max_freq_it = std::upper_bound(fftresult.freqs.begin(),
                                              fftresult.freqs.end(), maxfreq);
          size_t newsize = std::distance(fftresult.freqs.begin(), max_freq_it);
          fftresult.freqs.resize(newsize);
          fftresult.fourier.resize(newsize);
        }

        if (!saver) {
          outputC.result[k] = fftresult;
        }

        if (saver) {
          std::vector<double> PS = fftresult.computePowerSpectrum();
          for (size_t f = 0; f < fftresult.freqs.size(); ++f) {
            saver->saveRow(DataRow({solver.getA(), solver.getB(), solver.getC(),
                                    fftresult.freqs[f], PS[f]}));
          }
        }
        completedPoints++;
      }
    }

    int currentPercentage =
        static_cast<int>((completedPoints * 100) / totalPoints);
    if (currentPercentage > lastPercentage) {
      std::cout << "\rProgress: " << currentPercentage << "% ("
                << completedPoints << "/" << totalPoints << " points computed)"
                << std::flush;
      lastPercentage = currentPercentage;
    }
  }

  std::cout << "\n";

  return output;
}
