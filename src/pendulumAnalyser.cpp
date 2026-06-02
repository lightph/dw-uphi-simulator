#include "pendulumAnalyser.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

std::pair<size_t, size_t> PendulumAnalyser::stationaryIndices(double h) const {
  double startTime = 2 * M_PI * waitNPeriods;
  size_t startIndex = static_cast<size_t>(std::round(startTime / h));

  double windowLength = 2 * M_PI * averageNPeriods;
  size_t endIndex =
      startIndex + static_cast<size_t>(std::round(windowLength / h));

  return {startIndex, endIndex};
}

PendulumFourierResult
PendulumAnalyser::computeFftV(const PendulumResult &previousResult,
                              int sampleEvery, bool fftwMeasure) {
  std::pair<size_t, size_t> statIndices = stationaryIndices(previousResult.h);
  if (statIndices.second >= previousResult.N) {
    throw std::runtime_error("Time to average exceeds simulation time");
  }

  PendulumFourierResult output;

  fftwHandler.execute(previousResult.v, statIndices.first, statIndices.second,
                      sampleEvery, output.fourier, fftwMeasure);
  output.N = fftwHandler.currentN;

  output.freqs = fftwHandler.getFrequencies(previousResult.h * sampleEvery);
  return output;
}

double PendulumAnalyser::getAvgVel(const PendulumResult &prevResult) {
  std::pair<size_t, size_t> statIndices = stationaryIndices(prevResult.h);
  if (statIndices.second >= prevResult.N) {
    throw std::runtime_error("Time to average exceeds simulation time");
  }
  double displacement =
      prevResult.x[statIndices.second] - prevResult.x[statIndices.first];
  double time = static_cast<double>(prevResult.t[statIndices.second] -
                                    prevResult.t[statIndices.first]);
  return displacement / time;
}
