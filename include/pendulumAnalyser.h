#include "FftwHandler.h"
#include "ODPendulum.h"
#include "light.h"
#include <cstddef>
#include <fftw3.h>
#include <utility>

struct PendulumFourierResult {
  std::vector<double> freqs;
  std::vector<std::complex<double>> fourier;
  size_t N;
  std::vector<double> computePowerSpectrum() {
    return light::getPowerSpectrum(fourier, N);
  }
};

class PendulumAnalyser {
private:
  int waitNPeriods, averageNPeriods;
  FftwHandler fftwHandler;

  std::pair<size_t, size_t> stationaryIndices(double h) const;

public:
  PendulumAnalyser(int waitNPeriodss, int averageNPeriodss)
      : waitNPeriods(waitNPeriodss), averageNPeriods(averageNPeriodss) {}

  void setWaitNPeriods(int n) { waitNPeriods = n; }
  void setAverageNPeriods(int n) { averageNPeriods = n; }
  int getWaitNPeriods() const { return waitNPeriods; }
  int getAverageNPeriods() const { return averageNPeriods; }

  PendulumFourierResult computeFftV(const PendulumResult &prevResult,
                                    int sampleEvery = 1,
                                    bool fftwMeasure = false);

  double getAvgVel(const PendulumResult &prevResult);
};
