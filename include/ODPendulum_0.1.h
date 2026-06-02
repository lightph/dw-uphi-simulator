#pragma once

#include "FftwHandler.h"
#include <complex>
#include <fftw3.h>
#include <utility>
#include <vector>

struct DrivenODPendulum {
private:
  double A, B, C, h, t0, tf, x0;
  std::vector<double> t, x, v;
  int waitNPeriods, averageNPeriods;
  std::vector<double> powerSpectrum, fftfreqs;
  std::vector<std::complex<double>> fftV;
  FftwHandler fftwHandler;

public:
  DrivenODPendulum(double AA, double BB, double CC, double hh, double t00,
                   double tff, double x00)
      : A(AA), B(BB), C(CC), h(hh), t0(t00), tf(tff), x0(x00), t({}), x({}),
        v({}), waitNPeriods(20), averageNPeriods(100) {
    solve();
  };
  DrivenODPendulum(double AA, double BB, double CC, double hh, double t00,
                   double tff, double x00, int waitN, int averageN)
      : A(AA), B(BB), C(CC), h(hh), t0(t00), tf(tff), x0(x00), t({}), x({}),
        v({}), waitNPeriods(waitN), averageNPeriods(averageN) {
    solve();
  };

  void setA(double AA) { A = AA; }
  void setB(double BB) { B = BB; }
  void setC(double CC) { C = CC; }
  void seth(double hh) { h = hh; }
  void sett0(double tt) { t0 = tt; }
  void settf(double ttf) { tf = ttf; }
  void setWaitNPeriods(int n) { waitNPeriods = n; }
  void setAverageNPeriods(int n) { averageNPeriods = n; }

  double getA() const { return A; }
  double getB() const { return B; }
  double getC() const { return C; }
  double geth() const { return h; }
  double gett0() const { return t0; }
  double gettf() const { return tf; }
  int getWaitNPeriods() const { return waitNPeriods; }
  int getAverageNPeriods() const { return averageNPeriods; }

  const std::vector<double> &getx() const { return x; }
  const std::vector<double> &gett() const { return t; }
  const std::vector<double> &getv() const { return v; }
  const std::vector<double> &getPowerSpectrum() const { return powerSpectrum; }
  const std::vector<double> &getFftFreq() const { return fftfreqs; }

  void operator()(const double t, const double &x, double &dxdt) const;

  void solve();

  const std::vector<std::complex<double>> &computeFftV(bool measure = false);
  const std::vector<double> &computePowerSpectrum(bool measure = false);
  const double getCurrentN() const;

  std::pair<size_t, size_t> stationaryIndices() const;

  const std::vector<double> &computeFftfreq();

  double getAvgVel() const;

  void getAvgVelRange(std::vector<double> &AVector,
                      std::vector<double> &avgVelVector, double Amin,
                      double Amax, int n);

  double measureShapiroStep(double Atol, double vtol, double Aleft = 0.,
                            double Aright = 10.);

  void measureShapiroStepRange(std::vector<double> &CVector,
                               std::vector<double> &stepVector, double Cmin,
                               double Cmax, int n, double Atol, double vtol,
                               double Aleft = 0., double Aright = 0.);
};
