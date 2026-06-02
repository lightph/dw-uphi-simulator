#include "ODPendulum.h"
#include "light.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fftw3.h>
#include <utility>
#include <vector>

void DrivenODPendulum::operator()(const double t, const double &x,
                                  double &dxdt) const {
  dxdt = A + B * std::sin(x) + C * std::sin(t);
}

void DrivenODPendulum::solve() {
  tf = std::max(tf, t0 + (waitNPeriods + averageNPeriods + 1) * 2 * M_PI);

  int npoints = std::ceil((tf - t0) / h - 1.e-9) + 1;
  t.resize(npoints), x.resize(npoints), v.resize(npoints);

  double t_var = t0, x_var = x0, dxdt_var = 0.;
  (*this)(t_var, x_var, dxdt_var);
  for (int i = 0; i < npoints; ++i) {
    t_var = (i != npoints - 1) ? t0 + i * h : tf;

    double xi = x_var;

    (*this)(t_var, x_var, dxdt_var);
    double dxi = dxdt_var;

    t[i] = t_var;
    x[i] = xi;
    v[i] = dxi;

    double k1 = dxi;

    (*this)(t_var + h / 2, xi + k1 * h / 2, dxdt_var);
    double k2 = dxdt_var;

    (*this)(t_var + h / 2, xi + k2 * h / 2, dxdt_var);
    double k3 = dxdt_var;

    (*this)(t_var + h, xi + h * k3, dxdt_var);
    double k4 = dxdt_var;

    x_var += (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
  }
  return;
}

std::pair<size_t, size_t> DrivenODPendulum::stationaryIndices() const {
  double waitTime = 2.0 * M_PI * waitNPeriods;
  double totalTime = 2.0 * M_PI * (waitNPeriods + averageNPeriods);
  size_t offset1 = static_cast<size_t>(std::round(waitTime / h));
  size_t offset2 = static_cast<size_t>(std::round(totalTime / h));
  return std::make_pair(offset1, offset2);
}

const std::vector<std::complex<double>> &
DrivenODPendulum::computeFftV(bool measure) {
  auto cleanIndices = stationaryIndices();
  fftwHandler.execute(v, cleanIndices.first, cleanIndices.second, fftV,
                      measure);
  return fftV;
}

const std::vector<double> &
DrivenODPendulum::computePowerSpectrum(bool measure) {
  auto cleanIndices = stationaryIndices();
  computeFftV(measure);
  powerSpectrum = light::getPowerSpectrum(fftV, fftwHandler.currentN);
  return powerSpectrum;
}

const std::vector<double> &DrivenODPendulum::computeFftfreq() {
  fftfreqs = fftwHandler.getFrequencies(h);
  return fftfreqs;
}

double DrivenODPendulum::getAvgVel() const {
  auto cleanIndices = stationaryIndices();
  double displacement = x[cleanIndices.second - 1] - x[cleanIndices.first];
  double time =
      static_cast<double>(t[cleanIndices.second - 1] - t[cleanIndices.first]);
  return displacement / time;
}

void DrivenODPendulum::getAvgVelRange(std::vector<double> &AVector,
                                      std::vector<double> &avgVelVector,
                                      double Amin, double Amax, int n) {
  const double previousA = A;
  AVector.resize(n + 1);
  avgVelVector.resize(n + 1);

  double Astep = (Amax - Amin) / static_cast<double>(n);

  for (int i = 0; i < AVector.size(); ++i) {
    setA(Amin + i * Astep);
    AVector[i] = A;
    avgVelVector[i] = getAvgVel();
  }
  setA(previousA);
  return;
}

double DrivenODPendulum::measureShapiroStep(double Atol, double vtol,
                                            double Aleft, double Aright) {
  const double previousA = A;

  double Amid, stepBegin, stepEnd;

  double left = Aleft, right = Aright;
  while (right - left >= Atol) {
    Amid = 0.5 * (left + right);
    setA(Amid);
    solve();
    if (getAvgVel() >= 1. - vtol) {
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
    setA(Amid);
    solve();
    if (getAvgVel() > 1. + vtol) {
      right = Amid;
      continue;
    }
    left = Amid;
  }
  stepEnd = Amid;
  setA(previousA);
  return (stepEnd - stepBegin);
}

void DrivenODPendulum::measureShapiroStepRange(std::vector<double> &CVector,
                                               std::vector<double> &stepVector,
                                               double Cmin, double Cmax, int n,
                                               double Atol, double vtol,
                                               double Aleft, double Aright) {
  const double previousC = C;

  CVector.resize(n + 1);
  stepVector.resize(n + 1);

  double Cstep = (Cmax - Cmin) / static_cast<double>(n);

  for (int i = 0; i < CVector.size(); ++i) {
    setC(Cmin + i * Cstep);
    double step = measureShapiroStep(Atol, vtol);
    CVector[i] = C;
    stepVector[i] = step;
  }
  setC(previousC);
  return;
}

const double DrivenODPendulum::getCurrentN() const {
  return fftwHandler.currentN;
}
