#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

struct PendulumResult {
  std::vector<double> t, x, v;
  double A, B, C, h, t0, x0;
  size_t N;

  void setParams(double AA, double BB, double CC, double hh, double t00,
                 double x00, size_t NN) {
    A = AA;
    B = BB;
    C = CC;
    h = hh;
    t0 = t00;
    x0 = x00;
    N = NN;
  }
};

struct DrivenODPendulum {
private:
  double A, B, C, h, t0, x0;
  size_t N;

public:
  DrivenODPendulum(double AA, double BB, double CC, double hh, double t00,
                   double x00, size_t NN)
      : A(AA), B(BB), C(CC), h(hh), t0(t00), x0(x00), N(NN) {}
  void setA(double AA) { A = AA; }
  void setB(double BB) { B = BB; }
  void setC(double CC) { C = CC; }
  void seth(double hh) { h = hh; }
  void sett0(double tt) { t0 = tt; }

  double getA() const { return A; }
  double getB() const { return B; }
  double getC() const { return C; }
  double geth() const { return h; }
  double gett0() const { return t0; }

  void evaluateDerivative(const double t, const double &x, double &dxdt) const;

  PendulumResult solve() const;
};
