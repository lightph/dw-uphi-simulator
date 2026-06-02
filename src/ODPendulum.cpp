#include "ODPendulum.h"
#include <cmath>
#include <vector>

void DrivenODPendulum::evaluateDerivative(const double t, const double &x,
                                          double &dxdt) const {
  dxdt = A - B * std::sin(x) + C * std::cos(t);
}

PendulumResult DrivenODPendulum::solve() const {
  PendulumResult output;
  output.setParams(A, B, C, h, t0, x0, N);

  double tf = t0 + N * h;
  size_t npoints = std::round((tf - t0) / h);

  output.t.resize(npoints), output.x.resize(npoints), output.v.resize(npoints);

  double t_var = t0, x_var = x0, dxdt_var = 0.;
  for (size_t i = 0; i < npoints; ++i) {
    t_var = t0 + i * h;

    double xi = x_var;

    evaluateDerivative(t_var, x_var, dxdt_var);
    double dxi = dxdt_var;

    output.t[i] = t_var;
    output.x[i] = xi;
    output.v[i] = dxi;

    double k1 = dxi;

    evaluateDerivative(t_var + h / 2, xi + k1 * h / 2, dxdt_var);
    double k2 = dxdt_var;

    evaluateDerivative(t_var + h / 2, xi + k2 * h / 2, dxdt_var);
    double k3 = dxdt_var;

    evaluateDerivative(t_var + h, xi + h * k3, dxdt_var);
    double k4 = dxdt_var;

    x_var += (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
  }

  return output;
}
