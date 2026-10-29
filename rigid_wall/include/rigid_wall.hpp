#pragma once

#include <cmath>
#include <vector>
#include <tuple>
#include <complex>

namespace dw {
namespace rigid {

constexpr double PI_VAL = 3.14159265358979323846;
constexpr double TWO_PI_VAL = 2.0 * PI_VAL;

/**
 * @brief Evaluates Bessel function J_n(x) using trapezoidal quadrature on the integral definition:
 * J_n(x) = (1/\pi) \int_0^\pi \cos(n\theta - x\sin\theta) d\theta
 */
inline double bessel_j(int n, double x, int num_points = 500) {
    double sum = 0.0;
    double d_theta = PI_VAL / num_points;
    for (int i = 0; i <= num_points; ++i) {
        double theta = i * d_theta;
        double val = std::cos(n * theta - x * std::sin(theta));
        double weight = (i == 0 || i == num_points) ? 0.5 : 1.0;
        sum += weight * val;
    }
    return sum * d_theta / PI_VAL;
}

/**
 * @brief Computes 1st and 2nd order perturbation boundaries for step n:
 * 1st order: h0_center = n*Omega / alpha, width = |J_n(\alpha*ha / Omega)|
 * 2nd order: h0_center = n*Omega / alpha + (alpha/Omega)*A
 * where A = 0.5 * sum_{m != -n} [ J_m^2(\tilde{ha}) / (n + m) ]
 */
inline void compute_perturbation_bounds(int n, double alpha, double omega, double ha,
                                       double& center1, double& b1_low, double& b1_high,
                                       double& center2, double& b2_low, double& b2_high,
                                       double& shift, int max_m = 35) {
    double h_tilde_a = alpha * ha / omega;
    center1 = n * omega / alpha;
    double width = std::abs(bessel_j(n, h_tilde_a));
    b1_low = center1 - width;
    b1_high = center1 + width;

    if (n == 0) {
        shift = 0.0;
    } else {
        double A = 0.0;
        for (int m = -max_m; m <= max_m; ++m) {
            if (m == -n) continue;
            double jm = bessel_j(m, h_tilde_a);
            A += 0.5 * (jm * jm) / static_cast<double>(n + m);
        }
        shift = (alpha / omega) * A;
    }

    center2 = center1 + shift;
    b2_low = center2 - width;
    b2_high = center2 + width;
}

/**
 * @brief Computes steady-state velocities for ha = 0 (Adler equation)
 */
inline void adler_solution(double h0, double alpha, double omega,
                           double& v_psi, double& v_q) {
    if (std::abs(h0) <= 1.0) {
        v_psi = 0.0;
    } else {
        double sgn = (h0 > 0) ? 1.0 : -1.0;
        v_psi = sgn * (alpha / omega) * std::sqrt(h0 * h0 - 1.0);
    }
    v_q = ((alpha * alpha + 1.0) / (2.0 * omega)) * h0 - (1.0 / (2.0 * alpha)) * v_psi;
}

/**
 * @brief Single RK4 step for rigid wall precessional ODE:
 * d\psi_1 / d\tau = A - B * \sin(\psi_1) + C * \cos(\tau)
 */
inline double rk4_step(double psi, double A, double B, double C,
                       double t, double dt) {
    double k1 = dt * (A - B * std::sin(psi) + C * std::cos(t));
    double k2 = dt * (A - B * std::sin(psi + 0.5 * k1) + C * std::cos(t + 0.5 * dt));
    double k3 = dt * (A - B * std::sin(psi + 0.5 * k2) + C * std::cos(t + 0.5 * dt));
    double k4 = dt * (A - B * std::sin(psi + k3) + C * std::cos(t + dt));
    return psi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

/**
 * @brief Angular circular distance modulo 2*pi
 */
inline double circle_distance(double a, double b) {
    double diff = std::fmod(std::abs(a - b), TWO_PI_VAL);
    return diff > PI_VAL ? TWO_PI_VAL - diff : diff;
}

} // namespace rigid
} // namespace dw
