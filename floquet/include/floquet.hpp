#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace dw {
namespace floquet {

constexpr double PI_VAL = 3.14159265358979323846;
constexpr double TWO_PI_VAL = 2.0 * PI_VAL;

struct StabilityMetrics {
    double mu;     // Largest Floquet multiplier modulus: max(|\mu_1|, |\mu_2|)
    double sigma;  // Maximum singular value of monodromy matrix
};

/**
 * @brief Circular distance modulo 2*pi
 */
inline double circle_distance(double a, double b) {
    double diff = std::fmod(std::abs(a - b), TWO_PI_VAL);
    return diff > PI_VAL ? TWO_PI_VAL - diff : diff;
}

/**
 * @brief Computes Floquet stability metrics for a single mode kappa around background cycle psi_start.
 */
inline StabilityMetrics eval_floquet_mode(double x_start, double A, double B, double C,
                                         double kappa, double omega, double alpha,
                                         double dt, int steps_per_cycle,
                                         const double* cos_t0, const double* cos_thalf) {
    double kappa2 = 0.5 * kappa * kappa;
    double theta = x_start;
    double q1 = 1.0, p1 = 0.0;
    double q2 = 0.0, p2 = 1.0;

    for (int s = 0; s < steps_per_cycle; ++s) {
        int inext = (s + 1) % steps_per_cycle;
        double c_t0 = cos_t0[s];
        double c_thalf = cos_thalf[s];
        double c_t1 = cos_t0[inext];

        double cos_th1 = std::cos(theta);
        double term1 = kappa2 + cos_th1;
        double k1_th = dt * (A - B * std::sin(theta) + C * c_t0);
        double k1_q1 = dt * (-alpha * kappa2 * q1 + term1 * p1) / omega;
        double k1_p1 = dt * (-kappa2 * q1 - alpha * term1 * p1) / omega;
        double k1_q2 = dt * (-alpha * kappa2 * q2 + term1 * p2) / omega;
        double k1_p2 = dt * (-kappa2 * q2 - alpha * term1 * p2) / omega;

        double th2 = theta + 0.5 * k1_th;
        double cos_th2 = std::cos(th2);
        double term2 = kappa2 + cos_th2;
        double q1_2 = q1 + 0.5 * k1_q1;
        double p1_2 = p1 + 0.5 * k1_p1;
        double q2_2 = q2 + 0.5 * k1_q2;
        double p2_2 = p2 + 0.5 * k1_p2;

        double k2_th = dt * (A - B * std::sin(th2) + C * c_thalf);
        double k2_q1 = dt * (-alpha * kappa2 * q1_2 + term2 * p1_2) / omega;
        double k2_p1 = dt * (-kappa2 * q1_2 - alpha * term2 * p1_2) / omega;
        double k2_q2 = dt * (-alpha * kappa2 * q2_2 + term2 * p2_2) / omega;
        double k2_p2 = dt * (-kappa2 * q2_2 - alpha * term2 * p2_2) / omega;

        double th3 = theta + 0.5 * k2_th;
        double cos_th3 = std::cos(th3);
        double term3 = kappa2 + cos_th3;
        double q1_3 = q1 + 0.5 * k2_q1;
        double p1_3 = p1 + 0.5 * k2_p1;
        double q2_3 = q2 + 0.5 * k2_q2;
        double p2_3 = p2 + 0.5 * k2_p2;

        double k3_th = dt * (A - B * std::sin(th3) + C * c_thalf);
        double k3_q1 = dt * (-alpha * kappa2 * q1_3 + term3 * p1_3) / omega;
        double k3_p1 = dt * (-kappa2 * q1_3 - alpha * term3 * p1_3) / omega;
        double k3_q2 = dt * (-alpha * kappa2 * q2_3 + term3 * p2_3) / omega;
        double k3_p2 = dt * (-kappa2 * q2_3 - alpha * term3 * p2_3) / omega;

        double th4 = theta + k3_th;
        double cos_th4 = std::cos(th4);
        double term4 = kappa2 + cos_th4;
        double q1_4 = q1 + k3_q1;
        double p1_4 = p1 + k3_p1;
        double q2_4 = q2 + k3_q2;
        double p2_4 = p2 + k3_p2;

        double k4_th = dt * (A - B * std::sin(th4) + C * c_t1);
        double k4_q1 = dt * (-alpha * kappa2 * q1_4 + term4 * p1_4) / omega;
        double k4_p1 = dt * (-kappa2 * q1_4 - alpha * term4 * p1_4) / omega;
        double k4_q2 = dt * (-alpha * kappa2 * q2_4 + term4 * p2_4) / omega;
        double k4_p2 = dt * (-kappa2 * q2_4 - alpha * term4 * p2_4) / omega;

        theta += (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th) / 6.0;
        q1 += (k1_q1 + 2.0 * k2_q1 + 2.0 * k3_q1 + k4_q1) / 6.0;
        p1 += (k1_p1 + 2.0 * k2_p1 + 2.0 * k3_p1 + k4_p1) / 6.0;
        q2 += (k1_q2 + 2.0 * k2_q2 + 2.0 * k3_q2 + k4_q2) / 6.0;
        p2 += (k1_p2 + 2.0 * k2_p2 + 2.0 * k3_p2 + k4_p2) / 6.0;
    }

    StabilityMetrics res;

    // 1. Eigenvalues of monodromy matrix
    double tr = q1 + p2;
    double det = q1 * p2 - q2 * p1;
    double disc_mu = tr * tr - 4.0 * det;

    if (disc_mu >= 0.0) {
        double root = std::sqrt(disc_mu);
        double e1 = std::abs((tr + root) * 0.5);
        double e2 = std::abs((tr - root) * 0.5);
        res.mu = std::max(e1, e2);
    } else {
        res.mu = std::sqrt(std::max(det, 0.0));
    }

    // 2. Maximum singular value (transient amplification)
    double S = q1 * q1 + p1 * p1 + q2 * q2 + p2 * p2;
    double disc_sigma = S * S - 4.0 * det * det;
    if (disc_sigma < 0.0) disc_sigma = 0.0;
    res.sigma = std::sqrt(0.5 * (S + std::sqrt(disc_sigma)));

    return res;
}

} // namespace floquet
} // namespace dw

