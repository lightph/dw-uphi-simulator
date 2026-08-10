#pragma once

#include <cmath>
#include <vector>

#include "aligned_memory.hpp"
#include "fftw_handlers.hpp"

namespace dw {

template <typename Precision>
struct CpuBackend {
    using PrecisionType = Precision;

    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;

    using R2C = FftwHandlerR2C<Precision>;
    using C2R = FftwHandlerC2R<Precision>;
    using C2CIn = FftwHandlerC2CIn<Precision>;
    using C2COut = FftwHandlerC2COut<Precision>;

    template <typename T>
    using Vector = AlignedVector<T>;

    using ComplexVector = AlignedVector<Complex>;

    static void compute_difference(const typename Precision::Complex* u,
                                   const typename Precision::Complex* u_prev,
                                   typename Precision::Complex* delta, std::size_t size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < size; ++i) {
            delta[i] = u[i] - u_prev[i];
        }
    }
    static void synchronize() { return; }
    static void evaluate_nonlinear(const ComplexVector& z, ComplexVector& nl_out, Real alpha,
                                   Real h_val) {
        Complex prefactor(alpha / Real(2.0), Real(-1.0) / Real(2.0));

#pragma omp parallel for
        for (std::size_t j = 0; j < z.size(); ++j) {
            Real phi = -z[j].imag();
            Complex bracket(alpha * h_val, std::sin(Real(2.0) * phi));
            nl_out[j] = prefactor * bracket;
        }
    }

    static void step_linear(ComplexVector& z_hat, const ComplexVector& nl_hat,
                            const Complex* prop_z, const Complex* prop_nl) {
        Real norm_factor = Real(1.0) / static_cast<Real>(z_hat.size());

#pragma omp parallel for
        for (std::size_t j = 0; j < z_hat.size(); ++j) {
            z_hat[j] = (prop_z[j] * z_hat[j] + prop_nl[j] * nl_hat[j]) * norm_factor;
        }
    }

    static void copy(const ComplexVector& src, ComplexVector& dst) { dst = src; }

    static Real compute_diff_sq(const ComplexVector& u1, const ComplexVector& u2) {
        Real sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
        for (std::size_t i = 0; i < u1.size(); ++i) {
            Real dr = u1[i].real() - u2[i].real();
            Real di = u1[i].imag() - u2[i].imag();
            sum += dr * dr + di * di;
        }
        return sum;
    }

    struct ObsTuple {
        Real sin2phi, u, u2;
    };

    static ObsTuple compute_observables(const ComplexVector& z) {
        Real s_sin = 0.0, s_u = 0.0, s_u2 = 0.0;

#pragma omp parallel for reduction(+ : s_sin, s_u, s_u2)
        for (std::size_t i = 0; i < z.size(); ++i) {
            Real u = z[i].real();
            Real phi = -z[i].imag();
            s_sin += std::sin(Real(2.0) * phi);
            s_u += u;
            s_u2 += u * u;
        }
        return {s_sin, s_u, s_u2};
    }

    static void accumulate_power_spectrum(const ComplexVector& u_hat,
                                          AlignedVector<Real>& ps_accum) {
        std::size_t N = u_hat.size();
#pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t i_neg = (N - i) % N;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            ps_accum[i] += real_u * real_u + imag_u * imag_u;
        }
    }
    static Real compute_spectral_entropy(const ComplexVector& u_hat) {
        std::size_t N = u_hat.size();
        Real sum_S = 0.0;

#pragma omp parallel for reduction(+ : sum_S)
        for (std::size_t i = 1; i < N; ++i) {
            std::size_t i_neg = (N - i) % N;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            sum_S += real_u * real_u + imag_u * imag_u;
        }

        if (sum_S <= 0.0) return 0.0;

        Real entropy = 0.0;
#pragma omp parallel for reduction(+ : entropy)
        for (std::size_t i = 1; i < N; ++i) {
            std::size_t i_neg = (N - i) % N;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            Real S = real_u * real_u + imag_u * imag_u;

            if (S > 0.0) {
                Real p = S / sum_S;
                entropy -= p * std::log(p);
            }
        }
        // Normalize by max theoretical entropy of N-1 states
        return entropy / std::log(static_cast<Real>(N - 1));
    }

    static void fill_zero(AlignedVector<Real>& vec) {
        std::fill(vec.begin(), vec.end(), Real(0.0));
    }

    static std::vector<Real> download_array(const AlignedVector<Real>& vec) {
        return std::vector<Real>(vec.begin(), vec.end());
    }
    static std::vector<Complex> download_array(const ComplexVector& vec) {
        return std::vector<Complex>(vec.begin(), vec.end());
    }
    static void accumulate_height_histogram(const ComplexVector& z, Real mean_u, Real sigma_u,
                                            AlignedVector<unsigned long long>& hist, Real min_val,
                                            Real max_val) {
        std::size_t num_bins = hist.size();
        Real bin_width = (max_val - min_val) / static_cast<Real>(num_bins);

#pragma omp parallel
        {
            std::vector<unsigned long long> local_hist(num_bins, 0);
#pragma omp for
            for (std::size_t j = 0; j < z.size(); ++j) {
                Real norm_u = (z[j].real() - mean_u) / sigma_u;
                if (norm_u >= min_val && norm_u < max_val) {
                    int bin = static_cast<int>((norm_u - min_val) / bin_width);
                    if (bin >= 0 && bin < num_bins) {
                        local_hist[bin]++;
                    }
                }
            }
#pragma omp critical
            for (std::size_t b = 0; b < num_bins; ++b) {
                hist[b] += local_hist[b];
            }
        }
    }

    static void fill_zero_ull(AlignedVector<unsigned long long>& vec) {
        std::fill(vec.begin(), vec.end(), 0ULL);
    }

    static std::vector<unsigned long long> download_array_ull(
        const AlignedVector<unsigned long long>& vec) {
        return std::vector<unsigned long long>(vec.begin(), vec.end());
    }

    static void record_observables_async(const ComplexVector& z, AlignedVector<Real>& d_mean_u,
                                         AlignedVector<Real>& d_mean_u2,
                                         AlignedVector<Real>& d_mean_sin2phi,
                                         std::size_t step_idx) {
        Real s_u = 0.0, s_u2 = 0.0, s_sin = 0.0;

#pragma omp parallel for reduction(+ : s_u, s_u2, s_sin)
        for (std::size_t i = 0; i < z.size(); ++i) {
            Real u = z[i].real();
            Real phi = -z[i].imag();
            s_u += u;
            s_u2 += u * u;
            s_sin += std::sin(Real(2.0) * phi);
        }

        Real N = static_cast<Real>(z.size());
        d_mean_u[step_idx] = s_u / N;
        d_mean_u2[step_idx] = s_u2 / N;
        d_mean_sin2phi[step_idx] = s_sin / N;
    }
};

}  // namespace dw