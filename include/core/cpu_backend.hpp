#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "aligned_memory.hpp"
#include "fftw_handlers.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    template <typename StreamType = void*>
    static void synchronize(StreamType stream = nullptr) { return; }

    template <typename StreamType = void*>
    static void evaluate_nonlinear(const Complex* z, Complex* nl_out, std::size_t size,
                                   std::size_t batch, Real alpha, const Real* h0, const Real* ha,
                                   Real omega, Real t, StreamType stream = nullptr) {
        std::size_t total_elements = size * batch;
#pragma omp parallel for
        for (std::size_t j = 0; j < total_elements; ++j) {
            std::size_t b = j / size;
            Real h_val = h0[b] + ha[b] * std::cos(omega * t);
            Complex val = z[j];
            Real phi = -val.imag();
            Real sin_val = std::sin(Real(2.0) * phi);
            Real nl_real = Real(0.5) * (alpha * alpha * h_val + sin_val);
            Real nl_imag = Real(0.5) * (alpha * sin_val - alpha * h_val);
            nl_out[j] = Complex(nl_real, nl_imag);
        }
    }

    template <typename StreamType = void*>
    static void accumulate_sweep_metrics(const Complex* u, const Complex* u_prev, Real* acc_vu,
                                         Real* acc_vphi, Real* acc_rugosity, std::size_t size,
                                         std::size_t batch, Real dt, StreamType stream = nullptr) {
#pragma omp parallel for
        for (std::size_t b = 0; b < batch; ++b) {
            std::size_t offset = b * size;
            Real sum_u = 0.0, sum_u2 = 0.0, sum_du = 0.0, sum_dphi = 0.0;

            for (std::size_t i = 0; i < size; ++i) {
                Complex curr = u[offset + i];
                Complex prev = u_prev[offset + i];

                Real c_u = curr.real();
                Real du = c_u - prev.real();

                Real c_phi = -curr.imag();
                Real p_phi = -prev.imag();
                Real dphi = c_phi - p_phi;

                Real w_dphi = std::fmod(dphi + M_PI, 2.0 * M_PI);
                if (w_dphi < 0.0) w_dphi += 2.0 * M_PI;
                w_dphi -= M_PI;

                sum_u += c_u;
                sum_u2 += c_u * c_u;
                sum_du += du;
                sum_dphi += w_dphi;
            }

            Real inv_size = Real(1.0) / static_cast<Real>(size);
            Real mean_u = sum_u * inv_size;
            Real mean_u2 = sum_u2 * inv_size;

            Real rugosity = mean_u2 - (mean_u * mean_u);
            if (rugosity < Real(0.0)) rugosity = Real(0.0);

            acc_rugosity[b] += rugosity;
            acc_vu[b] += (sum_du * inv_size) / dt;
            acc_vphi[b] += (sum_dphi * inv_size) / dt;
        }
    }

    template <typename StreamType = void*>
    static void accumulate_strobo_metrics(const Complex* u, const Complex* u_strobo_prev,
                                          Real* acc_step, Real* acc_step2, std::size_t size,
                                          std::size_t batch, StreamType stream = nullptr) {
        std::size_t total = size * batch;
#pragma omp parallel for
        for (std::size_t j = 0; j < total; ++j) {
            Real du_step = u[j].real() - u_strobo_prev[j].real();
#pragma omp atomic
            acc_step[j] += du_step;
#pragma omp atomic
            acc_step2[j] += du_step * du_step;
        }
    }

    template <typename StreamType = void*>
    static void step_linear(Complex* z_hat, const Complex* nl_hat, const Complex* prop_z,
                            const Complex* prop_nl, std::size_t grid_size,
                            std::size_t total_elements, StreamType stream = nullptr) {
        Real norm_factor = Real(1.0) / static_cast<Real>(grid_size);
#pragma omp parallel for
        for (std::size_t j = 0; j < total_elements; ++j) {
            std::size_t local_idx = j % grid_size;
            z_hat[j] = (prop_z[local_idx] * z_hat[j] + prop_nl[local_idx] * nl_hat[j]) * norm_factor;
        }
    }

    template <typename StreamType = void*>
    static void copy(const Complex* src, Complex* dst, std::size_t size, StreamType stream = nullptr) {
        std::copy(src, src + size, dst);
    }

    template <typename StreamType = void*>
    static Real compute_diff_sq(const Complex* u1, const Complex* u2, std::size_t size, StreamType stream = nullptr) {
        Real sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
        for (std::size_t i = 0; i < size; ++i) {
            Real dr = u1[i].real() - u2[i].real();
            Real di = u1[i].imag() - u2[i].imag();
            sum += dr * dr + di * di;
        }
        return sum;
    }

    struct ObsTuple {
        Real sin2phi, u, var_sum;
    };

    template <typename StreamType = void*>
    static ObsTuple compute_observables(const Complex* z, std::size_t size, StreamType stream = nullptr) {
        Real s_sin = 0.0, s_u = 0.0;

#pragma omp parallel for reduction(+ : s_sin, s_u)
        for (std::size_t i = 0; i < size; ++i) {
            Real u = z[i].real();
            Real phi = -z[i].imag();
            s_sin += std::sin(Real(2.0) * phi);
            s_u += u;
        }

        Real mean_u = s_u / static_cast<Real>(size);
        Real var_sum = 0.0;

#pragma omp parallel for reduction(+ : var_sum)
        for (std::size_t i = 0; i < size; ++i) {
            Real diff = z[i].real() - mean_u;
            var_sum += diff * diff;
        }

        return {s_sin, s_u, var_sum};
    }

    template <typename StreamType = void*>
    static void accumulate_power_spectrum(const Complex* u_hat, Real* ps_accum, std::size_t size, StreamType stream = nullptr) {
#pragma omp parallel for
        for (std::size_t i = 0; i < size; ++i) {
            std::size_t i_neg = (size - i) % size;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            ps_accum[i] += real_u * real_u + imag_u * imag_u;
        }
    }

    template <typename StreamType = void*>
    static Real compute_spectral_entropy(const Complex* u_hat, std::size_t size, StreamType stream = nullptr) {
        Real sum_S = 0.0;

#pragma omp parallel for reduction(+ : sum_S)
        for (std::size_t i = 1; i < size; ++i) {
            std::size_t i_neg = (size - i) % size;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            
            Real k = static_cast<Real>((i <= size / 2) ? i : size - i);
            sum_S += (real_u * real_u + imag_u * imag_u) * (k * k);
        }

        if (sum_S <= 0.0) return 0.0;

        Real entropy = 0.0;
#pragma omp parallel for reduction(+ : entropy)
        for (std::size_t i = 1; i < size; ++i) {
            std::size_t i_neg = (size - i) % size;
            Real A = u_hat[i].real();
            Real B = u_hat[i].imag();
            Real C = u_hat[i_neg].real();
            Real D = u_hat[i_neg].imag();

            Real real_u = Real(0.5) * (A + C);
            Real imag_u = Real(0.5) * (B - D);
            
            Real k = static_cast<Real>((i <= size / 2) ? i : size - i);
            Real S = (real_u * real_u + imag_u * imag_u) * (k * k);

            if (S > 0.0) {
                Real p = S / sum_S;
                entropy -= p * std::log(p);
            }
        }
        return entropy / std::log(static_cast<Real>(size - 1));
    }

    template <typename StreamType = void*>
    static void fill_zero(Real* vec, std::size_t size, StreamType stream = nullptr) {
        std::fill(vec, vec + size, Real(0.0));
    }

    template <typename StreamType = void*>
    static std::vector<Real> download_array(const Real* vec, std::size_t size, StreamType stream = nullptr) {
        return std::vector<Real>(vec, vec + size);
    }

    template <typename StreamType = void*>
    static std::vector<Complex> download_array(const Complex* vec, std::size_t size, StreamType stream = nullptr) {
        return std::vector<Complex>(vec, vec + size);
    }

    template <typename StreamType = void*>
    static void accumulate_height_histogram(const Complex* z, std::size_t z_size, Real mean_u, Real sigma_u,
                                            unsigned long long* hist, std::size_t hist_size, Real min_val,
                                            Real max_val, StreamType stream = nullptr) {
        Real bin_width = (max_val - min_val) / static_cast<Real>(hist_size);

#pragma omp parallel
        {
            std::vector<unsigned long long> local_hist(hist_size, 0);
#pragma omp for
            for (std::size_t j = 0; j < z_size; ++j) {
                Real norm_u = (z[j].real() - mean_u) / sigma_u;
                if (norm_u >= min_val && norm_u < max_val) {
                    int bin = static_cast<int>((norm_u - min_val) / bin_width);
                    if (bin >= 0 && bin < hist_size) {
                        local_hist[bin]++;
                    }
                }
            }
#pragma omp critical
            for (std::size_t b = 0; b < hist_size; ++b) {
                hist[b] += local_hist[b];
            }
        }
    }

    template <typename StreamType = void*>
    static void fill_zero_ull(unsigned long long* vec, std::size_t size, StreamType stream = nullptr) {
        std::fill(vec, vec + size, 0ULL);
    }

    template <typename StreamType = void*>
    static std::vector<unsigned long long> download_array_ull(const unsigned long long* vec, std::size_t size, StreamType stream = nullptr) {
        return std::vector<unsigned long long>(vec, vec + size);
    }

    template <typename StreamType = void*>
    static void record_observables_async(const Complex* z, std::size_t size, Real* d_mean_u,
                                         Real* d_mean_u2, Real* d_mean_sin2phi,
                                         std::size_t step_idx, StreamType stream = nullptr) {
        Real s_u = 0.0, s_u2 = 0.0, s_sin = 0.0;

#pragma omp parallel for reduction(+ : s_u, s_u2, s_sin)
        for (std::size_t i = 0; i < size; ++i) {
            Real u = z[i].real();
            Real phi = -z[i].imag();
            s_u += u;
            s_u2 += u * u;
            s_sin += std::sin(Real(2.0) * phi);
        }

        Real N = static_cast<Real>(size);
        d_mean_u[step_idx] = s_u / N;
        d_mean_u2[step_idx] = s_u2 / N;
        d_mean_sin2phi[step_idx] = s_sin / N;
    }

    static unsigned long long binomial_coeff(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = n - k;
        unsigned long long res = 1;
        for (int i = 1; i <= k; ++i) {
            res = res * (n - i + 1) / i;
        }
        return res;
    }

    template <typename StreamType = void*>
    static std::vector<Real> compute_central_moments(const Complex* z, std::size_t size, int n_max, StreamType stream = nullptr) {
        ObsTuple obs = compute_observables(z, size, stream);
        Real mean_u = obs.u / static_cast<Real>(size);

        std::vector<Real> moments(n_max + 1, Real(0.0));
        moments[0] = Real(1.0);
        if (n_max >= 1) moments[1] = Real(0.0);

        if (n_max >= 2) {
            std::vector<Real> sums(n_max + 1, Real(0.0));
#pragma omp parallel
            {
                std::vector<Real> local_sums(n_max + 1, Real(0.0));
#pragma omp for
                for (std::size_t i = 0; i < size; ++i) {
                    Real diff = z[i].real() - mean_u;
                    Real current = diff * diff;
                    for (int k = 2; k <= n_max; ++k) {
                        local_sums[k] += current;
                        current *= diff;
                    }
                }
#pragma omp critical
                for (int k = 2; k <= n_max; ++k) {
                    sums[k] += local_sums[k];
                }
            }
            Real invN = Real(1.0) / static_cast<Real>(size);
            for (int k = 2; k <= n_max; ++k) {
                moments[k] = sums[k] * invN;
            }
        }
        return moments;
    }

    static std::vector<Real> compute_cumulants(Real mean_u, const std::vector<Real>& central_moments) {
        int n_max = static_cast<int>(central_moments.size()) - 1;
        std::vector<Real> cumulants(n_max + 1, Real(0.0));

        if (n_max >= 1) cumulants[1] = mean_u;

        for (int n = 2; n <= n_max; ++n) {
            Real sum = Real(0.0);
            for (int k = 1; k <= n - 1; ++k) {
                Real kc_k = (k == 1) ? Real(0.0) : cumulants[k];
                sum += static_cast<Real>(binomial_coeff(n - 1, k - 1)) * kc_k * central_moments[n - k];
            }
            cumulants[n] = central_moments[n] - sum;
        }

        return cumulants;
    }
};

}  // namespace dw
