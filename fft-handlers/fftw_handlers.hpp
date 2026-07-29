#pragma once

#include <utility>
#include <vector>

#include "aligned_memory.hpp"
#include "fftw_precision.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dw {

// ==============================================================================
// API Traits for FFTW (Replaces Preprocessor Macros)
// ==============================================================================
template <typename RealType>
struct FftwApi;

template <>
struct FftwApi<float> {
    using Plan = fftwf_plan;
    static void plan_with_nthreads(int nthreads) { fftwf_plan_with_nthreads(nthreads); }
    static Plan plan_dft_r2c_1d(int n, float* in, fftwf_complex* out, unsigned flags) {
        return fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }
    static Plan plan_dft_c2r_1d(int n, fftwf_complex* in, float* out, unsigned flags) {
        return fftwf_plan_dft_c2r_1d(n, in, out, flags);
    }
    static Plan plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign,
                            unsigned flags) {
        return fftwf_plan_dft_1d(n, in, out, sign, flags);
    }
    static void execute_dft_r2c(Plan p, float* in, fftwf_complex* out) {
        fftwf_execute_dft_r2c(p, in, out);
    }
    static void execute_dft_c2r(Plan p, fftwf_complex* in, float* out) {
        fftwf_execute_dft_c2r(p, in, out);
    }
    static void execute_dft(Plan p, fftwf_complex* in, fftwf_complex* out) {
        fftwf_execute_dft(p, in, out);
    }
    static void destroy_plan(Plan p) { fftwf_destroy_plan(p); }
};

template <>
struct FftwApi<double> {
    using Plan = fftw_plan;
    static void plan_with_nthreads(int nthreads) { fftw_plan_with_nthreads(nthreads); }
    static Plan plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags) {
        return fftw_plan_dft_r2c_1d(n, in, out, flags);
    }
    static Plan plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags) {
        return fftw_plan_dft_c2r_1d(n, in, out, flags);
    }
    static Plan plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
        return fftw_plan_dft_1d(n, in, out, sign, flags);
    }
    static void execute_dft_r2c(Plan p, double* in, fftw_complex* out) {
        fftw_execute_dft_r2c(p, in, out);
    }
    static void execute_dft_c2r(Plan p, fftw_complex* in, double* out) {
        fftw_execute_dft_c2r(p, in, out);
    }
    static void execute_dft(Plan p, fftw_complex* in, fftw_complex* out) {
        fftw_execute_dft(p, in, out);
    }
    static void destroy_plan(Plan p) { fftw_destroy_plan(p); }
};

/// @brief Utility functions for FFT operations.
namespace FftwUtils {
/// @brief Checks if a size is optimal for FFTW (factorizable by 2, 3, 5, 7).
/// @param n The size to check.
/// @return True if optimal, false otherwise.
inline bool isOptimalSize(size_t n) {
    if (n <= 0) return false;
    const int primes[] = {2, 3, 5, 7};
    for (int p : primes) {
        while (n % p == 0) n /= p;
    }
    return n == 1;
}

/// @brief Finds the next optimal FFT size greater than or equal to min_size.
/// @param min_size The minimum acceptable size.
/// @return An optimal size for FFT computation.
inline size_t getOptimalSize(size_t min_size) {
    size_t current_size = min_size;
    while (!isOptimalSize(current_size)) {
        current_size++;
    }
    return current_size;
}
}  // namespace FftwUtils

/// @brief Base class for FFTW plan management.
template <typename Precision>
class FftwBaseHandler {
   protected:
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using FftwReal = typename Precision::FftwReal;
    using FftwComplex = typename Precision::FftwComplex;
    using Plan = typename FftwApi<Real>::Plan;

    size_t current_n_ = 0;
    bool current_measure_ = false;
    Plan fft_plan_ = nullptr;
    AlignedVector<Real> freqs_;

    /// @brief Destroys the current FFTW plan if it exists.
    virtual void destroyPlan() {
        if (fft_plan_) {
            FftwApi<Real>::destroy_plan(fft_plan_);
            fft_plan_ = nullptr;
        }
    }

   public:
    FftwBaseHandler() = default;

    virtual ~FftwBaseHandler() { cleanup(); }

    // Disable copying
    FftwBaseHandler(const FftwBaseHandler&) = delete;
    FftwBaseHandler& operator=(const FftwBaseHandler&) = delete;

    // Enable moving
    FftwBaseHandler(FftwBaseHandler&& other) noexcept
        : current_n_(other.current_n_),
          current_measure_(other.current_measure_),
          fft_plan_(other.fft_plan_),
          freqs_(std::move(other.freqs_)) {
        other.fft_plan_ = nullptr;
        other.current_n_ = 0;
    }

    FftwBaseHandler& operator=(FftwBaseHandler&& other) noexcept {
        if (this != &other) {
            cleanup();
            current_n_ = other.current_n_;
            current_measure_ = other.current_measure_;
            fft_plan_ = other.fft_plan_;
            freqs_ = std::move(other.freqs_);
            other.fft_plan_ = nullptr;
            other.current_n_ = 0;
        }
        return *this;
    }

    /// @brief Prepares the FFT plan for a given size.
    /// @param N The number of elements.
    /// @param measure If true, uses FFTW_MEASURE for optimal performance.
    virtual void prepare(size_t N, bool measure = false) = 0;

    /// @brief Resets the handler and frees resources.
    void cleanup() {
        destroyPlan();
        current_n_ = 0;
        freqs_.clear();
    }

    /// @brief Returns the current internal transformation size.
    size_t getSize() const { return current_n_; }

    /// @brief Computes and returns the frequency mapping for the current size.
    /// @param h Spatial step size.
    const AlignedVector<Real>& getFrequencies(Real h) {
        int freq_n = (current_n_ / 2) + 1;
        if (freqs_.size() != static_cast<size_t>(freq_n)) {
            freqs_.resize(freq_n);
            const Real df = static_cast<Real>(1.0) / (static_cast<Real>(current_n_) * h);
            for (int i = 0; i < freq_n; ++i) {
                freqs_[i] = i * df;
            }
        }
        return freqs_;
    }
};

/// @brief Handler for Real-to-Complex 1D FFT.
template <typename Precision>
class FftwHandlerR2C : public FftwBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using FftwReal = typename Precision::FftwReal;
    using FftwComplex = typename Precision::FftwComplex;

   public:
    void prepare(size_t N, bool measure = false) override {
        if (N == this->current_n_ && measure == this->current_measure_) return;
        this->cleanup();
        this->current_n_ = N;
        this->current_measure_ = measure;

        unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        int threads_to_use = 1;
#ifdef _OPENMP
        threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
        FftwApi<Real>::plan_with_nthreads(threads_to_use);

        // RAII Dummy arrays to prevent memory leaks during plan creation
        AlignedVector<Real> dummy_in(this->current_n_);
        AlignedVector<Complex> dummy_out(this->current_n_ / 2 + 1);

        this->fft_plan_ = FftwApi<Real>::plan_dft_r2c_1d(
            static_cast<int>(this->current_n_), reinterpret_cast<FftwReal*>(dummy_in.data()),
            reinterpret_cast<FftwComplex*>(dummy_out.data()), flags);
    }

    /// @brief Executes the forward Real-to-Complex transform.
    void do_fft(const AlignedVector<Real>& in, AlignedVector<Complex>& out, bool measure = false) {
        if (in.empty()) return;
        size_t N = in.size();
        size_t out_size = (N / 2) + 1;

        if (out.size() != out_size) out.resize(out_size);
        prepare(N, measure);

        FftwApi<Real>::execute_dft_r2c(this->fft_plan_,
                                       reinterpret_cast<FftwReal*>(const_cast<Real*>(in.data())),
                                       reinterpret_cast<FftwComplex*>(out.data()));
    }
};

/// @brief Handler for Complex-to-Real 1D Inverse FFT.
/// @note This leaves the output unnormalized. To recover magnitudes, divide by N.
template <typename Precision>
class FftwHandlerC2R : public FftwBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using FftwReal = typename Precision::FftwReal;
    using FftwComplex = typename Precision::FftwComplex;

   public:
    void prepare(size_t N, bool measure = false) override {
        if (N == this->current_n_ && measure == this->current_measure_) return;
        this->cleanup();
        this->current_n_ = N;
        this->current_measure_ = measure;

        unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        int threads_to_use = 1;
#ifdef _OPENMP
        threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
        FftwApi<Real>::plan_with_nthreads(threads_to_use);

        AlignedVector<Complex> dummy_in(this->current_n_ / 2 + 1);
        AlignedVector<Real> dummy_out(this->current_n_);

        this->fft_plan_ = FftwApi<Real>::plan_dft_c2r_1d(
            static_cast<int>(this->current_n_), reinterpret_cast<FftwComplex*>(dummy_in.data()),
            reinterpret_cast<FftwReal*>(dummy_out.data()), flags);
    }

    /// @brief Executes the backward Complex-to-Real transform.
    void do_ifft(const AlignedVector<Complex>& in, AlignedVector<Real>& out, bool measure = false) {
        if (in.empty()) return;
        size_t expected_N = (in.size() - 1) * 2;

        if (out.size() != expected_N) out.resize(expected_N);
        prepare(expected_N, measure);

        FftwApi<Real>::execute_dft_c2r(
            this->fft_plan_, reinterpret_cast<FftwComplex*>(const_cast<Complex*>(in.data())),
            reinterpret_cast<FftwReal*>(out.data()));
    }
};

/// @brief Handler for in-place Complex-to-Complex 1D FFT.
template <typename Precision>
class FftwHandlerC2CIn : public FftwBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using FftwComplex = typename Precision::FftwComplex;

   private:
    int current_sign_ = 0;

   public:
    void prepare(size_t N, bool measure = false) override { prepare_c2c(N, FFTW_FORWARD, measure); }

    /// @brief Prepares the plan considering the direction of the transform.
    void prepare_c2c(size_t N, int sign, bool measure = false) {
        if (N == this->current_n_ && measure == this->current_measure_ && sign == current_sign_)
            return;
        this->cleanup();
        this->current_n_ = N;
        this->current_measure_ = measure;
        current_sign_ = sign;

        unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        int threads_to_use = 1;
#ifdef _OPENMP
        threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
        FftwApi<Real>::plan_with_nthreads(threads_to_use);

        AlignedVector<Complex> dummy(this->current_n_);

        this->fft_plan_ = FftwApi<Real>::plan_dft_1d(
            static_cast<int>(this->current_n_), reinterpret_cast<FftwComplex*>(dummy.data()),
            reinterpret_cast<FftwComplex*>(dummy.data()), current_sign_, flags);
    }

    /// @brief Executes the forward in-place Complex-to-Complex transform.
    void do_fft(AlignedVector<Complex>& inout, bool measure = false) {
        if (inout.empty()) return;
        prepare_c2c(inout.size(), FFTW_FORWARD, measure);
        FftwApi<Real>::execute_dft(this->fft_plan_, reinterpret_cast<FftwComplex*>(inout.data()),
                                   reinterpret_cast<FftwComplex*>(inout.data()));
    }

    /// @brief Executes the backward in-place Complex-to-Complex transform.
    /// @note This leaves the output unnormalized. To recover magnitudes, divide by N.
    void do_ifft(AlignedVector<Complex>& inout, bool measure = false) {
        if (inout.empty()) return;
        prepare_c2c(inout.size(), FFTW_BACKWARD, measure);
        FftwApi<Real>::execute_dft(this->fft_plan_, reinterpret_cast<FftwComplex*>(inout.data()),
                                   reinterpret_cast<FftwComplex*>(inout.data()));
    }
};

/// @brief Handler for out-of-place Complex-to-Complex 1D FFT.
template <typename Precision>
class FftwHandlerC2COut : public FftwBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using FftwComplex = typename Precision::FftwComplex;

   private:
    int current_sign_ = 0;

   public:
    void prepare(size_t N, bool measure = false) override { prepare_c2c(N, FFTW_FORWARD, measure); }

    /// @brief Prepares the plan considering the direction of the transform.
    void prepare_c2c(size_t N, int sign, bool measure = false) {
        if (N == this->current_n_ && measure == this->current_measure_ && sign == current_sign_)
            return;
        this->cleanup();
        this->current_n_ = N;
        this->current_measure_ = measure;
        current_sign_ = sign;

        unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        int threads_to_use = 1;
#ifdef _OPENMP
        threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
        FftwApi<Real>::plan_with_nthreads(threads_to_use);

        // Need separate dummy arrays for out-of-place planning
        AlignedVector<Complex> dummy_in(this->current_n_);
        AlignedVector<Complex> dummy_out(this->current_n_);

        this->fft_plan_ = FftwApi<Real>::plan_dft_1d(
            static_cast<int>(this->current_n_), reinterpret_cast<FftwComplex*>(dummy_in.data()),
            reinterpret_cast<FftwComplex*>(dummy_out.data()), current_sign_, flags);
    }

    /// @brief Executes the forward out-of-place Complex-to-Complex transform.
    void do_fft(const AlignedVector<Complex>& in, AlignedVector<Complex>& out,
                bool measure = false) {
        if (in.empty()) return;
        size_t N = in.size();
        if (out.size() != N) out.resize(N);

        prepare_c2c(N, FFTW_FORWARD, measure);
        FftwApi<Real>::execute_dft(this->fft_plan_,
                                   reinterpret_cast<FftwComplex*>(const_cast<Complex*>(in.data())),
                                   reinterpret_cast<FftwComplex*>(out.data()));
    }

    /// @brief Executes the backward out-of-place Complex-to-Complex transform.
    void do_ifft(const AlignedVector<Complex>& in, AlignedVector<Complex>& out,
                 bool measure = false) {
        if (in.empty()) return;
        size_t N = in.size();
        if (out.size() != N) out.resize(N);

        prepare_c2c(N, FFTW_BACKWARD, measure);
        FftwApi<Real>::execute_dft(this->fft_plan_,
                                   reinterpret_cast<FftwComplex*>(const_cast<Complex*>(in.data())),
                                   reinterpret_cast<FftwComplex*>(out.data()));
    }
};

}  // namespace dw