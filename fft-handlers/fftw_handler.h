#pragma once

#include "types.h"
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dw {

// ==============================================================================
// Precision Macros for FFTW
// ==============================================================================
#ifdef DOUBLE_PRECISION
#define DW_FFTW_PLAN_DFT_R2C_1D fftw_plan_dft_r2c_1d
#define DW_FFTW_PLAN_DFT_C2R_1D fftw_plan_dft_c2r_1d
#define DW_FFTW_PLAN_DFT_1D fftw_plan_dft_1d
#define DW_FFTW_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#define DW_FFTW_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#define DW_FFTW_EXECUTE_DFT fftw_execute_dft
#define DW_FFTW_PLAN_WITH_NTHREADS fftw_plan_with_nthreads
#define DW_FFTW_DESTROY_PLAN fftw_destroy_plan
#else
#define DW_FFTW_PLAN_DFT_R2C_1D fftwf_plan_dft_r2c_1d
#define DW_FFTW_PLAN_DFT_C2R_1D fftwf_plan_dft_c2r_1d
#define DW_FFTW_PLAN_DFT_1D fftwf_plan_dft_1d
#define DW_FFTW_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
#define DW_FFTW_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#define DW_FFTW_EXECUTE_DFT fftwf_execute_dft
#define DW_FFTW_PLAN_WITH_NTHREADS fftwf_plan_with_nthreads
#define DW_FFTW_DESTROY_PLAN fftwf_destroy_plan
#endif

/// @brief Utility functions for FFT operations.
namespace FftwUtils {
/// @brief Checks if a size is optimal for FFTW (factorizable by 2, 3, 5, 7).
/// @param n The size to check.
/// @return True if optimal, false otherwise.
inline bool isOptimalSize(size_t n) {
  if (n <= 0)
    return false;
  const int primes[] = {2, 3, 5, 7};
  for (int p : primes) {
    while (n % p == 0)
      n /= p;
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
} // namespace FftwUtils

/// @brief Base class for FFTW plan management.
class FftwBaseHandler {
protected:
  size_t currentN = 0;
  bool currentMeasure = false;
  FftwPlan fftPlan = nullptr;
  AlignedVector<Real> freqs;

  /// @brief Destroys the current FFTW plan if it exists.
  inline virtual void destroyPlan() {
    if (fftPlan) {
      DW_FFTW_DESTROY_PLAN(fftPlan);
      fftPlan = nullptr;
    }
  }

public:
  FftwBaseHandler() = default;

  inline virtual ~FftwBaseHandler() { cleanup(); }

  FftwBaseHandler(const FftwBaseHandler &) = delete;
  FftwBaseHandler &operator=(const FftwBaseHandler &) = delete;

  /// @brief Prepares the FFT plan for a given size.
  /// @param N The number of elements.
  /// @param measure If true, uses FFTW_MEASURE for optimal performance.
  virtual void prepare(size_t N, bool measure = false) = 0;

  /// @brief Resets the handler and frees resources.
  inline void cleanup() {
    destroyPlan();
    currentN = 0;
    freqs.clear();
  }

  /// @brief Returns the current internal transformation size.
  inline size_t getSize() const { return currentN; }

  /// @brief Computes and returns the frequency mapping for the current size.
  /// @param h Spatial step size.
  inline const AlignedVector<Real> &getFrequencies(Real h) {
    int freqN = (currentN / 2) + 1;
    if (freqs.size() != static_cast<size_t>(freqN)) {
      freqs.resize(freqN);
      const Real df =
          static_cast<Real>(1.0) / (static_cast<Real>(currentN) * h);
      for (int i = 0; i < freqN; ++i) {
        freqs[i] = i * df;
      }
    }
    return freqs;
  }
};

/// @brief Handler for Real-to-Complex 1D FFT.
class FftwHandlerR2C : public FftwBaseHandler {
public:
  inline void prepare(size_t N, bool measure = false) override {
    if (N == currentN && measure == currentMeasure)
      return;
    cleanup();
    currentN = N;
    currentMeasure = measure;

    unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

    int threads_to_use = 1;
#ifdef _OPENMP
    threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
    DW_FFTW_PLAN_WITH_NTHREADS(threads_to_use);

    // RAII Dummy arrays to prevent memory leaks during plan creation
    AlignedVector<Real> dummy_in(currentN);
    AlignedVector<Complex> dummy_out(currentN / 2 + 1);

    fftPlan = DW_FFTW_PLAN_DFT_R2C_1D(
        currentN, reinterpret_cast<FftwReal *>(dummy_in.data()),
        reinterpret_cast<FftwComplex *>(dummy_out.data()), flags);
  }

  /// @brief Executes the forward Real-to-Complex transform.
  inline void do_fft(const AlignedVector<Real> &in, AlignedVector<Complex> &out,
                     bool measure = false) {
    if (in.empty())
      return;
    size_t N = in.size();
    size_t out_size = (N / 2) + 1;

    if (out.size() != out_size)
      out.resize(out_size);
    prepare(N, measure);

    DW_FFTW_EXECUTE_DFT_R2C(
        fftPlan, reinterpret_cast<FftwReal *>(const_cast<Real *>(in.data())),
        reinterpret_cast<FftwComplex *>(out.data()));
  }
};

/// @brief Handler for Complex-to-Real 1D Inverse FFT.
class FftwHandlerC2R : public FftwBaseHandler {
public:
  inline void prepare(size_t N, bool measure = false) override {
    if (N == currentN && measure == currentMeasure)
      return;
    cleanup();
    currentN = N;
    currentMeasure = measure;

    unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

    int threads_to_use = 1;
#ifdef _OPENMP
    threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
#endif
    DW_FFTW_PLAN_WITH_NTHREADS(threads_to_use);

    AlignedVector<Complex> dummy_in(currentN / 2 + 1);
    AlignedVector<Real> dummy_out(currentN);

    fftPlan = DW_FFTW_PLAN_DFT_C2R_1D(
        currentN, reinterpret_cast<FftwComplex *>(dummy_in.data()),
        reinterpret_cast<FftwReal *>(dummy_out.data()), flags);
  }

  /// @brief Executes the backward Complex-to-Real transform.
  inline void do_ifft(const AlignedVector<Complex> &in,
                      AlignedVector<Real> &out, bool measure = false) {
    if (in.empty())
      return;
    size_t expected_N = (in.size() - 1) * 2;

    if (out.size() != expected_N)
      out.resize(expected_N);
    prepare(expected_N, measure);

    DW_FFTW_EXECUTE_DFT_C2R(
        fftPlan,
        reinterpret_cast<FftwComplex *>(const_cast<Complex *>(in.data())),
        reinterpret_cast<FftwReal *>(out.data()));
  }
};

} // namespace dw
