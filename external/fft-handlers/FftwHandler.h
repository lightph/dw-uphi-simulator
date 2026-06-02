#pragma once

#include <complex>
#include <fftw3.h>
#include <vector>
#include <new>
#include <cstddef>

#ifdef DOUBLE_PRECISION
using FFTW_REAL = double;
using FFTW_COMPLEX_STD = std::complex<double>;
using FFTW_COMPLEX_NATIVE = fftw_complex;
using FFTW_PLAN = fftw_plan;

#define FFTW_MALLOC fftw_malloc
#define FFTW_FREE fftw_free
#define FFTW_ALLOC_REAL fftw_alloc_real
#define FFTW_ALLOC_COMPLEX fftw_alloc_complex
#define FFTW_DESTROY_PLAN fftw_destroy_plan
#define FFTW_PLAN_DFT_R2C_1D fftw_plan_dft_r2c_1d
#define FFTW_PLAN_DFT_C2R_1D fftw_plan_dft_c2r_1d
#define FFTW_PLAN_DFT_1D fftw_plan_dft_1d
#define FFTW_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#define FFTW_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#define FFTW_EXECUTE_DFT fftw_execute_dft
#define FFTW_PLAN_WITH_NTHREADS fftw_plan_with_nthreads
#else
using FFTW_REAL = float;
using FFTW_COMPLEX_STD = std::complex<float>;
using FFTW_COMPLEX_NATIVE = fftwf_complex;
using FFTW_PLAN = fftwf_plan;

#define FFTW_MALLOC fftwf_malloc
#define FFTW_FREE fftwf_free
#define FFTW_ALLOC_REAL fftwf_alloc_real
#define FFTW_ALLOC_COMPLEX fftwf_alloc_complex
#define FFTW_DESTROY_PLAN fftwf_destroy_plan
#define FFTW_PLAN_DFT_R2C_1D fftwf_plan_dft_r2c_1d
#define FFTW_PLAN_DFT_C2R_1D fftwf_plan_dft_c2r_1d
#define FFTW_PLAN_DFT_1D fftwf_plan_dft_1d
#define FFTW_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
#define FFTW_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#define FFTW_EXECUTE_DFT fftwf_execute_dft
#define FFTW_PLAN_WITH_NTHREADS fftwf_plan_with_nthreads
#endif

template <class T>
struct FftwAllocator
{
  using value_type = T;

  FftwAllocator() = default;
  template <class U>
  constexpr FftwAllocator(const FftwAllocator<U> &) noexcept {}

  T *allocate(std::size_t n)
  {
    if (auto p = static_cast<T *>(FFTW_MALLOC(n * sizeof(T))))
    {
      return p;
    }
    throw std::bad_alloc();
  }

  void deallocate(T *p, std::size_t) noexcept
  {
    FFTW_FREE(p);
  }
};

using AlignedRealVector = std::vector<FFTW_REAL, FftwAllocator<FFTW_REAL>>;
using AlignedComplexVector = std::vector<FFTW_COMPLEX_STD, FftwAllocator<FFTW_COMPLEX_STD>>;

namespace FftwUtils
{
  inline bool isOptimalSize(size_t n)
  {
    if (n <= 0)
      return false;

    const int primes[] = {2, 3, 5, 7};
    for (int p : primes)
    {
      while (n % p == 0)
      {
        n /= p;
      }
    }
    return n == 1;
  }

  inline size_t getOptimalSize(size_t min_size)
  {
    size_t current_size = min_size;
    while (!isOptimalSize(current_size))
    {
      current_size++;
    }
    return current_size;
  }
}

class FftwBaseHandler
{
protected:
  size_t currentN = 0;
  bool currentMeasure = false;
  FFTW_PLAN fftPlan = nullptr;
  AlignedRealVector freqs;

  virtual void destroyPlan()
  {
    if (fftPlan)
    {
      FFTW_DESTROY_PLAN(fftPlan);
      fftPlan = nullptr;
    }
  }

public:
  FftwBaseHandler() = default;
  virtual ~FftwBaseHandler();

  FftwBaseHandler(const FftwBaseHandler &) = delete;
  FftwBaseHandler &operator=(const FftwBaseHandler &) = delete;

  virtual void prepare(size_t N, bool measure = false) = 0;
  void cleanup();

  size_t getSize() const { return currentN; }

  const AlignedRealVector &getFrequencies(FFTW_REAL h);
};

class FftwHandlerR2C : public FftwBaseHandler
{
public:
  void prepare(size_t N, bool measure = false) override;
  void do_fft(const AlignedRealVector &in, AlignedComplexVector &out, bool measure = false);
};

class FftwHandlerC2R : public FftwBaseHandler
{
public:
  void prepare(size_t N, bool measure = false) override;
  void do_ifft(const AlignedComplexVector &in, AlignedRealVector &out, bool measure = false);
};

class FftwHandlerC2COut : public FftwBaseHandler
{
private:
  FFTW_PLAN inversePlan = nullptr;

protected:
  void destroyPlan() override;

public:
  ~FftwHandlerC2COut() override;
  void prepare(size_t N, bool measure = false) override;
  void do_fft(const AlignedComplexVector &in, AlignedComplexVector &out, bool measure = false);
  void do_ifft(const AlignedComplexVector &in, AlignedComplexVector &out, bool measure = false);
};

class FftwHandlerC2CIn : public FftwBaseHandler
{
private:
  FFTW_PLAN inversePlan = nullptr;

protected:
  void destroyPlan() override;

public:
  ~FftwHandlerC2CIn() override;
  void prepare(size_t N, bool measure = false) override;
  void do_fft(AlignedComplexVector &in, bool measure = false);
  void do_ifft(AlignedComplexVector &in, bool measure = false);
};