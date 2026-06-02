#pragma once

#include "types.cuh"

#include <complex>
#include <vector>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cufft.h>

namespace CufftUtils
{
  inline bool isOptimalSize(size_t n)
  {
    if (n <= 0)
      return false;
    const int primes[] = {2, 3, 5, 7};
    for (int p : primes)
    {
      while (n % p == 0)
        n /= p;
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

#ifdef DOUBLE_PRECISION
  inline cufftResult cufftExecC2C_Wrap(cufftHandle plan, cuda_math::CUFFT_COMPLEX *idata, cuda_math::CUFFT_COMPLEX *odata, int direction)
  {
    return cufftExecZ2Z(plan, idata, odata, direction);
  }
  inline cufftResult cufftExecR2C_Wrap(cufftHandle plan, cuda_math::CUFFT_REAL *idata, cuda_math::CUFFT_COMPLEX *odata)
  {
    return cufftExecD2Z(plan, idata, odata);
  }
  inline cufftResult cufftExecC2R_Wrap(cufftHandle plan, cuda_math::CUFFT_COMPLEX *idata, cuda_math::CUFFT_REAL *odata)
  {
    return cufftExecZ2D(plan, idata, odata);
  }
  constexpr cufftType CUFFT_C2C_PLAN = CUFFT_Z2Z;
  constexpr cufftType CUFFT_R2C_PLAN = CUFFT_D2Z;
  constexpr cufftType CUFFT_C2R_PLAN = CUFFT_Z2D;
#else
  inline cufftResult cufftExecC2C_Wrap(cufftHandle plan, cuda_math::CUFFT_COMPLEX *idata, cuda_math::CUFFT_COMPLEX *odata, int direction)
  {
    return cufftExecC2C(plan, idata, odata, direction);
  }
  inline cufftResult cufftExecR2C_Wrap(cufftHandle plan, cuda_math::CUFFT_REAL *idata, cuda_math::CUFFT_COMPLEX *odata)
  {
    return cufftExecR2C(plan, idata, odata);
  }
  inline cufftResult cufftExecC2R_Wrap(cufftHandle plan, cuda_math::CUFFT_COMPLEX *idata, cuda_math::CUFFT_REAL *odata)
  {
    return cufftExecC2R(plan, idata, odata);
  }
  constexpr cufftType CUFFT_C2C_PLAN = CUFFT_C2C;
  constexpr cufftType CUFFT_R2C_PLAN = CUFFT_R2C;
  constexpr cufftType CUFFT_C2R_PLAN = CUFFT_C2R;
#endif
}

class CufftHandler
{
protected:
  size_t currentN = 0;
  cufftHandle fftPlan = 0;

  void createPlan(size_t N, cufftType planType);

public:
  CufftHandler() = default;
  virtual ~CufftHandler();

  CufftHandler(const CufftHandler &) = delete;
  CufftHandler &operator=(const CufftHandler &) = delete;

  virtual void prepare(size_t N) = 0;
  void cleanup();

  size_t getSize() const { return currentN; }
  void setStream(cudaStream_t stream)
  {
    if (fftPlan != 0)
    {
      cufftSetStream(fftPlan, stream);
    }
  }
};

class CufftHandlerR2C : public CufftHandler
{
public:
  void prepare(size_t N) override;
  void do_fft(const cuda_math::REAL_DEVICE_VECTOR &in,
              cuda_math::COMPLEX_DEVICE_VECTOR &out);
};

class CufftHandlerC2R : public CufftHandler
{
public:
  void prepare(size_t N) override;
  void do_ifft(const cuda_math::COMPLEX_DEVICE_VECTOR &in,
               cuda_math::REAL_DEVICE_VECTOR &out);
};

class CufftHandlerC2COut : public CufftHandler
{
public:
  void prepare(size_t N) override;
  void do_fft(const cuda_math::COMPLEX_DEVICE_VECTOR &in, cuda_math::COMPLEX_DEVICE_VECTOR &out);
  void do_ifft(const cuda_math::COMPLEX_DEVICE_VECTOR &in, cuda_math::COMPLEX_DEVICE_VECTOR &out);
};

class CufftHandlerC2CIn : public CufftHandler
{
public:
  void prepare(size_t N) override;
  void do_fft(cuda_math::COMPLEX_DEVICE_VECTOR &in);
  void do_ifft(cuda_math::COMPLEX_DEVICE_VECTOR &in);
};