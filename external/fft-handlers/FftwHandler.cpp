#include "FftwHandler.h"
#include <omp.h> // Required for omp_get_max_threads()

FftwBaseHandler::~FftwBaseHandler()
{
  cleanup();
}

void FftwBaseHandler::cleanup()
{
  destroyPlan();
  currentN = 0;
  freqs.clear();
}

const AlignedRealVector &FftwBaseHandler::getFrequencies(FFTW_REAL h)
{
  int freqN = (currentN / 2) + 1;

  if (freqs.size() != static_cast<size_t>(freqN))
  {
    freqs.resize(freqN);
    const FFTW_REAL df = static_cast<FFTW_REAL>(1.0) / (static_cast<FFTW_REAL>(currentN) * h);

    for (int i = 0; i < freqN; ++i)
    {
      freqs[i] = i * df;
    }
  }
  return freqs;
}

void FftwHandlerR2C::prepare(size_t N, bool measure)
{
  if (N == currentN && measure == currentMeasure)
    return;
  cleanup();
  currentN = N;
  currentMeasure = measure;

  unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

  int threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
  FFTW_PLAN_WITH_NTHREADS(threads_to_use);

  FFTW_REAL *dummy_in = FFTW_ALLOC_REAL(currentN);
  FFTW_COMPLEX_NATIVE *dummy_out = FFTW_ALLOC_COMPLEX(currentN / 2 + 1);

  fftPlan = FFTW_PLAN_DFT_R2C_1D(currentN, dummy_in, dummy_out, flags);

  FFTW_FREE(dummy_in);
  FFTW_FREE(dummy_out);
}

void FftwHandlerR2C::do_fft(const AlignedRealVector &in, AlignedComplexVector &out, bool measure)
{
  if (in.empty())
    return;

  size_t N = in.size();
  size_t out_size = (N / 2) + 1;

  if (out.size() != out_size)
  {
    out.resize(out_size);
  }

  prepare(N, measure);

  FFTW_EXECUTE_DFT_R2C(fftPlan,
                       const_cast<FFTW_REAL *>(in.data()),
                       reinterpret_cast<FFTW_COMPLEX_NATIVE *>(out.data()));
}

void FftwHandlerC2R::prepare(size_t N, bool measure)
{
  if (N == currentN && measure == currentMeasure)
    return;
  cleanup();
  currentN = N;
  currentMeasure = measure;

  unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

  int threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
  FFTW_PLAN_WITH_NTHREADS(threads_to_use);

  FFTW_COMPLEX_NATIVE *dummy_in = FFTW_ALLOC_COMPLEX(currentN / 2 + 1);
  FFTW_REAL *dummy_out = FFTW_ALLOC_REAL(currentN);

  fftPlan = FFTW_PLAN_DFT_C2R_1D(currentN, dummy_in, dummy_out, flags);

  FFTW_FREE(dummy_in);
  FFTW_FREE(dummy_out);
}

void FftwHandlerC2R::do_ifft(const AlignedComplexVector &in, AlignedRealVector &out, bool measure)
{
  if (in.empty())
    return;

  size_t expected_N = (in.size() - 1) * 2;
  if (out.size() != expected_N)
  {
    out.resize(expected_N);
  }

  prepare(expected_N, measure);

  FFTW_EXECUTE_DFT_C2R(fftPlan,
                       reinterpret_cast<FFTW_COMPLEX_NATIVE *>(const_cast<FFTW_COMPLEX_STD *>(in.data())),
                       out.data());
}

FftwHandlerC2COut::~FftwHandlerC2COut()
{
  cleanup();
}

void FftwHandlerC2COut::destroyPlan()
{
  FftwBaseHandler::destroyPlan();
  if (inversePlan)
  {
    FFTW_DESTROY_PLAN(inversePlan);
    inversePlan = nullptr;
  }
}

void FftwHandlerC2COut::prepare(size_t N, bool measure)
{
  if (N == currentN && measure == currentMeasure)
    return;
  cleanup();
  currentN = N;
  currentMeasure = measure;

  unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

  int threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
  FFTW_PLAN_WITH_NTHREADS(threads_to_use);

  FFTW_COMPLEX_NATIVE *dummy_in = FFTW_ALLOC_COMPLEX(currentN);
  FFTW_COMPLEX_NATIVE *dummy_out = FFTW_ALLOC_COMPLEX(currentN);

  fftPlan = FFTW_PLAN_DFT_1D(currentN, dummy_in, dummy_out, FFTW_FORWARD, flags);
  inversePlan = FFTW_PLAN_DFT_1D(currentN, dummy_in, dummy_out, FFTW_BACKWARD, flags);

  FFTW_FREE(dummy_in);
  FFTW_FREE(dummy_out);
}

void FftwHandlerC2COut::do_fft(const AlignedComplexVector &in, AlignedComplexVector &out, bool measure)
{
  if (in.empty())
    return;

  if (out.size() != in.size())
  {
    out.resize(in.size());
  }

  prepare(in.size(), measure);

  FFTW_EXECUTE_DFT(fftPlan,
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(const_cast<FFTW_COMPLEX_STD *>(in.data())),
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(out.data()));
}

void FftwHandlerC2COut::do_ifft(const AlignedComplexVector &in, AlignedComplexVector &out, bool measure)
{
  if (in.empty())
    return;

  if (out.size() != in.size())
  {
    out.resize(in.size());
  }

  prepare(in.size(), measure);

  FFTW_EXECUTE_DFT(inversePlan,
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(const_cast<FFTW_COMPLEX_STD *>(in.data())),
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(out.data()));
}

FftwHandlerC2CIn::~FftwHandlerC2CIn()
{
  cleanup();
}

void FftwHandlerC2CIn::destroyPlan()
{
  FftwBaseHandler::destroyPlan();
  if (inversePlan)
  {
    FFTW_DESTROY_PLAN(inversePlan);
    inversePlan = nullptr;
  }
}

void FftwHandlerC2CIn::prepare(size_t N, bool measure)
{
  if (N == currentN && measure == currentMeasure)
    return;
  cleanup();
  currentN = N;
  currentMeasure = measure;

  unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

  int threads_to_use = (N >= 16384) ? omp_get_max_threads() : 1;
  FFTW_PLAN_WITH_NTHREADS(threads_to_use);

  FFTW_COMPLEX_NATIVE *dummy_in = FFTW_ALLOC_COMPLEX(currentN);

  fftPlan = FFTW_PLAN_DFT_1D(currentN, dummy_in, dummy_in, FFTW_FORWARD, flags);
  inversePlan = FFTW_PLAN_DFT_1D(currentN, dummy_in, dummy_in, FFTW_BACKWARD, flags);

  FFTW_FREE(dummy_in);
}

void FftwHandlerC2CIn::do_fft(AlignedComplexVector &in, bool measure)
{
  if (in.empty())
    return;
  prepare(in.size(), measure);

  FFTW_EXECUTE_DFT(fftPlan,
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(in.data()),
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(in.data()));
}

void FftwHandlerC2CIn::do_ifft(AlignedComplexVector &in, bool measure)
{
  if (in.empty())
    return;
  prepare(in.size(), measure);

  FFTW_EXECUTE_DFT(inversePlan,
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(in.data()),
                   reinterpret_cast<FFTW_COMPLEX_NATIVE *>(in.data()));
}