#include "cufftHandler.cuh"

CufftHandler::~CufftHandler()
{
  cleanup();
}

void CufftHandler::cleanup()
{
  if (fftPlan != 0)
  {
    cufftDestroy(fftPlan);
    fftPlan = 0;
  }
  currentN = 0;
}

void CufftHandler::createPlan(size_t N, cufftType planType)
{
  if (N == currentN)
    return;

  cleanup();
  currentN = N;

  CUFFT_CHECK(cufftPlan1d(&fftPlan, currentN, planType, 1));
}

// --- CufftHandlerR2C ---

void CufftHandlerR2C::prepare(size_t N)
{
  createPlan(N, CufftUtils::CUFFT_R2C_PLAN);
}

void CufftHandlerR2C::do_fft(const cuda_math::REAL_DEVICE_VECTOR &in,
                             cuda_math::COMPLEX_DEVICE_VECTOR &out)
{
  if (in.empty())
    return;

  size_t N = in.size();
  size_t out_size = (N / 2) + 1;

  if (out.size() != out_size)
  {
    out.resize(out_size);
  }

  prepare(N);

  CUFFT_CHECK(CufftUtils::cufftExecR2C_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_REAL *>(const_cast<cuda_math::REAL *>(thrust::raw_pointer_cast(in.data()))),
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(out.data()))));
}

// --- CufftHandlerC2R ---

void CufftHandlerC2R::prepare(size_t N)
{
  createPlan(N, CufftUtils::CUFFT_C2R_PLAN);
}

void CufftHandlerC2R::do_ifft(const cuda_math::COMPLEX_DEVICE_VECTOR &in,
                              cuda_math::REAL_DEVICE_VECTOR &out)
{
  if (in.empty())
    return;

  size_t expected_N = (in.size() - 1) * 2;
  if (out.size() != expected_N)
  {
    out.resize(expected_N);
  }
  size_t N = expected_N;

  prepare(N);

  CUFFT_CHECK(CufftUtils::cufftExecC2R_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(const_cast<cuda_math::COMPLEX *>(thrust::raw_pointer_cast(in.data()))),
      reinterpret_cast<cuda_math::CUFFT_REAL *>(thrust::raw_pointer_cast(out.data()))));
}

// --- CufftHandlerC2COut ---

void CufftHandlerC2COut::prepare(size_t N)
{
  createPlan(N, CufftUtils::CUFFT_C2C_PLAN);
}

void CufftHandlerC2COut::do_fft(const cuda_math::COMPLEX_DEVICE_VECTOR &in, cuda_math::COMPLEX_DEVICE_VECTOR &out)
{
  if (in.empty())
    return;

  if (out.size() != in.size())
  {
    out.resize(in.size());
  }

  prepare(in.size());

  CUFFT_CHECK(CufftUtils::cufftExecC2C_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(const_cast<cuda_math::COMPLEX *>(thrust::raw_pointer_cast(in.data()))),
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(out.data())),
      CUFFT_FORWARD));
}

void CufftHandlerC2COut::do_ifft(const cuda_math::COMPLEX_DEVICE_VECTOR &in, cuda_math::COMPLEX_DEVICE_VECTOR &out)
{
  if (in.empty())
    return;

  if (out.size() != in.size())
  {
    out.resize(in.size());
  }

  prepare(in.size());

  CUFFT_CHECK(CufftUtils::cufftExecC2C_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(const_cast<cuda_math::COMPLEX *>(thrust::raw_pointer_cast(in.data()))),
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(out.data())),
      CUFFT_INVERSE));
}

// --- CufftHandlerC2CIn ---

void CufftHandlerC2CIn::prepare(size_t N)
{
  createPlan(N, CufftUtils::CUFFT_C2C_PLAN);
}

void CufftHandlerC2CIn::do_fft(cuda_math::COMPLEX_DEVICE_VECTOR &in)
{
  if (in.empty())
    return;

  prepare(in.size());

  CUFFT_CHECK(CufftUtils::cufftExecC2C_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(in.data())),
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(in.data())),
      CUFFT_FORWARD));
}

void CufftHandlerC2CIn::do_ifft(cuda_math::COMPLEX_DEVICE_VECTOR &in)
{
  if (in.empty())
    return;

  prepare(in.size());

  CUFFT_CHECK(CufftUtils::cufftExecC2C_Wrap(
      fftPlan,
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(in.data())),
      reinterpret_cast<cuda_math::CUFFT_COMPLEX *>(thrust::raw_pointer_cast(in.data())),
      CUFFT_INVERSE));
}