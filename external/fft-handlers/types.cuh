#pragma once

#include <cstdio>
#include <cstdlib>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cufft.h>

namespace cuda_math
{

#define CUDA_CHECK(err) cuda_math::check_cuda(err, __FILE__, __LINE__)

    inline void check_cuda(cudaError_t err, const char *file, int line)
    {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s in file '%s' at line %i.\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
        }
    }

#define CUFFT_CHECK(err) cuda_math::check_cufft(err, __FILE__, __LINE__)

    inline void check_cufft(cufftResult err, const char *file, int line)
    {
        if (err != CUFFT_SUCCESS)
        {
            fprintf(stderr, "cuFFT error code %d in file '%s' at line %i.\n", static_cast<int>(err), file, line);
            exit(EXIT_FAILURE);
        }
    }

#ifdef DOUBLE_PRECISION
    using REAL = double;
    using CUFFT_COMPLEX = cufftDoubleComplex;
    using CUFFT_REAL = cufftDoubleReal;
#else
    using REAL = float;
    using CUFFT_COMPLEX = cufftComplex;
    using CUFFT_REAL = cufftReal;
#endif

    using COMPLEX = thrust::complex<REAL>;
    using REAL_DEVICE_VECTOR = thrust::device_vector<REAL>;
    using REAL_HOST_VECTOR = thrust::host_vector<REAL>;
    using COMPLEX_DEVICE_VECTOR = thrust::device_vector<COMPLEX>;
    using COMPLEX_HOST_VECTOR = thrust::host_vector<COMPLEX>;

} // namespace cuda_math