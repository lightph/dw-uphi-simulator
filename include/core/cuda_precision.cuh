#pragma once
#include <cufft.h>
#include <thrust/complex.h>  // Include thrust complex

#include "precision.hpp"

namespace dw {

struct CudaDoublePrecision : public DoublePrecision {
    using Complex = thrust::complex<double>;
    using CufftReal = cufftDoubleReal;
    using CufftComplex = cufftDoubleComplex;
};

}  // namespace dw
