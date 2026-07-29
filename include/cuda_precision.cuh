#pragma once
#include <cufft.h>

#include "precision.hpp"

namespace dw {

struct CudaSinglePrecision : public SinglePrecision {
    using CufftReal = cufftReal;
    using CufftComplex = cufftComplex;
};

struct CudaDoublePrecision : public DoublePrecision {
    using CufftReal = cufftDoubleReal;
    using CufftComplex = cufftDoubleComplex;
};

}  // namespace dw