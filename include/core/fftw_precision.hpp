#pragma once
#include <fftw3.h>

#include "precision.hpp"

namespace dw {

struct FftwDoublePrecision : public DoublePrecision {
    using FftwReal = double;
    using FftwComplex = fftw_complex;
};

}  // namespace dw
