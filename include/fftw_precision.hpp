#pragma once
#include <fftw3.h>

#include "precision.hpp"

struct FftwSinglePrecision : public SinglePrecision {
    using FftwReal = float;
    using FftwComplex = fftwf_complex;
};

struct FftwDoublePrecision : public DoublePrecision {
    using FftwReal = double;
    using FftwComplex = fftw_complex;
};