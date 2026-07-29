#pragma once
#include <complex>

namespace dw {

struct SinglePrecision {
    using Real = float;
    using Complex = std::complex<float>;
};

struct DoublePrecision {
    using Real = double;
    using Complex = std::complex<double>;
};

}  // namespace dw