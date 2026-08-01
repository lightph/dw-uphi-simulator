#pragma once

#include <cstddef>

namespace dw {
template <typename Backend>
class State {
   public:
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    ComplexVector u;
    ComplexVector u_prev;
    ComplexVector delta_u;

    ComplexVector u_hat;
    ComplexVector nl_eval;
    ComplexVector nl_hat;

    std::size_t size_;

    explicit State(std::size_t size)
        : u(size),
          u_prev(size),
          delta_u(size),
          u_hat(size),
          nl_eval(size),
          nl_hat(size),
          size_(size) {}

    void compute_delta() {
        Backend::compute_difference(u.data(), u_prev.data(), delta_u.data(), size_);
    }
};

}  // namespace dw