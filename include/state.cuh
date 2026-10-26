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

    ComplexVector u_hat;
    ComplexVector nl_eval;
    ComplexVector nl_hat;

    std::size_t size_;
    std::size_t batch_;

    explicit State(std::size_t size, std::size_t batch = 1)
        : u(size * batch),
          u_prev(size * batch),
          u_hat(size * batch),
          nl_eval(size * batch),
          nl_hat(size * batch),
          size_(size),
          batch_(batch) {}

    // Sub-grid accessors
    Complex* get_u(std::size_t b = 0) { return u.data() + b * size_; }
    const Complex* get_u(std::size_t b = 0) const { return u.data() + b * size_; }

    Complex* get_u_prev(std::size_t b = 0) { return u_prev.data() + b * size_; }
    const Complex* get_u_prev(std::size_t b = 0) const { return u_prev.data() + b * size_; }

    Complex* get_u_hat(std::size_t b = 0) { return u_hat.data() + b * size_; }
    const Complex* get_u_hat(std::size_t b = 0) const { return u_hat.data() + b * size_; }

    Complex* get_nl_eval(std::size_t b = 0) { return nl_eval.data() + b * size_; }
    const Complex* get_nl_eval(std::size_t b = 0) const { return nl_eval.data() + b * size_; }

    Complex* get_nl_hat(std::size_t b = 0) { return nl_hat.data() + b * size_; }
    const Complex* get_nl_hat(std::size_t b = 0) const { return nl_hat.data() + b * size_; }
};

}  // namespace dw
