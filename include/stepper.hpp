#pragma once

#include <cstddef>

namespace dw {

template <typename Backend>
class PseudospectralStepper {
   public:
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    using C2COut = typename Backend::C2COut;

   private:
    C2COut fft_handler;
    C2COut ifft_handler;
    std::size_t size_;

   public:
    explicit PseudospectralStepper(std::size_t size) : size_(size) {
        fft_handler.prepare(size);
        ifft_handler.prepare(size);
    }

    template <typename LinearStepper, typename NonlinearEvaluator>
    void step(State<Backend>& state, double dt, LinearStepper&& lin_step,
              NonlinearEvaluator&& nonlin_eval) {
        nonlin_eval(state.u, state.nl_eval);

        fft_handler.do_fft(state.u, state.u_hat);
        fft_handler.do_fft(state.nl_eval, state.nl_hat);

        lin_step(state.u_hat, state.nl_hat, dt);

        ifft_handler.do_ifft(state.u_hat, state.u);
    }
};

}  // namespace dw