#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "state.cuh"

namespace dw {

template <typename Backend>
class PseudospectralStepper {
   public:
    using Complex = typename Backend::PrecisionType::Complex;
    using ComplexVector = typename Backend::template Vector<Complex>;

    using C2COut = typename Backend::C2COut;

   private:
    C2COut fft_handler;
    std::size_t size_;
    std::size_t batch_;

   public:
    explicit PseudospectralStepper(std::size_t size, std::size_t batch = 1)
        : size_(size), batch_(batch) {
        fft_handler.prepare(size_, batch_);
    }

    template <typename LinearStepper, typename NonlinearEvaluator>
    void step(State<Backend>& state, double dt, LinearStepper&& lin_step,
              NonlinearEvaluator&& nonlin_eval, cudaStream_t stream = nullptr) {
        // Pass raw pointers, total element count, and the stream to the evaluators
        std::size_t total_elements = state.size_ * state.batch_;

        nonlin_eval(state.u.data(), state.nl_eval.data(), total_elements, stream);

        fft_handler.do_fft(state.u, state.u_hat, state.size_, state.batch_, stream);
        fft_handler.do_fft(state.nl_eval, state.nl_hat, state.size_, state.batch_, stream);

        // We pass state.size_ explicitly so the linear stepper can use it for correct FFT
        // normalization, while total_elements dictates the kernel launch bounds.
        lin_step(state.u_hat.data(), state.nl_hat.data(), dt, state.size_, total_elements, stream);

        fft_handler.do_ifft(state.u_hat, state.u, state.size_, state.batch_, stream);
    }
};

}  // namespace dw