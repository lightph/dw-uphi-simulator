#pragma once

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "types.hpp"

#ifdef HAS_CUDA

// Optional macro if you don't already have one defined globally
#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                                                          \
    do {                                                                           \
        cufftResult err = call;                                                    \
        if (err != CUFFT_SUCCESS) {                                                \
            fprintf(stderr, "cuFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)
#endif

namespace dw {

// ==============================================================================
// Precision Macros for cuFFT
// ==============================================================================
#ifdef DOUBLE_PRECISION
#define DW_CUFFT_EXEC_C2C cufftExecZ2Z
#define DW_CUFFT_EXEC_R2C cufftExecD2Z
#define DW_CUFFT_EXEC_C2R cufftExecZ2D
constexpr cufftType DW_CUFFT_C2C_PLAN = CUFFT_Z2Z;
constexpr cufftType DW_CUFFT_R2C_PLAN = CUFFT_D2Z;
constexpr cufftType DW_CUFFT_C2R_PLAN = CUFFT_Z2D;
#else
#define DW_CUFFT_EXEC_C2C cufftExecC2C
#define DW_CUFFT_EXEC_R2C cufftExecR2C
#define DW_CUFFT_EXEC_C2R cufftExecC2R
constexpr cufftType DW_CUFFT_C2C_PLAN = CUFFT_C2C;
constexpr cufftType DW_CUFFT_R2C_PLAN = CUFFT_R2C;
constexpr cufftType DW_CUFFT_C2R_PLAN = CUFFT_C2R;
#endif

/// @brief Base class for managing cuFFT handles.
class CufftHandler {
   protected:
    size_t current_n_ = 0;
    cufftHandle fft_plan_ = 0;

    /// @brief Creates and allocates the cuFFT plan.
    void createPlan(size_t N, cufftType planType) {
        if (N == current_n_) return;
        cleanup();
        current_n_ = N;
        CUFFT_CHECK(cufftPlan1d(&fft_plan_, current_n_, planType, 1));
    }

   public:
    CufftHandler() = default;

    virtual ~CufftHandler() { cleanup(); }

    CufftHandler(const CufftHandler&) = delete;
    CufftHandler& operator=(const CufftHandler&) = delete;

    /// @brief Prepares the FFT plan for a specific size.
    virtual void prepare(size_t N) = 0;

    /// @brief Cleans up and destroys the cuFFT plan.
    void cleanup() {
        if (fft_plan_ != 0) {
            cufftDestroy(fft_plan_);
            fft_plan_ = 0;
        }
        current_n_ = 0;
    }

    /// @brief Gets the size of the current plan.
    size_t getSize() const { return current_n_; }

    /// @brief Assigns a CUDA stream to the cuFFT execution plan.
    void setStream(cudaStream_t stream) {
        if (fft_plan_ != 0) {
            cufftSetStream(fft_plan_, stream);
        }
    }
};

/// @brief cuFFT Handler for Real-to-Complex transforms.
class CufftHandlerR2C : public CufftHandler {
   public:
    void prepare(size_t N) override { createPlan(N, DW_CUFFT_R2C_PLAN); }

    /// @brief Executes the forward Real-to-Complex transform on device memory.
    void do_fft(const DeviceVector<Real>& in, DeviceVector<Complex>& out) {
        if (in.empty()) return;

        size_t N = in.size();
        size_t out_size = (N / 2) + 1;

        if (out.size() != out_size) out.resize(out_size);
        prepare(N);

        CUFFT_CHECK(DW_CUFFT_EXEC_R2C(
            fft_plan_,
            reinterpret_cast<CufftReal*>(const_cast<Real*>(thrust::raw_pointer_cast(in.data()))),
            reinterpret_cast<CufftComplex*>(thrust::raw_pointer_cast(out.data()))));
    }
};

/// @brief cuFFT Handler for Complex-to-Real transforms.
class CufftHandlerC2R : public CufftHandler {
   public:
    void prepare(size_t N) override { createPlan(N, DW_CUFFT_C2R_PLAN); }

    /// @brief Executes the backward Complex-to-Real transform on device memory.
    void do_ifft(const DeviceVector<Complex>& in, DeviceVector<Real>& out) {
        if (in.empty()) return;

        size_t expected_N = (in.size() - 1) * 2;
        if (out.size() != expected_N) out.resize(expected_N);

        prepare(expected_N);

        CUFFT_CHECK(
            DW_CUFFT_EXEC_C2R(fft_plan_,
                              reinterpret_cast<CufftComplex*>(
                                  const_cast<Complex*>(thrust::raw_pointer_cast(in.data()))),
                              reinterpret_cast<CufftReal*>(thrust::raw_pointer_cast(out.data()))));
    }
};

/// @brief cuFFT Handler for in-place Complex-to-Complex transforms.
class CufftHandlerC2CIn : public CufftHandler {
   public:
    void prepare(size_t N) override { createPlan(N, DW_CUFFT_C2C_PLAN); }

    /// @brief Executes an in-place forward transform.
    void do_fft(DeviceVector<Complex>& in) {
        if (in.empty()) return;
        prepare(in.size());

        CUFFT_CHECK(DW_CUFFT_EXEC_C2C(
            fft_plan_, reinterpret_cast<CufftComplex*>(thrust::raw_pointer_cast(in.data())),
            reinterpret_cast<CufftComplex*>(thrust::raw_pointer_cast(in.data())), CUFFT_FORWARD));
    }

    /// @brief Executes an in-place inverse transform.
    void do_ifft(DeviceVector<Complex>& in) {
        if (in.empty()) return;
        prepare(in.size());

        CUFFT_CHECK(DW_CUFFT_EXEC_C2C(
            fft_plan_, reinterpret_cast<CufftComplex*>(thrust::raw_pointer_cast(in.data())),
            reinterpret_cast<CufftComplex*>(thrust::raw_pointer_cast(in.data())), CUFFT_INVERSE));
    }
};

}  // namespace dw

#endif  // HAS_CUDA