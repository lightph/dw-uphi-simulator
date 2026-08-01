#pragma once

#include <cufft.h>

#include <utility>
#include <vector>

#include "cuda_memory.cuh"
#include "cuda_precision.cuh"
#include "cuda_utils.cuh"

namespace dw {

// ==============================================================================
// API Traits for cuFFT (Replaces Preprocessor Macros)
// ==============================================================================
template <typename RealType>
struct CufftApi;

template <>
struct CufftApi<float> {
    static constexpr cufftType R2C_PLAN = CUFFT_R2C;
    static constexpr cufftType C2R_PLAN = CUFFT_C2R;
    static constexpr cufftType C2C_PLAN = CUFFT_C2C;

    static cufftResult exec_r2c(cufftHandle plan, cufftReal* idata, cufftComplex* odata) {
        return cufftExecR2C(plan, idata, odata);
    }
    static cufftResult exec_c2r(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
        return cufftExecC2R(plan, idata, odata);
    }
    static cufftResult exec_c2c(cufftHandle plan, cufftComplex* idata, cufftComplex* odata,
                                int direction) {
        return cufftExecC2C(plan, idata, odata, direction);
    }
};

template <>
struct CufftApi<double> {
    static constexpr cufftType R2C_PLAN = CUFFT_D2Z;
    static constexpr cufftType C2R_PLAN = CUFFT_Z2D;
    static constexpr cufftType C2C_PLAN = CUFFT_Z2Z;

    static cufftResult exec_r2c(cufftHandle plan, cufftDoubleReal* idata,
                                cufftDoubleComplex* odata) {
        return cufftExecD2Z(plan, idata, odata);
    }
    static cufftResult exec_c2r(cufftHandle plan, cufftDoubleComplex* idata,
                                cufftDoubleReal* odata) {
        return cufftExecZ2D(plan, idata, odata);
    }
    static cufftResult exec_c2c(cufftHandle plan, cufftDoubleComplex* idata,
                                cufftDoubleComplex* odata, int direction) {
        return cufftExecZ2Z(plan, idata, odata, direction);
    }
};

/// @brief Base class for managing cuFFT handles.
template <typename Precision>
class CufftBaseHandler {
   protected:
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using CufftReal = typename Precision::CufftReal;
    using CufftComplex = typename Precision::CufftComplex;

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
    CufftBaseHandler() = default;

    virtual ~CufftBaseHandler() { cleanup(); }

    // Disable copying
    CufftBaseHandler(const CufftBaseHandler&) = delete;
    CufftBaseHandler& operator=(const CufftBaseHandler&) = delete;

    // Enable moving
    CufftBaseHandler(CufftBaseHandler&& other) noexcept
        : current_n_(other.current_n_), fft_plan_(other.fft_plan_) {
        other.fft_plan_ = 0;
        other.current_n_ = 0;
    }

    CufftBaseHandler& operator=(CufftBaseHandler&& other) noexcept {
        if (this != &other) {
            cleanup();
            current_n_ = other.current_n_;
            fft_plan_ = other.fft_plan_;
            other.fft_plan_ = 0;
            other.current_n_ = 0;
        }
        return *this;
    }

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
template <typename Precision>
class CufftHandlerR2C : public CufftBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using CufftReal = typename Precision::CufftReal;
    using CufftComplex = typename Precision::CufftComplex;

   public:
    void prepare(size_t N) override { this->createPlan(N, CufftApi<Real>::R2C_PLAN); }

    /// @brief Executes the forward Real-to-Complex transform on Unified Memory.
    void do_fft(const CudaVector<Real>& in, CudaVector<Complex>& out) {
        if (in.empty()) return;

        size_t N = in.size();
        size_t out_size = (N / 2) + 1;

        if (out.size() != out_size) out.resize(out_size);
        prepare(N);

        CUFFT_CHECK(CufftApi<Real>::exec_r2c(
            this->fft_plan_, reinterpret_cast<CufftReal*>(const_cast<Real*>(in.data())),
            reinterpret_cast<CufftComplex*>(out.data())));
    }
};

/// @brief cuFFT Handler for Complex-to-Real transforms.
template <typename Precision>
class CufftHandlerC2R : public CufftBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using CufftReal = typename Precision::CufftReal;
    using CufftComplex = typename Precision::CufftComplex;

   public:
    void prepare(size_t N) override { this->createPlan(N, CufftApi<Real>::C2R_PLAN); }

    /// @brief Executes the backward Complex-to-Real transform on Unified Memory.
    void do_ifft(const CudaVector<Complex>& in, CudaVector<Real>& out) {
        if (in.empty()) return;

        size_t expected_N = (in.size() - 1) * 2;
        if (out.size() != expected_N) out.resize(expected_N);

        prepare(expected_N);

        CUFFT_CHECK(CufftApi<Real>::exec_c2r(
            this->fft_plan_, reinterpret_cast<CufftComplex*>(const_cast<Complex*>(in.data())),
            reinterpret_cast<CufftReal*>(out.data())));
    }
};

/// @brief cuFFT Handler for in-place Complex-to-Complex transforms.
template <typename Precision>
class CufftHandlerC2CIn : public CufftBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using CufftComplex = typename Precision::CufftComplex;

   public:
    void prepare(size_t N) override { this->createPlan(N, CufftApi<Real>::C2C_PLAN); }

    /// @brief Executes an in-place forward transform.
    void do_fft(CudaVector<Complex>& in) {
        if (in.empty()) return;
        prepare(in.size());

        CUFFT_CHECK(
            CufftApi<Real>::exec_c2c(this->fft_plan_, reinterpret_cast<CufftComplex*>(in.data()),
                                     reinterpret_cast<CufftComplex*>(in.data()), CUFFT_FORWARD));
    }

    /// @brief Executes an in-place inverse transform.
    void do_ifft(CudaVector<Complex>& in) {
        if (in.empty()) return;
        prepare(in.size());

        CUFFT_CHECK(
            CufftApi<Real>::exec_c2c(this->fft_plan_, reinterpret_cast<CufftComplex*>(in.data()),
                                     reinterpret_cast<CufftComplex*>(in.data()), CUFFT_INVERSE));
    }
};

/// @brief cuFFT Handler for out-of-place Complex-to-Complex transforms.
template <typename Precision>
class CufftHandlerC2COut : public CufftBaseHandler<Precision> {
    using Real = typename Precision::Real;
    using Complex = typename Precision::Complex;
    using CufftComplex = typename Precision::CufftComplex;

   public:
    void prepare(size_t N) override { this->createPlan(N, CufftApi<Real>::C2C_PLAN); }

    /// @brief Executes an out-of-place forward transform.
    void do_fft(const CudaVector<Complex>& in, CudaVector<Complex>& out) {
        if (in.empty()) return;
        size_t N = in.size();
        if (out.size() != N) out.resize(N);
        prepare(N);

        CUFFT_CHECK(CufftApi<Real>::exec_c2c(
            this->fft_plan_, reinterpret_cast<CufftComplex*>(const_cast<Complex*>(in.data())),
            reinterpret_cast<CufftComplex*>(out.data()), CUFFT_FORWARD));
    }

    /// @brief Executes an out-of-place inverse transform.
    void do_ifft(const CudaVector<Complex>& in, CudaVector<Complex>& out) {
        if (in.empty()) return;
        size_t N = in.size();
        if (out.size() != N) out.resize(N);
        prepare(N);

        CUFFT_CHECK(CufftApi<Real>::exec_c2c(
            this->fft_plan_, reinterpret_cast<CufftComplex*>(const_cast<Complex*>(in.data())),
            reinterpret_cast<CufftComplex*>(out.data()), CUFFT_INVERSE));
    }
};

}  // namespace dw