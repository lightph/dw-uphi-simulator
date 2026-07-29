#pragma once

#include <complex>
#include <vector>
#include <new>

#include <fftw3.h>

#ifdef HAS_CUDA
#include <cufft.h>
#include <thrust/device_vector.h>
#endif

/// @brief Core namespace for the domain wall simulator.
namespace dw
{

#ifdef DOUBLE_PRECISION
    // ==========================================
    // Double Precision Types
    // ==========================================

    /// Standard C++ double-precision floating point.
    using Real = double;

    /// Standard C++ double-precision complex number.
    using Complex = std::complex<double>;

    /// FFTW double-precision real number.
    using FftwReal = double;

    /// FFTW double-precision complex number.
    using FftwComplex = fftw_complex;

    /// FFTW double-precision execution plan.
    using FftwPlan = fftw_plan;

#ifdef HAS_CUDA
    /// cuFFT double-precision real number.
    using CufftReal = cufftDoubleReal;

    /// cuFFT double-precision complex number.
    using CufftComplex = cufftDoubleComplex;
#endif

#else
    // ==========================================
    // Single Precision Types
    // ==========================================

    /// Standard C++ single-precision floating point.
    using Real = float;

    /// Standard C++ single-precision complex number.
    using Complex = std::complex<float>;

    /// FFTW single-precision real number.
    using FftwReal = float;

    /// FFTW single-precision complex number.
    using FftwComplex = fftwf_complex;

    /// FFTW single-precision execution plan.
    using FftwPlan = fftwf_plan;

#ifdef HAS_CUDA
    /// cuFFT single-precision real number.
    using CufftReal = cufftReal;

    /// cuFFT single-precision complex number.
    using CufftComplex = cufftComplex;
#endif

#endif // End of precision selection

    // ==========================================
    // Vector Types and Allocators
    // ==========================================

    /// @brief Standard host vector using the default allocator.
    /// @tparam T Type of the elements.
    template <typename T>
    using HostVector = std::vector<T>;

#ifdef DOUBLE_PRECISION
    /// Wrapper for FFTW double-precision memory allocation.
    inline void *fft_malloc(std::size_t n) { return fftw_malloc(n); }

    /// Wrapper for FFTW double-precision memory deallocation.
    inline void fft_free(void *p) { fftw_free(p); }
#else
    /// Wrapper for FFTW single-precision memory allocation.
    inline void *fft_malloc(std::size_t n) { return fftwf_malloc(n); }

    /// Wrapper for FFTW single-precision memory deallocation.
    inline void fft_free(void *p) { fftwf_free(p); }
#endif

    /// @brief Custom allocator using FFTW's aligned memory routines.
    ///
    /// This ensures memory is properly aligned for SIMD vectorization,
    /// maximizing FFT performance on the CPU.
    /// @tparam T Type of the elements to allocate.
    template <typename T>
    struct FftwAllocator
    {
        /// Type of elements managed by the allocator.
        using value_type = T;

        /// Default constructor.
        FftwAllocator() = default;

        /// Copy constructor for rebinding.
        template <typename U>
        constexpr FftwAllocator(const FftwAllocator<U> &) noexcept {}

        /// @brief Allocates aligned memory.
        /// @param n Number of elements to allocate.
        /// @return Pointer to the allocated memory.
        /// @throws std::bad_alloc if the underlying allocation fails.
        T *allocate(std::size_t n)
        {
            if (auto p = static_cast<T *>(fft_malloc(n * sizeof(T))))
            {
                return p;
            }
            throw std::bad_alloc();
        }

        /// @brief Deallocates aligned memory.
        /// @param p Pointer to the memory to deallocate.
        void deallocate(T *p, std::size_t) noexcept
        {
            fft_free(p);
        }
    };

    /// @brief CPU vector guaranteed to have SIMD-aligned memory for FFTW.
    /// @tparam T Type of the elements.
    template <typename T>
    using AlignedVector = std::vector<T, FftwAllocator<T>>;

#ifdef HAS_CUDA
    /// @brief Device vector for CUDA memory management.
    /// @tparam T Type of the elements.
    template <typename T>
    using DeviceVector = thrust::device_vector<T>;
#endif

} // namespace dw