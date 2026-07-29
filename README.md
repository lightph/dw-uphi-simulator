# Domain Wall Core (dw_core)

A high-performance, header-only C++/CUDA library for magnetic domain wall simulations. It provides an RAII-compliant abstraction layer over FFTW (CPU) and cuFFT (GPU) for efficient spectral computations.

## Features

* **Header-Only Architecture:** Designed as a CMake `INTERFACE` library (`dw_core`) for aggressive compiler inlining and seamless integration.
* **Dual Execution Environments:** Support for CPU-based execution via OpenMP-accelerated FFTW3 and GPU-accelerated execution via NVIDIA cuFFT.
* **Precision Toggling:** Configurable for single or double precision via CMake definitions.
* **Memory Safety:** Implements standard C++ allocators to bridge SIMD-aligned C-style memory allocations with standard containers like `std::vector` and Thrust device vectors.

## Requirements

* C++17 or newer
* CMake 3.15+
* FFTW3 (with OpenMP support recommended)
* CUDA Toolkit (if compiling with `HAS_CUDA`)

## Usage and Integration

Because `dw_core` is header-only, you do not need to build it as a shared or static library. Include it in your project by linking against the interface target in your `CMakeLists.txt`:

```cmake
# Example integration
add_executable(my_simulation main.cpp)
target_link_libraries(my_simulation PRIVATE dw_core)
```

## Implementation Notes

### FFT Normalization
The FFT handlers wrap the underlying C APIs closely for maximum performance. Consequently, forward and inverse transforms are **unnormalized**. 

If you perform a forward transform followed immediately by an inverse transform, the resulting data will be scaled by a factor of `N` (the logical size of the array). The calling application is responsible for dividing the output by `N` to recover the original magnitudes.

### Thread Safety and Plans
FFTW and cuFFT plan generation is not inherently thread-safe. Plans should be prepared and managed by a single thread before executing the transforms in parallel or on a CUDA stream.
