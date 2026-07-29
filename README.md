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
