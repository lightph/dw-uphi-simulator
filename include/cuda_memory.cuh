#pragma once

#include <cuda_runtime.h>

#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace dw {

template <typename T>
struct CudaAllocator {
    using value_type = T;

    CudaAllocator() = default;

    template <class U>
    constexpr CudaAllocator(const CudaAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        T* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaAllocator failed to allocate memory: ") +
                                     cudaGetErrorString(err));
        }

        return ptr;
    }

    void deallocate(T* p, std::size_t) noexcept { cudaFree(p); }
};

template <typename T, typename U>
bool operator==(const CudaAllocator<T>&, const CudaAllocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const CudaAllocator<T>&, const CudaAllocator<U>&) {
    return false;
}

template <typename T>
using CudaVector = std::vector<T, CudaAllocator<T>>;

}  // namespace dw