#pragma once

#include <fftw3.h>

#include <limits>
#include <new>
#include <vector>

namespace dw {

template <typename T>
struct FftwAllocator {
    using value_type = T;

    FftwAllocator() = default;

    template <class U>
    constexpr FftwAllocator(const FftwAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        if (auto p = static_cast<T*>(fftw_malloc(n * sizeof(T)))) {
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t) noexcept { fftw_free(p); }
};

template <typename T, typename U>
bool operator==(const FftwAllocator<T>&, const FftwAllocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const FftwAllocator<T>&, const FftwAllocator<U>&) {
    return false;
}

template <typename T>
using AlignedVector = std::vector<T, FftwAllocator<T>>;

}  // namespace dw