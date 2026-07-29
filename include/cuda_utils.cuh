#pragma once

#ifdef HAS_CUDA

#include <cufft.h>

#include <stdexcept>
#include <string>

#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                                                                     \
    do {                                                                                      \
        cufftResult err = call;                                                               \
        if (err != CUFFT_SUCCESS) {                                                           \
            throw std::runtime_error("cuFFT error " + std::to_string(err) + " at " +          \
                                     std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        }                                                                                     \
    } while (0)
#endif

#endif  // HAS_CUDA