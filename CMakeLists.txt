cmake_minimum_required(VERSION 3.20)
project(DomainWallSimulator LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================
# Build Options
# ==========================================
option(USE_CUDA "Enable CUDA support" ON)
option(DOUBLE_PRECISION "Use double precision instead of single precision" OFF)

# Set CUDA architectures. "native" detects the GPU on the current build machine.
# This can be overridden at configure time: -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP)

find_library(FFTW3_LIB NAMES fftw3 REQUIRED)
find_library(FFTW3F_LIB NAMES fftw3f REQUIRED)
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h REQUIRED)

# ==========================================
# Core Library Target
# ==========================================
add_library(dw_core INTERFACE)

# Include directories
target_include_directories(dw_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fft-handlers>
    ${FFTW3_INCLUDE_DIR}
)

if(DOUBLE_PRECISION)
    target_compile_definitions(dw_core INTERFACE DOUBLE_PRECISION)
    target_link_libraries(dw_core INTERFACE ${FFTW3_LIB})
else()
    target_link_libraries(dw_core INTERFACE ${FFTW3F_LIB})
endif()

if(USE_CUDA)
    target_compile_definitions(dw_core INTERFACE HAS_CUDA)
    target_link_libraries(dw_core INTERFACE CUDA::cufft)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(dw_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# ==========================================
# Executable Targets
# ==========================================
add_executable(simulator src/main.cpp)

target_link_libraries(simulator PRIVATE dw_core)
