cmake_minimum_required(VERSION 3.20)

project(DomainWallSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ==============================================================================
# Output Directories
# ==============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==============================================================================
# Language and Flags
# ==============================================================================
 set(CMAKE_CUDA_ARCHITECTURES  60 61 70 75 80 86 CACHE STRING "CUDA architectures to build for") 
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>"
)

# ==============================================================================
# Dependencies
# ==============================================================================
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

pkg_check_modules(FFTW3_SP REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(FFTW3_DP REQUIRED IMPORTED_TARGET fftw3)

find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp REQUIRED)
find_library(FFTW3_OMP_LIB NAMES fftw3_omp REQUIRED)

# ==============================================================================
# Core Header-Only Library Target
# ==============================================================================
add_library(dw_core INTERFACE)

target_include_directories(dw_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core/fft-handlers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core>
)

target_link_libraries(dw_core INTERFACE 
    PkgConfig::FFTW3_SP 
    PkgConfig::FFTW3_DP
    ${FFTW3F_OMP_LIB}
    ${FFTW3_OMP_LIB}
    Threads::Threads
    OpenMP::OpenMP_CXX
    CUDA::cudart 
    CUDA::cufft
)

target_compile_definitions(dw_core INTERFACE HAS_CUDA)

# ==============================================================================
# Executable Targets
# ==============================================================================
add_executable(test_windows src/test_windows.cpp)
set_source_files_properties(src/test_windows.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_windows PRIVATE dw_core)

add_executable(scan src/scan.cpp)
set_source_files_properties(src/scan.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(scan PRIVATE dw_core)