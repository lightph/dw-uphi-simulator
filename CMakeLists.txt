cmake_minimum_required(VERSION 3.19)

project(DomainWallSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

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
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures to build for")

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

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
    Threads::Threads
    CUDA::cudart
    CUDA::cufft
)

target_compile_definitions(dw_core INTERFACE HAS_CUDA)

# ==============================================================================
# Executable Targets
# ==============================================================================
add_executable(main src/main.cu)
target_link_libraries(main PRIVATE dw_core)

add_executable(moments src/moments.cu)
target_link_libraries(moments PRIVATE dw_core)


add_executable(snapshots src/snapshots.cu)
target_link_libraries(snapshots PRIVATE dw_core)


add_executable(movies src/movies.cu)
target_link_libraries(movies PRIVATE dw_core)


add_executable(sweep src/sweep_main.cu)
target_link_libraries(sweep PRIVATE dw_core)

add_executable(sat_check src/saturation_check.cu)
target_link_libraries(sat_check PRIVATE dw_core)
