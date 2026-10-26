cmake_minimum_required(VERSION 3.19)

project(DomainWallSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS ON)
endif()

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
if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures to build for")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>"
    )
endif()

# ==============================================================================
# Dependencies
# ==============================================================================
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(USE_CUDA)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

find_package(Threads REQUIRED)

# Always find CPU dependencies (FFTW3 and OpenMP)
find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
pkg_check_modules(FFTW3_DP REQUIRED IMPORTED_TARGET fftw3)
find_library(FFTW3_OMP_LIB NAMES fftw3_omp REQUIRED)

# Conditionally find GPU dependencies
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ==============================================================================
# Core Header-Only Library Target
# ==============================================================================
add_library(dw_core INTERFACE)

target_include_directories(dw_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core/fft-handlers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core>
)

# Always link CPU dependencies
target_link_libraries(dw_core INTERFACE 
    Threads::Threads
    PkgConfig::FFTW3_DP
    ${FFTW3_OMP_LIB}
    OpenMP::OpenMP_CXX
)

# Conditionally link GPU dependencies
if(USE_CUDA)
    target_link_libraries(dw_core INTERFACE CUDA::cudart CUDA::cufft)
    target_compile_definitions(dw_core INTERFACE HAS_CUDA)
endif()

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

# If CUDA is disabled, treat .cu files as standard C++ files
if(NOT USE_CUDA)
    set_source_files_properties(src/main.cu src/moments.cu src/snapshots.cu src/movies.cu src/sweep_main.cu src/saturation_check.cu PROPERTIES LANGUAGE CXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/main.cu src/moments.cu src/snapshots.cu src/movies.cu src/sweep_main.cu src/saturation_check.cu PROPERTIES COMPILE_FLAGS "-x c++")
    endif()
endif()
