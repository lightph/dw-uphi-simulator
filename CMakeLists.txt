cmake_minimum_required(VERSION 3.20)
project(DomainWallSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Output Directories
# ==============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==============================================================================
# Build Options
# ==============================================================================
option(USE_CUDA "Enable CUDA support" ON)
option(USE_DOUBLE_PRECISION "Use double precision (ON) or single precision (OFF)" OFF)

# ==============================================================================
# Language and Flags
# ==============================================================================
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        
        # Consistent with your reference but exposed as a cache string for flexibility
        set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86 CACHE STRING "CUDA architectures to build for")
        
        # Add the extended lambda flag
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
        
        message(STATUS ">>> CUDA enabled (Archs: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        set(USE_CUDA OFF)
        message(WARNING ">>> CUDA compiler not found. Falling back to CPU-only build.")
    endif()
endif()

# ==============================================================================
# Precision Configuration
# ==============================================================================
if(USE_DOUBLE_PRECISION)
    set(PREC_SUFFIX "_dp")
    set(FFTW_PKG "fftw3")
    set(FFTW_OMP_LIB "fftw3_omp")
else()
    set(PREC_SUFFIX "_sp")
    set(FFTW_PKG "fftw3f")
    set(FFTW_OMP_LIB "fftw3f_omp")
endif()
message(STATUS ">>> Building with precision: ${PREC_SUFFIX}")

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
pkg_check_modules(FFTW3 REQUIRED ${FFTW_PKG})

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ==============================================================================
# Core Header-Only Library Target
# ==============================================================================
add_library(dw_core INTERFACE)

target_include_directories(dw_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fft-handlers>
    ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(dw_core INTERFACE 
    ${FFTW3_LIBRARIES} 
    ${FFTW_OMP_LIB}
    Threads::Threads
    OpenMP::OpenMP_CXX
)

if(USE_DOUBLE_PRECISION)
    target_compile_definitions(dw_core INTERFACE DOUBLE_PRECISION)
endif()

if(USE_CUDA)
    target_compile_definitions(dw_core INTERFACE HAS_CUDA)
    target_link_libraries(dw_core INTERFACE CUDA::cudart CUDA::cufft)
endif()

# ==============================================================================
# Executable Targets
# ==============================================================================
add_executable(dummy${PREC_SUFFIX} src/main.cpp)
target_link_libraries(dummy${PREC_SUFFIX} PRIVATE dw_core)