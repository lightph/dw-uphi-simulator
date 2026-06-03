cmake_minimum_required(VERSION 3.15)
project(investigacion1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==============================================================================
# GPU/CUDA Toggle
# ==============================================================================
option(USE_CUDA "Enable CUDA support" ON)
option(USE_DOUBLE_PRECISION "Use double precision (ON) or single precision (OFF)" ON)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
        add_compile_definitions(HAS_CUDA)
        
        # Add the extended lambda flag here
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
        
        message(STATUS ">>> CUDA enabled (Archs: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        set(USE_CUDA OFF)
        message(WARNING ">>> CUDA compiler not found. Falling back to CPU-only build.")
    endif()
endif()

# ==============================================================================
# Precision 
# ==============================================================================
if(USE_DOUBLE_PRECISION)
    add_compile_definitions(DOUBLE_PRECISION)
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

include_directories(
    fft-handlers
    domain_wall_simulator/include 
    include
)

add_library(filesaver INTERFACE)
target_include_directories(filesaver INTERFACE include)
target_link_libraries(filesaver INTERFACE Threads::Threads stdc++fs)

# CPU FFT Handler (Always built)
add_library(fftw_handler${PREC_SUFFIX} STATIC fft-handlers/FftwHandler.cpp)
target_link_libraries(fftw_handler${PREC_SUFFIX} PUBLIC 
    ${FFTW3_LIBRARIES} 
    ${FFTW_OMP_LIB} 
    OpenMP::OpenMP_CXX
)

# Core Simulator
set(SIMULATOR_SOURCES 
    domain_wall_simulator/src/SimulatorFactory.cpp
    domain_wall_simulator/src/SimulatorCPU.cpp
)

if(USE_CUDA)
    # GPU FFT Handler (Only if CUDA is ON)
    add_library(cufft_handler${PREC_SUFFIX} STATIC fft-handlers/cufftHandler.cu)
    target_link_libraries(cufft_handler${PREC_SUFFIX} PUBLIC CUDA::cudart CUDA::cufft stdc++)
    
    list(APPEND SIMULATOR_SOURCES domain_wall_simulator/src/SimulatorGPU.cu)
endif()

add_library(simulator_core${PREC_SUFFIX} STATIC ${SIMULATOR_SOURCES})
target_link_libraries(simulator_core${PREC_SUFFIX} PUBLIC fftw_handler${PREC_SUFFIX} filesaver)

if(USE_CUDA)
    target_link_libraries(simulator_core${PREC_SUFFIX} PUBLIC cufft_handler${PREC_SUFFIX})
endif()

# ==============================================================================
# Programs
# ==============================================================================
set(PROGS h_sweep complete_h_sweep structure_analysis movie)
foreach(prog IN LISTS PROGS)
    add_executable(${prog}${PREC_SUFFIX} programs/${prog}.cpp)
    target_link_libraries(${prog}${PREC_SUFFIX} PRIVATE simulator_core${PREC_SUFFIX})
endforeach()
