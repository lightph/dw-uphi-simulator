cmake_minimum_required(VERSION 3.19)

project(DomainWallSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    option(USE_CUDA "Build with CUDA support" ON)
else()
    option(USE_CUDA "Build with CUDA support" OFF)
endif()

if(USE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "USE_CUDA is ON, but CUDA compiler (nvcc) was not found.")
    endif()
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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Disable PIE to allow linking against static libraries not compiled with -fPIC
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
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
list(APPEND CMAKE_PREFIX_PATH "/usr/local" "/usr/local/lib" "/usr/lib64")
if(DEFINED ENV{PKG_CONFIG_PATH})
    set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
else()
    set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:/usr/lib64/pkgconfig:/usr/share/pkgconfig")
endif()

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
pkg_check_modules(FFTW3_DP REQUIRED IMPORTED_TARGET fftw3)
find_library(FFTW3_OMP_LIB NAMES fftw3_omp fftw3_omp.so.3 libfftw3_omp.so.3 HINTS /usr/lib64 /usr/local/lib /usr/lib REQUIRED)

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

add_executable(time_evo src/time_evo.cu)
target_link_libraries(time_evo PRIVATE dw_core)

# Modular aliases
add_executable(pde_main ALIAS main)
add_executable(pde_time_evo ALIAS time_evo)
add_executable(pde_sweep ALIAS sweep)
add_executable(pde_snapshots ALIAS snapshots)
add_executable(pde_moments ALIAS moments)
add_executable(pde_movies ALIAS movies)
add_executable(pde_sat_check ALIAS sat_check)

# ==============================================================================
# Rigid Wall Module Targets
# ==============================================================================
add_executable(rigid_wall_sweep_cpu rigid_wall/src/rigid_wall_sweep_cpu.cpp)
target_include_directories(rigid_wall_sweep_cpu PRIVATE rigid_wall/include)
target_link_libraries(rigid_wall_sweep_cpu PRIVATE OpenMP::OpenMP_CXX Threads::Threads)

if(USE_CUDA)
    add_executable(rigid_wall_sweep_gpu rigid_wall/src/rigid_wall_sweep_gpu.cu)
    target_include_directories(rigid_wall_sweep_gpu PRIVATE rigid_wall/include)
    target_link_libraries(rigid_wall_sweep_gpu PRIVATE CUDA::cudart)
endif()

# ==============================================================================
# Floquet Stability Module Targets
# ==============================================================================
add_executable(floquet_sweep_cpu floquet/src/floquet_sweep_cpu.cpp)
target_include_directories(floquet_sweep_cpu PRIVATE floquet/include)
target_link_libraries(floquet_sweep_cpu PRIVATE OpenMP::OpenMP_CXX Threads::Threads)

if(USE_CUDA)
    add_executable(floquet_sweep_gpu floquet/src/floquet_sweep_gpu.cu)
    target_include_directories(floquet_sweep_gpu PRIVATE floquet/include)
    target_link_libraries(floquet_sweep_gpu PRIVATE CUDA::cudart)
endif()

# If CUDA is disabled, treat .cu files as standard C++ files
if(NOT USE_CUDA)
    set_source_files_properties(src/main.cu src/moments.cu src/snapshots.cu src/movies.cu src/sweep_main.cu src/saturation_check.cu src/time_evo.cu PROPERTIES LANGUAGE CXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/main.cu src/moments.cu src/snapshots.cu src/movies.cu src/sweep_main.cu src/saturation_check.cu src/time_evo.cu PROPERTIES COMPILE_FLAGS "-x c++")
    endif()
endif()
