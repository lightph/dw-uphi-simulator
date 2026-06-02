#include "DomainWallSimulator.h"
#include <stdexcept>
#include <iostream>

// Forward declarations to the internal routines
std::unique_ptr<IDomainWallSimulator> create_cpu_simulator(const SimulatorConfig &config);

#ifdef HAS_CUDA
// These only exist if we compiled with CUDA support
std::unique_ptr<IDomainWallSimulator> create_gpu_simulator(const SimulatorConfig &config);
bool is_cuda_available();
#endif

std::unique_ptr<IDomainWallSimulator> create_simulator(const SimulatorConfig &config)
{
    if (config.backend == ComputeBackend::GPU)
    {
#ifdef HAS_CUDA
        if (is_cuda_available())
        {
            return create_gpu_simulator(config);
        }
        else
        {
            throw std::runtime_error("Runtime Error: GPU backend requested, but no CUDA-capable device was found.");
        }
#else
        // This block executes if the code was compiled with USE_CUDA=OFF
        throw std::runtime_error("Compile-time Error: GPU backend requested, but the program was compiled without CUDA support.");
#endif
    }

    // Default to CPU (Always available)
    return create_cpu_simulator(config);
}