#include "DomainWallSimulator.h"
#include <stdexcept>
#include <iostream>

std::unique_ptr<IDomainWallSimulator> create_cpu_simulator(const SimulatorConfig &config);

#ifdef HAS_CUDA
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
        throw std::runtime_error("Compile-time Error: GPU backend requested, but the program was compiled without CUDA support.");
#endif
    }
    return create_cpu_simulator(config);
}