#pragma once

#include "cuda_runtime.h"

#include "../params.h"

#define CUDA_CALL(result) \
    checkCudaError(result, __FUNCTION__, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t result, const std::string& functionName,
                           const std::string& fileName, int lineNumber) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << functionName << " at " << fileName
                  << ":" << lineNumber << " - " << cudaGetErrorString(result)
                  << std::endl;
    }
}

enum GridType { Half, Full, Linear };

class KernelCaller {
private:
    // One-dimensional grid
    static const unsigned int dimBlockLinear =
        mhd::parameters::KernelRunParameters::blockSizeLinear;
    static const unsigned int dimGridLinear =
        mhd::parameters::KernelRunParameters::gridSizeLinear;

    // Two-dimensional grid
    static const unsigned int dimBlockX =
        mhd::parameters::KernelRunParameters::blockSizeX;
    static const unsigned int dimBlockY =
        mhd::parameters::KernelRunParameters::blockSizeY;

    static const unsigned int dimGridX =
        mhd::parameters::KernelRunParameters::gridSizeX;
    static const unsigned int dimGridY =
        mhd::parameters::KernelRunParameters::gridSizeY;

public:
    template <GridType Type, typename Kernel, typename... TArgs>
    static void call(Kernel kernel, TArgs... args) {
        dim3 dimBlock;
        dim3 dimGrid;

        switch (Type) {
            case Half:
                dimBlock = dim3(dimBlockX, dimBlockY, 1);
                dimGrid = dim3(dimGridX, dimGridY / 2, 1);
                break;
            case Full:
                dimBlock = dim3(dimBlockX, dimBlockY, 1);
                dimGrid = dim3(dimGridX, dimGridY, 1);
                break;
            case Linear:
                dimBlock = dim3(dimBlockLinear, 1, 1);
                dimGrid = dim3(dimGridLinear, 1, 1);
                break;
        }

        kernel<<<dimGrid, dimBlock>>>(args...);

#ifdef DEBUG
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
#endif  // DEBUG
    }
};

#define CallKernel KernelCaller::call<Half>
#define CallKernelFull KernelCaller::call<Full>
#define CallKernelLinear KernelCaller::call<Linear>
