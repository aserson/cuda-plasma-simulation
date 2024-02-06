#pragma once

#include "cuda_runtime.h"

#include "Params.h"

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
