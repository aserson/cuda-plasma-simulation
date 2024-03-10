#pragma once

#include "cuda_runtime.h"

#include <iostream>

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

inline void printFunctionName(const char* functionName) {
    std::cout << "Running cuda kernel " << functionName << "..." << std::endl;
}

inline void checkCudaKernelError() {
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

class KernelCaller {
private:
    // Two-dimensional grid
    unsigned int _dimBlockX;
    unsigned int _dimBlockY;
    unsigned int _dimGridX;
    unsigned int _dimGridY;

    // One-dimensional grid
    unsigned int _dimBlockLinear;
    unsigned int _dimGridLinear;
    size_t _sharedSize;

public:
    KernelCaller(unsigned int gridLength, unsigned int dimBlockX,
                 unsigned int dimBlockY, unsigned int sharedLength) {
        _dimBlockX = dimBlockX;
        _dimBlockY = dimBlockY;
        _dimGridX = gridLength / dimBlockX;
        _dimGridY = gridLength / dimBlockY;

        _dimBlockLinear = sharedLength;
        _dimGridLinear = gridLength * gridLength / sharedLength;
        _sharedSize = sharedLength * sizeof(double);
    }

    template <typename Kernel, typename... TArgs>
    void call(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callFull(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callLinear(Kernel kernel, TArgs... args);
};

template <typename Kernel, typename... TArgs>
void KernelCaller::call(Kernel kernel, TArgs... args) {
    dim3 dimBlock = dim3(_dimBlockX, _dimBlockY, 1);
    dim3 dimGrid = dim3(_dimGridX, _dimGridY / 2, 1);
    size_t sharedSize = 0;

#ifdef __CUDACC__
    kernel<<<dimGrid, dimBlock, sharedSize>>>(args...);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
#endif  // __CUDACC__
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callFull(Kernel kernel, TArgs... args) {
    dim3 dimBlock = dim3(_dimBlockX, _dimBlockY, 1);
    dim3 dimGrid = dim3(_dimGridX, _dimGridY, 1);
    size_t sharedSize = 0;

#ifdef __CUDACC__
    kernel<<<dimGrid, dimBlock, sharedSize>>>(args...);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
#endif  // __CUDACC__
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callLinear(Kernel kernel, TArgs... args) {
    dim3 dimGrid = dim3(_dimGridLinear, 1, 1);
    dim3 dimBlock = dim3(_dimBlockLinear, 1, 1);
    size_t sharedSize = _sharedSize;

#ifdef __CUDACC__
    kernel<<<dimGrid, dimBlock, sharedSize>>>(args...);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
#endif  // __CUDACC__
}