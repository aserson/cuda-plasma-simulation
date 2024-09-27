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
    unsigned int _dimBlock;
    unsigned int _dimGrid;

public:
    KernelCaller(unsigned int gridLength, unsigned int dimBlockX,
                 unsigned int dimBlockY, unsigned int dimBlock) {
        _dimBlockX = dimBlockX;
        _dimBlockY = dimBlockY;
        _dimGridX = gridLength / dimBlockX;
        _dimGridY = gridLength / dimBlockY;

        _dimBlock = dimBlock;
        _dimGrid = gridLength * gridLength / dimBlock;
    }

    template <typename Kernel, typename... TArgs>
    void call(cudaStream_t& stream, Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callFull(cudaStream_t& stream, Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    unsigned int callLinear(cudaStream_t& stream, unsigned int size,
                            Kernel kernel, TArgs... args);

    template <typename Kernel>
    void callReduction(cudaStream_t& stream, unsigned int size, Kernel kernel,
                       const double* input, double* output);

    template <typename Kernel, typename... TArgs>
    void callKernel(cudaStream_t& stream, Kernel kernel, dim3 dimBlock,
                    dim3 dimGrid, size_t sharedSize, TArgs... args);
};

template <typename Kernel, typename... TArgs>
void KernelCaller::call(cudaStream_t& stream, Kernel kernel, TArgs... args) {
    dim3 dimBlock(_dimBlockX, _dimBlockY);
    dim3 dimGrid(_dimGridX, _dimGridY / 2);
    size_t sharedSize = 0;

    callKernel(stream, kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callFull(cudaStream_t& stream, Kernel kernel,
                            TArgs... args) {
    dim3 dimBlock(_dimBlockY, _dimBlockY);
    dim3 dimGrid(_dimGridY, _dimGridY);
    size_t sharedSize = 0;

    callKernel(stream, kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
unsigned int KernelCaller::callLinear(cudaStream_t& stream, unsigned int size,
                                      Kernel kernel, TArgs... args) {
    unsigned int blocks = (size + _dimBlock - 1) / _dimBlock;
    size_t sharedSize = _dimBlock * sizeof(float);

    callKernel(stream, kernel, _dimBlock, blocks, sharedSize, args...);

    return blocks;
}

template <typename Kernel>
void KernelCaller::callReduction(cudaStream_t& stream, unsigned int size,
                                 Kernel kernel, const double* input,
                                 double* output) {

    unsigned int taskSize =
        callLinear(stream, size, kernel, input, size, output);

    while (taskSize > 1) {
        taskSize =
            callLinear(stream, taskSize, kernel, output, taskSize, output);
    }
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callKernel(cudaStream_t& stream, Kernel kernel,
                              dim3 dimBlock, dim3 dimGrid, size_t sharedSize,
                              TArgs... args) {
#ifdef __CUDACC__
    kernel<<<dimGrid, dimBlock, sharedSize, stream>>>(args...);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
#endif  // __CUDACC__
}