#pragma once

#include "Helper.cuh"
#include "KernelCaller.cuh"

__global__ void MultDouble_kernel(const double* input, unsigned int gridLength,
                                  double value, double* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    output[idx] = input[idx] * value;
}

__global__ void MultComplex_kernel(const cufftDoubleComplex* input,
                                   unsigned int gridLength, double value,
                                   cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = input[idx].x * value;
    output[idx].y = input[idx].y * value;
}

__global__ void DiffByX_kernel(const cufftDoubleComplex* input,
                               unsigned int gridLength,
                               cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }

    output[idx].x = -(double)x * input[idx].y;
    output[idx].y = (double)x * input[idx].x;
}

__global__ void DiffByY_kernel(const cufftDoubleComplex* input,
                               unsigned int gridLength,
                               cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = -(double)y * input[idx].y;
    output[idx].y = (double)y * input[idx].x;
}

__global__ void LaplasOperator_kernel(const cufftDoubleComplex* input,
                                      unsigned int gridLength,
                                      cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = -(double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ void MinusLaplasOperator_kernel(const cufftDoubleComplex* input,
                                           unsigned int gridLength,
                                           cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ void InverseLaplasOperator_kernel(const cufftDoubleComplex* input,
                                             unsigned int gridLength,
                                             cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (idx == 0) ? 0.0 : (-1.) / (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ void MinusInverseLaplasOperator_kernel(
    const cufftDoubleComplex* input, unsigned int gridLength,
    cufftDoubleComplex* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (idx == 0) ? 0.0 : 1. / (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}
