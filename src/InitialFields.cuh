#pragma once

#include <cstdlib>

#include <curand_kernel.h>

#include "Helper.cuh"
#include "KernelCaller.cuh"
#include "SimpleFunctions.cuh"

__global__ void FillWithZero_kernel(cufftDoubleComplex* field,
                                    unsigned int gridLength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    field[idx].x = 0.0;
    field[idx].y = 0.0;
}

void FillRandomly(CpuHalfDoubleBuffer& field) {
    for (int idx = 0; idx < field.fullLength(); idx++) {
        field[idx] = 2.f * M_PI * (double)rand() / RAND_MAX;
    }
}

__global__ void FillNormallyWithPhases_kernel(cufftDoubleComplex* f,
                                              unsigned int gridLength,
                                              unsigned int averageWN,
                                              double* phases) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double k = sqrt((double)(x * x + y * y));

    double tmp = (double)(gridLength * gridLength) *
                 exp(-(k * k) / (2.f * (double)(averageWN * averageWN)));

    if (k > 0) {
        f[idx].x = cos(phases[idx]) * tmp / sqrt(k);
        f[idx].y = sin(phases[idx]) * tmp / sqrt(k);
    } else {
        f[idx].x = 0.0;
        f[idx].y = 0.0;
    }
}

__global__ void FillNormally_kernel(cufftDoubleComplex* f,
                                    unsigned int gridLength,
                                    unsigned int averageWN,
                                    unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    curandState state;
    curand_init(seed, idx, 0, &state);
    double phase = 2.f * M_PI * curand_uniform(&state);

    if (x > gridLength / 2)
        x = x - gridLength;
    double k = sqrt((double)(x * x + y * y));

    double value =
        (k > 0) ? (double)(gridLength * gridLength) *
                      exp(-(k * k) / (2.f * (double)(averageWN * averageWN))) /
                      sqrt(k)
                : 0.0;

    f[idx].x = cos(phase) * value;
    f[idx].y = sin(phase) * value;
}

void FillNormally(GpuComplexBuffer& field, unsigned int averageWN,
                  unsigned long seed) {
    CallKernel(FillNormally_kernel, field.data(), field.length(), averageWN,
               seed);
}

void Normallize(GpuComplexBuffer& field, double value) {
    CallKernel(MultComplex_kernel, field.data(), field.length(), value,
               field.data());
}