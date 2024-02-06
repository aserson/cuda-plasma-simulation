#pragma once

#include "Helper.cuh"
#include "KernelCaller.cuh"
#include "SimpleFunctions.cuh"

__global__ void Max_kernel(double* input, double* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double a[mhd::parameters::KernelRunParameters::blockSizeLinear];
    a[threadIdx.x] = fabs(input[idx]);

    __syncthreads();

    for (unsigned int i = 2; i < blockDim.x + 1; i = i * 2) {
        if (idx % i == 0) {
            a[threadIdx.x] = (a[threadIdx.x + i / 2] > a[threadIdx.x])
                                 ? a[threadIdx.x + i / 2]
                                 : a[threadIdx.x];
        }
        __syncthreads();
    }

    output[blockIdx.x] = a[0];
}

double Max(const mhd::FastFourierTransformator& transformator,
           AuxiliaryFields& aux, const GpuComplexBuffer& field) {
    double lambda = mhd::parameters::SimulationParameters::lambda;

    CallKernel(DiffByX_kernel, field.data(), field.length(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());
    CallKernelLinear(Max_kernel, aux._doubleTmpA.data(),
                     aux._doubleTmpB.data());
    cudaMemcpy(aux._bufferX.data(), aux._doubleTmpB.data(), aux._bufferX.size(),
               cudaMemcpyDeviceToHost);

    CallKernel(DiffByY_kernel, field.data(), field.length(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());
    CallKernelLinear(Max_kernel, aux._doubleTmpA.data(),
                     aux._doubleTmpB.data());
    cudaMemcpy(aux._bufferY.data(), aux._doubleTmpB.data(), aux._bufferY.size(),
               cudaMemcpyDeviceToHost);

    double v = 0.0;
    for (unsigned int i = 0; i < aux._bufferX.length(); i++) {
        v = (fabs(aux._bufferX[i]) > v) ? fabs(aux._bufferX[i]) : v;
        v = (fabs(aux._bufferY[i]) > v) ? fabs(aux._bufferY[i]) : v;
    }

    return v * lambda;
}

__global__ void EnergyTransform_kernelDirty(double* velocityX,
                                            double* velocityY, double* energy) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;
    double lambda = mhd::parameters::SimulationParameters::lambda;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    velocityX[idx] *= lambda;
    velocityY[idx] *= lambda;

    energy[idx] =
        (velocityX[idx] * velocityX[idx] + velocityY[idx] * velocityY[idx]) /
        2.;
}

__global__ void EnergyTransform_kernel(double* velocity, double* energy) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;
    double lambda = mhd::parameters::SimulationParameters::lambda;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    energy[idx] = (velocity[idx] * velocity[idx] * lambda * lambda) / 2.;
}

__global__ void EnergyIntegrate_kernel(double* field, double* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double a[mhd::parameters::KernelRunParameters::blockSizeLinear];
    a[threadIdx.x] = field[idx];

    __syncthreads();

    for (unsigned int i = 2; i < blockDim.x + 1; i = i * 2) {
        if (idx % i == 0) {
            a[threadIdx.x] = a[threadIdx.x] + a[threadIdx.x + i / 2];
        }
        __syncthreads();
    }

    sum[blockIdx.x] = a[0];
}

double Energy(const mhd::FastFourierTransformator& transformator,
              AuxiliaryFields& aux, const GpuComplexBuffer& field) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;
    double lambda = mhd::parameters::SimulationParameters::lambda;

    CallKernel(DiffByX_kernel, field.data(), gridLength,
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());
    CallKernelFull(EnergyTransform_kernel, aux._doubleTmpA.data(),
                   aux._doubleTmpA.data());
    CallKernelLinear(EnergyIntegrate_kernel, aux._doubleTmpA.data(),
                     aux._doubleTmpB.data());
    aux._bufferX.copyFromDevice(aux._doubleTmpB.data());

    CallKernel(DiffByY_kernel, field.data(), gridLength,
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());
    CallKernelFull(EnergyTransform_kernel, aux._doubleTmpA.data(),
                   aux._doubleTmpA.data());
    CallKernelLinear(EnergyIntegrate_kernel, aux._doubleTmpA.data(),
                     aux._doubleTmpB.data());
    aux._bufferY.copyFromDevice(aux._doubleTmpB.data());

    double e = 0.0;
    for (int i = 0; i < aux._bufferX.length(); i++) {
        e += aux._bufferX[i] + aux._bufferY[i];
    }

    return e * (4. * M_PI * M_PI) * lambda;
}

double EnergyDirty(const mhd::FastFourierTransformator& transformator,
                   AuxiliaryFields& aux, const GpuComplexBuffer& field) {
    double lambda = mhd::parameters::SimulationParameters::lambda;

    CallKernel(DiffByX_kernel, field.data(), field.length(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());

    CallKernel(DiffByY_kernel, field.data(), field.length(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpB.data());

    CallKernelFull(EnergyTransform_kernelDirty, aux._doubleTmpA.data(),
                   aux._doubleTmpB.data(), aux._doubleTmpC.data());
    CallKernelLinear(EnergyIntegrate_kernel, aux._doubleTmpC.data(),
                     aux._doubleTmpA.data());

    aux._bufferX.copyFromDevice(aux._doubleTmpA.data());

    double e = 0.0;
    for (int i = 0; i < aux._bufferX.length(); i++) {
        e += aux._bufferX[i];
    }

    return (4. * M_PI * M_PI) * lambda * e;
}
