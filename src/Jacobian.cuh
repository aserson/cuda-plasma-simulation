#pragma once

#include "Helper.cuh"
#include "KernelCaller.cuh"

enum JacobianType { First = 1, Second, Third };

__global__ void DealaliasingDiffByX_kernel(const cufftDoubleComplex* input,
                                           cufftDoubleComplex* output) {
    const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;
    const unsigned int dealWN =
        mhd::parameters::SimulationParameters::dealaliasingWaveNumber;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        output[idx].x = -(double)x * input[idx].y;
        output[idx].y = (double)x * input[idx].x;
    } else {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

__global__ void DealaliasingDiffByY_kernel(const cufftDoubleComplex* input,
                                           cufftDoubleComplex* output) {
    const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;
    const unsigned int dealWN =
        mhd::parameters::SimulationParameters::dealaliasingWaveNumber;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        output[idx].x = -(double)y * input[idx].y;
        output[idx].y = (double)y * input[idx].x;
    } else {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

__global__ void JacobianFirstPart_kernel(double* inputA, double* inputB,
                                         double* output) {
    const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;
    const double lambda = mhd::parameters::SimulationParameters::lambda;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    output[idx] = inputA[idx] * inputB[idx] * lambda * lambda;
}

__global__ void JacobianSecondPart_kernel(double* inputA, double* inputB,
                                          double* output) {
    const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;
    const double lambda = mhd::parameters::SimulationParameters::lambda;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    output[idx] = output[idx] - inputA[idx] * inputB[idx] * lambda * lambda;
}

__global__ void Dealaliasing_kernel(cufftDoubleComplex* output) {
    const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;
    const unsigned int dealWN =
        mhd::parameters::SimulationParameters::dealaliasingWaveNumber;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) >= dealWN) || (abs(y) >= dealWN)) {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

void calcJacobian(const GpuComplexBuffer& leftField,
                  const GpuComplexBuffer& rightField,
                  const mhd::FastFourierTransformator& transformator,
                  AuxiliaryFields& aux) {
    CallKernel(DealaliasingDiffByX_kernel, leftField.data(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());

    CallKernel(DealaliasingDiffByY_kernel, rightField.data(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpB.data());

    KernelCaller::call<GridType::Full>(
        JacobianFirstPart_kernel, aux._doubleTmpA.data(),
        aux._doubleTmpB.data(), aux._doubleTmpC.data());

    CallKernel(DealaliasingDiffByY_kernel, leftField.data(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpA.data());

    CallKernel(DealaliasingDiffByX_kernel, rightField.data(),
               aux._complexTmp.data());
    transformator.inverseFFT(aux._complexTmp.data(), aux._doubleTmpB.data());

    KernelCaller::call<GridType::Full>(
        JacobianSecondPart_kernel, aux._doubleTmpA.data(),
        aux._doubleTmpB.data(), aux._doubleTmpC.data());

    transformator.forwardFFT(aux._doubleTmpC.data(), aux._complexTmp.data());

    CallKernel(Dealaliasing_kernel, aux._complexTmp.data());
}
