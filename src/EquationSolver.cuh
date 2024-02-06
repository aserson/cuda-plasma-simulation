#pragma once

#include "Helper.cuh"
#include "Jacobian.cuh"
#include "SimpleFunctions.cuh"

__global__ void FirstRigthPart_kernel(cufftDoubleComplex* w,
                                      cufftDoubleComplex* jacobian,
                                      cufftDoubleComplex* rightPart) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;
    double nu = mhd::parameters::EquationCoefficients::nu;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

    rightPart[idx].x = jacobian[idx].x - nu * value * w[idx].x;
    rightPart[idx].y = jacobian[idx].y - nu * value * w[idx].y;
}

__global__ void SecondRigthPart_kernel(cufftDoubleComplex* jacobian,
                                       cufftDoubleComplex* rightPart) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    rightPart[idx].x += jacobian[idx].x;
    rightPart[idx].y += jacobian[idx].y;
}

__global__ void ThirdRigthPart_kernel(cufftDoubleComplex* a,
                                      cufftDoubleComplex* jacobian,
                                      cufftDoubleComplex* rightPart) {
    unsigned int gridLength = mhd::parameters::SimulationParameters::gridLength;
    double eta = mhd::parameters::EquationCoefficients::eta;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

    rightPart[idx].x = jacobian[idx].x - eta * value * a[idx].x;
    rightPart[idx].y = jacobian[idx].y - eta * value * a[idx].y;
}

void KineticRightSideCalc(GpuComplexBuffer& rightPart,
                          const mhd::FastFourierTransformator& transformator,
                          CalculatedFields& calc, AuxiliaryFields& aux) {
    calcJacobian(calc._streamFunction, calc._vorticity, transformator, aux);
    CallKernel(FirstRigthPart_kernel, calc._vorticity.data(),
               aux._complexTmp.data(), rightPart.data());

    calcJacobian(calc._magneticPotential, calc._current, transformator, aux);
    CallKernel(SecondRigthPart_kernel, aux._complexTmp.data(),
               rightPart.data());
}

void MagneticRightSideCalc(GpuComplexBuffer& rightPart,
                           const mhd::FastFourierTransformator& transformator,
                           CalculatedFields& calc, AuxiliaryFields& aux) {
    calcJacobian(calc._streamFunction, calc._magneticPotential, transformator,
                 aux);
    CallKernel(ThirdRigthPart_kernel, calc._magneticPotential.data(),
               aux._complexTmp.data(), rightPart.data());
}

__global__ void TimeScheme_kernel(cufftDoubleComplex* field,
                                  const cufftDoubleComplex* oldField,
                                  const cufftDoubleComplex* rightPart,
                                  unsigned int gridLength, double dt,
                                  double value = 1.) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (gridLength / 2 + 1) * x + y;

    field[idx].x = oldField[idx].x + value * rightPart[idx].x * dt;
    field[idx].y = oldField[idx].y + value * rightPart[idx].y * dt;
}

void TimeScheme(GpuComplexBuffer& field, const GpuComplexBuffer& oldField,
                const GpuComplexBuffer& rightPart,
                mhd::parameters::CurrentParameters& params) {
    CallKernel(TimeScheme_kernel, field.data(), oldField.data(),
               rightPart.data(), field.length(), params.timeStep, 1.);
}

void UpdateStreamFunction(CalculatedFields& calc) {
    CallKernel(MinusInverseLaplasOperator_kernel, calc._vorticity.data(),
               calc._vorticity.length(), calc._streamFunction.data());
}

void UpdateVorticity(CalculatedFields& calc) {
    CallKernel(MinusLaplasOperator_kernel, calc._streamFunction.data(),
               calc._streamFunction.length(), calc._vorticity.data());
}

void UpdateMagneticPotential(CalculatedFields& calc) {
    CallKernel(InverseLaplasOperator_kernel, calc._current.data(),
               calc._current.length(), calc._magneticPotential.data());
}

void UpdateCurrent(CalculatedFields& calc) {
    CallKernel(LaplasOperator_kernel, calc._magneticPotential.data(),
               calc._magneticPotential.length(), calc._current.data());
}
