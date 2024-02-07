#pragma once
#include "cuda_runtime.h"

#include <iostream>
#include <string>

#include <cufft.h>
#include <curand_kernel.h>

#include "Buffers.cuh"
#include "FastFourierTransformator.cuh"
#include "HelperKernels.cuh"
#include "KernelCaller.cuh"
#include "Params.h"
#include "Writer.cuh"

namespace mhd {
class Helper {
protected:
    //Simulation parameters
    parameters::SimulationParameters _simulationParams;
    parameters::CurrentParameters _params;

    //Auxiliary classes
    FFTransformator _transformator;
    Writer _writer;

    //Calculated Fields
    GpuComplexBuffer _vorticity;
    GpuComplexBuffer _stream;
    GpuComplexBuffer _current;
    GpuComplexBuffer _potential;

    //Auxiliary Fields: Equations
    GpuComplexBuffer _oldVorticity, _oldPotential, _rightPart;

    //Auxiliary Fields: Temporary
    GpuComplexBuffer _complexBuffer;
    GpuDoubleBuffer _doubleBufferA, _doubleBufferB, _doubleBufferC;

    //Auxiliary Fields: Temporary
    CpuLinearDoubleBuffer _cpuLinearBufferX;
    CpuLinearDoubleBuffer _cpuLinearBufferY;

    double maxRotorAmplitude(const GpuComplexBuffer& field) {
        unsigned int gridLength = _simulationParams.gridLength;
        unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
        const unsigned int sharedMemLength =
            _simulationParams.bufferLengthLinear;
        double lambda = _simulationParams.lambda;

        CallKernel(DiffByX_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        CallKernelLinear(Max_kernel<sharedMemLength>, _doubleBufferA.data(),
                         _doubleBufferB.data());
        _cpuLinearBufferX.copyFromDevice(_doubleBufferB.data());

        CallKernel(DiffByY_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        CallKernelLinear(Max_kernel<sharedMemLength>, _doubleBufferA.data(),
                         _doubleBufferB.data());
        _cpuLinearBufferY.copyFromDevice(_doubleBufferB.data());

        double v = 0.0;
        for (unsigned int i = 0; i < gridLengthLinear; i++) {
            v = (fabs(_cpuLinearBufferX[i]) > v) ? fabs(_cpuLinearBufferX[i])
                                                 : v;
            v = (fabs(_cpuLinearBufferY[i]) > v) ? fabs(_cpuLinearBufferY[i])
                                                 : v;
        }

        return lambda * v;
    }

    double calcEnergy(const GpuComplexBuffer& field) {
        unsigned int gridLength = _simulationParams.gridLength;
        unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
        const unsigned int sharedMemLength =
            _simulationParams.bufferLengthLinear;
        double lambda = _simulationParams.lambda;

        CallKernel(DiffByX_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);

        CallKernel(DiffByY_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferB);

        CallKernelFull(EnergyTransform_kernel, _doubleBufferA.data(),
                       _doubleBufferB.data(), _doubleBufferC.data());
        CallKernelLinear(EnergyIntegrate_kernel<sharedMemLength>,
                         _doubleBufferC.data(), _doubleBufferA.data());

        _cpuLinearBufferX.copyFromDevice(_doubleBufferA.data());

        double e = 0.0;
        for (int i = 0; i < gridLengthLinear; i++) {
            e += _cpuLinearBufferX[i];
        }

        return (4. * M_PI * M_PI) * lambda * e;
    }

    void normallize(GpuComplexBuffer& field, double ratio) {
        CallKernel(MultComplex_kernel, field.data(), field.length(), ratio,
                   field.data());
    }

public:
    Helper()
        : _transformator(_simulationParams.gridLength),
          _vorticity(_simulationParams.gridLength),
          _stream(_simulationParams.gridLength),
          _current(_simulationParams.gridLength),
          _potential(_simulationParams.gridLength),
          _oldVorticity(_simulationParams.gridLength),
          _oldPotential(_simulationParams.gridLength),
          _rightPart(_simulationParams.gridLength),
          _complexBuffer(_simulationParams.gridLength),
          _doubleBufferA(_simulationParams.gridLength),
          _doubleBufferB(_simulationParams.gridLength),
          _doubleBufferC(_simulationParams.gridLength),
          _cpuLinearBufferX(_simulationParams.gridLengthLinear),
          _cpuLinearBufferY(_simulationParams.gridLengthLinear) {}

    const GpuDoubleBuffer& getVorticity() {
        CallKernel(MultComplex_kernel, _vorticity.data(), _vorticity.length(),
                   _simulationParams.lambda, _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        return _doubleBufferA;
    }

    const GpuDoubleBuffer& getStream() {
        CallKernel(MultComplex_kernel, _stream.data(), _stream.length(),
                   _simulationParams.lambda, _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        return _doubleBufferA;
    }

    const GpuDoubleBuffer& getCurrent() {
        CallKernel(MultComplex_kernel, _current.data(), _current.length(),
                   _simulationParams.lambda, _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        return _doubleBufferA;
    }

    const GpuDoubleBuffer& getPotential() {
        CallKernel(MultComplex_kernel, _potential.data(), _potential.length(),
                   _simulationParams.lambda, _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        return _doubleBufferA;
    }

    void updateEnergies() {
        _params.kineticEnergy = calcEnergy(_stream);
        _params.magneticEnergy = calcEnergy(_potential);
    }

    void updateStream() {
        CallKernel(MinusInverseLaplasOperator_kernel, _vorticity.data(),
                   _vorticity.length(), _stream.data());
    }

    void updateVorticity() {
        CallKernel(MinusLaplasOperator_kernel, _stream.data(), _stream.length(),
                   _vorticity.data());
    }

    void updatePotential() {
        CallKernel(InverseLaplasOperator_kernel, _current.data(),
                   _current.length(), _potential.data());
    }

    void updateCurrent() {
        CallKernel(LaplasOperator_kernel, _potential.data(),
                   _potential.length(), _current.data());
    }

    void timeStep() {
        _params.time += _params.timeStep;
        _params.stepNumber++;
    }

    void saveOldFields() {
        _vorticity.copyToDevice(_oldVorticity.data());
        _potential.copyToDevice(_oldPotential.data());
    }

    void updateTimeStep() {
        double cfl = mhd::parameters::SimulationParameters::cft;
        double gridStep = mhd::parameters::SimulationParameters::gridStep;
        double maxTimeStep = mhd::parameters::SimulationParameters::maxTimeStep;

        _params.maxVelocityField = maxRotorAmplitude(_stream);
        _params.maxMagneticField = maxRotorAmplitude(_potential);

        _params.timeStep = (_params.maxMagneticField > _params.maxVelocityField)
                               ? cfl * gridStep / _params.maxMagneticField
                               : cfl * gridStep / _params.maxVelocityField;
        if (_params.timeStep > maxTimeStep)
            _params.timeStep = maxTimeStep;
    }

    void fillNormally(unsigned long seed, int offset = 1) {
        mhd::parameters::InitialCondition initParams;
        double ratio;

        CallKernel(FillNormally_kernel, _stream.data(), _stream.length(),
                   initParams.averageWN, seed);
        ratio = std::sqrt(initParams.kineticEnergy / calcEnergy(_stream));
        normallize(_stream, ratio);

        CallKernel(FillNormally_kernel, _potential.data(), _potential.length(),
                   initParams.averageWN, seed);
        ratio = std::sqrt(initParams.magneticEnergy / calcEnergy(_potential));
        normallize(_potential, ratio);
    }

    void saveData(const std::filesystem::path& outputDir) {
        if (_params.time >= _params.timeOut) {

            _writer.saveField<Vorticity>(getVorticity(), outputDir,
                                         _params.stepNumberOut);
            _writer.saveField<Stream>(getStream(), outputDir,
                                      _params.stepNumberOut);
            _writer.saveField<Current>(getCurrent(), outputDir,
                                       _params.stepNumberOut);
            _writer.saveField<Potential>(getPotential(), outputDir,
                                         _params.stepNumberOut);

            _writer.saveCurentParams(_params, outputDir);

            _params.stepNumberOut++;
            _params.timeOut += _params.timeStepOut;
        }
    }

    void saveDataLite(const std::filesystem::path& outputDir) {
        if (_params.time >= _params.timeOut) {

            _writer.saveField<Vorticity>(getVorticity(), outputDir,
                                         _params.stepNumberOut);
            _writer.saveField<Current>(getCurrent(), outputDir,
                                       _params.stepNumberOut);

            _writer.saveCurentParams(_params, outputDir);

            _params.stepNumberOut++;
            _params.timeOut += _params.timeStepOut;
        }
    }

    void printCurrentParams() {
        printf("%f\t%d\t%f\t%f\t%f\t%f\n", _params.time, _params.stepNumber,
               _params.timeStep, _params.kineticEnergy, _params.magneticEnergy,
               _params.kineticEnergy + _params.magneticEnergy);
    }

    bool shouldContinue() { return _params.time < _simulationParams.time; }
};
}  // namespace mhd