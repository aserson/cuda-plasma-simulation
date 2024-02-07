#include "Helper.cuh"

#include <iostream>
#include <string>

#include "Buffers.cuh"
#include "HelperKernels.cuh"
#include "KernelCaller.cuh"

namespace mhd {
double Helper::maxRotorAmplitude(const GpuComplexBuffer2D& field) {
    unsigned int gridLength = _simulationParams.gridLength;
    unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
    const unsigned int sharedMemLength = _simulationParams.bufferLengthLinear;
    double lambda = _simulationParams.lambda;

    CallKernel(DiffByX_kernel, field.data(), gridLength, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    CallKernelLinear(Max_kernel<sharedMemLength>, _doubleBufferA.data(),
                     _doubleBufferB.data());
    _cpuLinearBufferX.copyFromDevice(_doubleBufferB.data());

    CallKernel(DiffByY_kernel, field.data(), gridLength, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    CallKernelLinear(Max_kernel<sharedMemLength>, _doubleBufferA.data(),
                     _doubleBufferB.data());
    _cpuLinearBufferY.copyFromDevice(_doubleBufferB.data());

    double v = 0.0;
    for (unsigned int i = 0; i < gridLengthLinear; i++) {
        v = (fabs(_cpuLinearBufferX[i]) > v) ? fabs(_cpuLinearBufferX[i]) : v;
        v = (fabs(_cpuLinearBufferY[i]) > v) ? fabs(_cpuLinearBufferY[i]) : v;
    }

    return lambda * v;
}

double Helper::calcEnergy(const GpuComplexBuffer2D& field) {
    unsigned int gridLength = _simulationParams.gridLength;
    unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
    const unsigned int sharedMemLength = _simulationParams.bufferLengthLinear;
    double lambda = _simulationParams.lambda;

    CallKernel(DiffByX_kernel, field.data(), gridLength, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);

    CallKernel(DiffByY_kernel, field.data(), gridLength, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferB);

    CallKernelFull(EnergyTransform_kernel, _doubleBufferA.data(),
                   _doubleBufferB.data(), _doubleBufferC.data());
    CallKernelLinear(EnergyIntegrate_kernel<sharedMemLength>,
                     _doubleBufferC.data(), _doubleBufferA.data());

    _cpuLinearBufferX.copyFromDevice(_doubleBufferA.data());

    double e = 0.0;
    for (unsigned int i = 0; i < gridLengthLinear; i++) {
        e += _cpuLinearBufferX[i];
    }

    return (4. * M_PI * M_PI) * lambda * e;
}

void Helper::normallize(GpuComplexBuffer2D& field, double ratio) {
    CallKernel(MultComplex_kernel, field.data(), field.length(), ratio,
               field.data());
}

Helper::Helper()
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

const GpuDoubleBuffer2D& Helper::getVorticity() {
    CallKernel(MultComplex_kernel, _vorticity.data(), _vorticity.length(),
               _simulationParams.lambda, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    return _doubleBufferA;
}

const GpuDoubleBuffer2D& Helper::getStream() {
    CallKernel(MultComplex_kernel, _stream.data(), _stream.length(),
               _simulationParams.lambda, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    return _doubleBufferA;
}

const GpuDoubleBuffer2D& Helper::getCurrent() {
    CallKernel(MultComplex_kernel, _current.data(), _current.length(),
               _simulationParams.lambda, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    return _doubleBufferA;
}

const GpuDoubleBuffer2D& Helper::getPotential() {
    CallKernel(MultComplex_kernel, _potential.data(), _potential.length(),
               _simulationParams.lambda, _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);
    return _doubleBufferA;
}

void Helper::updateEnergies() {
    _params.kineticEnergy = calcEnergy(_stream);
    _params.magneticEnergy = calcEnergy(_potential);
}

void Helper::updateStream() {
    CallKernel(MinusInverseLaplasOperator_kernel, _vorticity.data(),
               _vorticity.length(), _stream.data());
}

void Helper::updateVorticity() {
    CallKernel(MinusLaplasOperator_kernel, _stream.data(), _stream.length(),
               _vorticity.data());
}

void Helper::updatePotential() {
    CallKernel(InverseLaplasOperator_kernel, _current.data(), _current.length(),
               _potential.data());
}

void Helper::updateCurrent() {
    CallKernel(LaplasOperator_kernel, _potential.data(), _potential.length(),
               _current.data());
}

void Helper::timeStep() {
    _params.time += _params.timeStep;
    _params.stepNumber++;
}

void Helper::saveOldFields() {
    _vorticity.copyToDevice(_oldVorticity.data());
    _potential.copyToDevice(_oldPotential.data());
}

void Helper::updateTimeStep() {
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

void Helper::fillNormally(unsigned long seed, int offset) {
    mhd::parameters::InitialCondition initParams;
    double ratio;

    CallKernel(FillNormally_kernel, _stream.data(), _stream.length(),
               initParams.averageWN, seed);
    ratio = std::sqrt(initParams.kineticEnergy / calcEnergy(_stream));
    normallize(_stream, ratio);
    updateVorticity();

    CallKernel(FillNormally_kernel, _potential.data(), _potential.length(),
               initParams.averageWN, seed + offset);
    ratio = std::sqrt(initParams.magneticEnergy / calcEnergy(_potential));
    normallize(_potential, ratio);
    updateCurrent();
}

void Helper::saveData(const std::filesystem::path& outputDir) {
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

void Helper::saveDataLite(const std::filesystem::path& outputDir) {
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

void Helper::printCurrentParams() {
    printf("%f\t%d\t%f\t%f\t%f\t%f\n", _params.time, _params.stepNumber,
           _params.timeStep, _params.kineticEnergy, _params.magneticEnergy,
           _params.kineticEnergy + _params.magneticEnergy);
}

bool Helper::shouldContinue() {
    return _params.time < _simulationParams.time;
}

CudaTimeCounter::CudaTimeCounter() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
}

CudaTimeCounter::~CudaTimeCounter() {
    cudaEventDestroy(_stop);
    cudaEventDestroy(_start);
}

void CudaTimeCounter::start() {
    cudaEventRecord(_start, 0);
}

void CudaTimeCounter::stop() {
    cudaEventRecord(_stop, 0);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&time, _start, _stop);
}

float CudaTimeCounter::getTime() {
    return time / 1000;
}
}  // namespace mhd