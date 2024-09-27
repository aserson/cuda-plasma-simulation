#include "cuda/Helper.cuh"

#include <iostream>
#include <string>

#include "cuda/Buffers.cuh"
#include "cuda/HelperKernels.cuh"

namespace mhd {
double Helper::maxRotorAmplitude(const GpuComplexBuffer2D& field) {
    _caller.call(_stream1, DiffByX_kernel, field.data(), _configs._gridLength,
                 complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());

    _caller.callReduction(_stream1, doubleBufferA().fullLength(), Max_kernel,
                          doubleBufferA().data(), doubleBufferB().data());
    double v1 = 0.;
    CUDA_CALL(cudaMemcpyAsync(&v1, doubleBufferB().data(), sizeof(double),
                              cudaMemcpyDeviceToHost, _stream1));

    _caller.call(_stream1, DiffByY_kernel, field.data(), _configs._gridLength,
                 complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());

    _caller.callReduction(_stream1, doubleBufferA().fullLength(), Max_kernel,
                          doubleBufferA().data(), doubleBufferB().data());
    double v2 = 0.;
    CUDA_CALL(cudaMemcpyAsync(&v1, doubleBufferB().data(), sizeof(double),
                              cudaMemcpyDeviceToHost, _stream1));

    return _configs._lambda * std::max(v1, v2);
}

double Helper::calcEnergy(cudaStream_t& stream,
                          const GpuComplexBuffer2D& field) {
    _caller.call(stream, DiffByX_kernel, field.data(), _configs._gridLength,
                 complexBuffer().data());
    _transformator.inverse(stream, complexBuffer(), doubleBufferA());

    _caller.call(stream, DiffByY_kernel, field.data(), _configs._gridLength,
                 complexBuffer().data());
    _transformator.inverse(stream, complexBuffer(), doubleBufferB());

    _caller.callFull(stream, EnergyTransform_kernel, doubleBufferA().data(),
                     doubleBufferB().data(), doubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);

    _caller.callReduction(_stream1, doubleBufferC().fullLength(),
                          EnergyIntegrate_kernel, doubleBufferC().data(),
                          doubleBufferA().data());
    double energy = 0.;
    CUDA_CALL(cudaMemcpyAsync(&energy, doubleBufferA().data(), sizeof(double),
                              cudaMemcpyDeviceToHost, stream));

    return (4. * M_PI * M_PI) * _configs._lambda * energy;
}

void Helper::normallize(GpuComplexBuffer2D& field, double ratio) {
    _caller.call(_stream1, MultComplex_kernel, field.data(), field.length(),
                 ratio, field.data());
}

Helper::Helper(const Configs& configs)
    : _configs(configs),
      _transformator(configs._gridLength),
      _fields(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._dimBlock) {

    CUDA_CALL(cudaStreamCreate(&_stream1));
    CUDA_CALL(cudaStreamCreate(&_stream2));
}

Helper::~Helper() {
    CUDA_CALL(cudaStreamDestroy(_stream1));
    CUDA_CALL(cudaStreamDestroy(_stream2));
}

const GpuDoubleBuffer2D& Helper::getVorticity() {
    _caller.call(_stream1, MultComplex_kernel, vorticity().data(),
                 vorticity().length(), _configs._lambda,
                 complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());
    return doubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getStream() {
    _caller.call(_stream1, MultComplex_kernel, stream().data(),
                 stream().length(), _configs._lambda, complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());
    return doubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getCurrent() {
    _caller.call(_stream1, MultComplex_kernel, current().data(),
                 current().length(), _configs._lambda, complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());
    return doubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getPotential() {
    _caller.call(_stream1, MultComplex_kernel, potential().data(),
                 potential().length(), _configs._lambda,
                 complexBuffer().data());
    _transformator.inverse(_stream1, complexBuffer(), doubleBufferA());
    return doubleBufferA();
}

void Helper::updateEnergies() {
    _currents.kineticEnergy = calcEnergy(_stream1, stream());
    _currents.magneticEnergy = calcEnergy(_stream1, potential());
}

void Helper::updateStream() {
    _caller.call(_stream1, MinusInverseLaplasOperator_kernel,
                 vorticity().data(), vorticity().length(), stream().data());
}

void Helper::updateVorticity() {
    _caller.call(_stream1, MinusLaplasOperator_kernel, stream().data(),
                 stream().length(), vorticity().data());
}

void Helper::updatePotential() {
    _caller.call(_stream1, InverseLaplasOperator_kernel, current().data(),
                 current().length(), potential().data());
}

void Helper::updateCurrent() {
    _caller.call(_stream1, LaplasOperator_kernel, potential().data(),
                 potential().length(), current().data());
}

void Helper::timeStep() {
    _currents.time += _currents.timeStep;
    _currents.stepNumber++;
}

void Helper::saveOldFields() {
    cudaStreamSynchronize(_stream1);
    vorticity().copyToDevice(_stream1, oldVorticity().data());
    potential().copyToDevice(_stream1, oldPotential().data());
}

void Helper::updateTimeStep() {
    double cfl = _configs._cfl;
    double gridStep = _configs._gridStep;
    double maxTimeStep = _configs._maxTimeStep;

    _currents.maxVelocityField = maxRotorAmplitude(stream());
    _currents.maxMagneticField = maxRotorAmplitude(potential());

    _currents.timeStep =
        cfl * gridStep /
        fmax(_currents.maxVelocityField, _currents.maxMagneticField);

    _currents.timeStep = fmin(_currents.timeStep, maxTimeStep);
}

void Helper::fillNormally(unsigned long seed, int offset) {
    GpuStateBuffer2D state(_configs._gridLength);
    double ratio, energy;

    _caller.call(_stream1, FillStates, state.data(), state.length(), offset);
    _caller.call(_stream1, FillNormally_kernel, stream().data(), state.data(),
                 stream().length(), _configs._averageWN);
    energy = calcEnergy(_stream1, stream());
    ratio = (energy > 0) ? std::sqrt(_configs._kineticEnergy / energy) : 1.;
    normallize(stream(), ratio);
    updateVorticity();

    _caller.call(_stream1, FillStates, state.data(), state.length(),
                 offset + offset);
    _caller.call(_stream1, FillNormally_kernel, potential().data(),
                 state.data(), potential().length(), _configs._averageWN);
    energy = calcEnergy(_stream1, potential());
    ratio = (energy > 0) ? std::sqrt(_configs._magneticEnergy / energy) : 1.;
    normallize(potential(), ratio);
    updateCurrent();
}

bool Helper::shouldContinue() {
    return _currents.time <= _configs._time;
}

CudaTimeCounter::CudaTimeCounter() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    start();
}

CudaTimeCounter::CudaTimeCounter(const std::string& startMessage)
    : CudaTimeCounter::CudaTimeCounter() {
    std::cout << startMessage;
    start();
}

CudaTimeCounter::~CudaTimeCounter() {
    cudaEventDestroy(_stop);
    cudaEventDestroy(_start);
}

void CudaTimeCounter::restart(const std::string& startMessage) {
    std::cout << startMessage;
    start();
}

void CudaTimeCounter::done(const std::string& stopMessage) {
    stop();
    std::cout << stopMessage << getTime() << std::endl;
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