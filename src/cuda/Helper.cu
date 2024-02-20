#include "Helper.cuh"

#include <iostream>
#include <string>

#include "Buffers.cuh"
#include "HelperKernels.cuh"

namespace mhd {
double Helper::maxRotorAmplitude(const GpuComplexBuffer2D& field) {
    _caller.call(DiffByX_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel, DoubleBufferA().data(),
                       DoubleBufferB().data());
    CpuLinearBufferX().copyFromDevice(DoubleBufferB().data());

    _caller.call(DiffByY_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel, DoubleBufferA().data(),
                       DoubleBufferB().data());
    CpuLinearBufferY().copyFromDevice(DoubleBufferB().data());

    double v = 0.;
    for (unsigned int i = 0; i < _configs._linearLength; i++) {
        v = (fabs(CpuLinearBufferX()[i]) > v) ? fabs(CpuLinearBufferX()[i]) : v;
        v = (fabs(CpuLinearBufferY()[i]) > v) ? fabs(CpuLinearBufferY()[i]) : v;
    }

    return _configs._lambda * v;
}

double Helper::calcEnergy(const GpuComplexBuffer2D& field) {
    _caller.call(DiffByX_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());

    _caller.call(DiffByY_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferB());

    _caller.callFull(EnergyTransform_kernel, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);
    _caller.callLinear(EnergyIntegrate_kernel, DoubleBufferC().data(),
                       DoubleBufferA().data());

    _fields._cpuLinearBufferX.copyFromDevice(DoubleBufferA().data());

    double e = 0.0;
    for (unsigned int i = 0; i < _configs._linearLength; i++) {
        e += CpuLinearBufferX()[i];
    }

    return (4. * M_PI * M_PI) * _configs._lambda * e;
}

void Helper::normallize(GpuComplexBuffer2D& field, double ratio) {
    _caller.call(MultComplex_kernel, field.data(), field.length(), ratio,
                 field.data());
}

Helper::Helper(const Configs& configs)
    : _configs(configs),
      _transformator(configs._gridLength),
      _fields(configs._gridLength, configs._linearLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength),
      _writer(configs._gridLength),
      _currents(configs._outputStep, configs._outputStart) {}

const GpuDoubleBuffer2D& Helper::getVorticity() {
    _caller.call(MultComplex_kernel, Vorticity().data(), Vorticity().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getStream() {
    _caller.call(MultComplex_kernel, Stream().data(), Stream().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getCurrent() {
    _caller.call(MultComplex_kernel, Current().data(), Current().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getPotential() {
    _caller.call(MultComplex_kernel, Potential().data(), Potential().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

GpuComplexBuffer2D& Helper::Vorticity() {
    return _fields._vorticity;
}

GpuComplexBuffer2D& Helper::Stream() {
    return _fields._stream;
}

GpuComplexBuffer2D& Helper::Current() {
    return _fields._current;
}

GpuComplexBuffer2D& Helper::Potential() {
    return _fields._potential;
}

GpuComplexBuffer2D& Helper::OldVorticity() {
    return _fields._oldVorticity;
}

GpuComplexBuffer2D& Helper::OldPotential() {
    return _fields._oldPotential;
}

GpuComplexBuffer2D& Helper::RightPart() {
    return _fields._rightPart;
}

GpuComplexBuffer2D& Helper::ComplexBuffer() {
    return _fields._complexBuffer;
}

GpuDoubleBuffer2D& Helper::DoubleBufferA() {
    return _fields._doubleBufferA;
}

GpuDoubleBuffer2D& Helper::DoubleBufferB() {
    return _fields._doubleBufferB;
}

GpuDoubleBuffer2D& Helper::DoubleBufferC() {
    return _fields._doubleBufferC;
}

CpuDoubleBuffer1D& Helper::CpuLinearBufferX() {
    return _fields._cpuLinearBufferX;
}

CpuDoubleBuffer1D& Helper::CpuLinearBufferY() {
    return _fields._cpuLinearBufferY;
}

CpuDoubleBuffer2D& Helper::Output() {
    return _fields._output;
}

const GpuComplexBuffer2D& Helper::Vorticity() const {
    return _fields._vorticity;
}

const GpuComplexBuffer2D& Helper::Stream() const {
    return _fields._stream;
}

const GpuComplexBuffer2D& Helper::Current() const {
    return _fields._current;
}

const GpuComplexBuffer2D& Helper::Potential() const {
    return _fields._potential;
}

const GpuComplexBuffer2D& Helper::OldVorticity() const {
    return _fields._oldVorticity;
}

const GpuComplexBuffer2D& Helper::OldPotential() const {
    return _fields._oldPotential;
}

const GpuComplexBuffer2D& Helper::RightPart() const {
    return _fields._rightPart;
}

const GpuComplexBuffer2D& Helper::ComplexBuffer() const {
    return _fields._complexBuffer;
}

const GpuDoubleBuffer2D& Helper::DoubleBufferA() const {
    return _fields._doubleBufferA;
}

const GpuDoubleBuffer2D& Helper::DoubleBufferB() const {
    return _fields._doubleBufferB;
}

const GpuDoubleBuffer2D& Helper::DoubleBufferC() const {
    return _fields._doubleBufferC;
}

const CpuDoubleBuffer1D& Helper::CpuLinearBufferX() const {
    return _fields._cpuLinearBufferX;
}

const CpuDoubleBuffer1D& Helper::CpuLinearBufferY() const {
    return _fields._cpuLinearBufferY;
}

const CpuDoubleBuffer2D& Helper::Output() const {
    return _fields._output;
}

void Helper::updateEnergies() {
    _currents.kineticEnergy = calcEnergy(Stream());
    _currents.magneticEnergy = calcEnergy(Potential());
}

void Helper::updateStream() {
    _caller.call(MinusInverseLaplasOperator_kernel, Vorticity().data(),
                 Vorticity().length(), Stream().data());
}

void Helper::updateVorticity() {
    _caller.call(MinusLaplasOperator_kernel, Stream().data(), Stream().length(),
                 Vorticity().data());
}

void Helper::updatePotential() {
    _caller.call(InverseLaplasOperator_kernel, Current().data(),
                 Current().length(), Potential().data());
}

void Helper::updateCurrent() {
    _caller.call(LaplasOperator_kernel, Potential().data(),
                 Potential().length(), Current().data());
}

void Helper::timeStep() {
    _currents.time += _currents.timeStep;
    _currents.stepNumber++;
}

void Helper::saveOldFields() {
    Vorticity().copyToDevice(OldVorticity().data());
    Potential().copyToDevice(OldPotential().data());
}

void Helper::updateTimeStep() {
    double cfl = _configs._cfl;
    double gridStep = _configs._gridStep;
    double maxTimeStep = _configs._maxTimeStep;

    _currents.maxVelocityField = maxRotorAmplitude(Stream());
    _currents.maxMagneticField = maxRotorAmplitude(Potential());

    _currents.timeStep =
        (_currents.maxMagneticField > _currents.maxVelocityField)
            ? cfl * gridStep / _currents.maxMagneticField
            : cfl * gridStep / _currents.maxVelocityField;
    if (_currents.timeStep > maxTimeStep)
        _currents.timeStep = maxTimeStep;
}

void Helper::fillNormally(unsigned long seed, int offset) {
    GpuStateBuffer2D state(_configs._gridLength);
    double ratio, energy;

    _caller.call(FillStates, state.data(), state.length(), offset);
    _caller.call(FillNormally_kernel, Stream().data(), state.data(),
                 Stream().length(), _configs._averageWN);
    energy = calcEnergy(Stream());
    ratio = (energy > 0) ? std::sqrt(_configs._kineticEnergy / energy) : 1.;
    normallize(Stream(), ratio);
    updateVorticity();

    _caller.call(FillStates, state.data(), state.length(), offset + offset);
    _caller.call(FillNormally_kernel, Potential().data(), state.data(),
                 Potential().length(), _configs._averageWN);
    energy = calcEnergy(Potential());
    ratio = (energy > 0) ? std::sqrt(_configs._magneticEnergy / energy) : 1.;
    normallize(Potential(), ratio);
    updateCurrent();
}

void Helper::saveData(const std::filesystem::path& outputDir) {
    if ((_currents.time >= _currents.outputTime) &&
        (_currents.time <= _configs._outputStop)) {
        _writer.saveVorticity(getVorticity().data(), outputDir,
                              _currents.outputNumber);
        _writer.saveCurrent(getCurrent().data(), outputDir,
                            _currents.outputNumber);
        _writer.saveStream(getStream().data(), outputDir,
                           _currents.outputNumber);
        _writer.savePotential(getPotential().data(), outputDir,
                              _currents.outputNumber);

        _writer.saveCurrents(_currents, outputDir, _currents.outputNumber);

        _currents.outputNumber++;
        _currents.outputTime += _currents.outputStep;

        printCurrents();
    }
}

void Helper::saveDataLite(const std::filesystem::path& outputDir) {
    if ((_currents.time >= _currents.outputTime) &&
        (_currents.time < _configs._outputStop)) {
        _writer.saveStream(getStream().data(), outputDir,
                           _currents.outputNumber);
        _writer.savePotential(getPotential().data(), outputDir,
                              _currents.outputNumber);

        _writer.saveCurrents(_currents, outputDir, _currents.outputNumber);

        _currents.outputNumber++;
        _currents.outputTime += _currents.outputStep;

        printCurrents();
    }
}

void Helper::printCurrents() {
    if (_currents.stepNumber == 0) {
        std::cout << std::left;
        std::cout << std::setw(6) << "Step:";
        std::cout << std::setw(6) << "Time:";
        std::cout << std::right;
        std::cout << std::setw(10) << "dTime:";
        std::cout << std::setw(11) << "Ekin:";
        std::cout << std::setw(12) << "Emag:";
        std::cout << std::setw(12) << "Esum:";
        std::cout << std::endl;

        std::cout
            << "_____________________________________________________________"
            << std::endl;
    }
    std::cout << " ";
    std::cout << std::left;
    std::cout << std::setw(8) << _currents.stepNumber;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << std::setw(6) << _currents.time;

    std::cout << std::fixed << std::setprecision(4) << std::right;
    std::cout << std::setw(10) << _currents.timeStep;
    std::cout << std::setw(12) << _currents.kineticEnergy;
    std::cout << std::setw(12) << _currents.magneticEnergy;
    std::cout << std::setw(12)
              << _currents.kineticEnergy + _currents.magneticEnergy
              << std::endl;
}

bool Helper::shouldContinue() {
    return _currents.time <= _configs._time;
}

void Helper::printField(const GpuComplexBuffer2D& field,
                        const std::string& message) {
    GpuComplexBuffer2D tmpComplex(field.length());
    tmpComplex.copyFromDevice(field.data());

    GpuDoubleBuffer2D tmpDouble(field.length());
    FastFourierTransformator transformator(field.length());
    transformator.inverseFFT(tmpComplex.data(), tmpDouble.data());
    _caller.callFull(MultDouble_kernel, tmpDouble.data(), tmpDouble.length(),
                     _configs._lambda, tmpDouble.data());

    Output().copyFromDevice(tmpDouble.data());

    std::cout << "Field " << message << ":	" << Output()[0] << "	"
              << Output()[1] << "	" << Output()[2] << "	" << Output()[3]
              << "	" << Output()[4] << std::endl
              << std::endl;
}

template <bool IsNormalized>
void Helper::printField(const GpuDoubleBuffer2D& field,
                        const std::string& message) {
    GpuDoubleBuffer2D tmpDouble(field.length());
    tmpDouble.copyFromDevice(field.data());

    double lambda = (IsNormalized) ? 1. : _configs._lambda;
    CallKernelFull(MultDouble_kernel, tmpDouble.data(), tmpDouble.length(),
                   lambda, tmpDouble.data());

    Output().copyFromDevice(tmpDouble.data());

    std::cout << "Field " << message << ":	" << Output()[0] << "	"
              << Output()[1] << "	" << Output()[2] << "	" << Output()[3]
              << "	" << Output()[4] << std::endl
              << std::endl;
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