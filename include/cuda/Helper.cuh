#pragma once

#include <filesystem>

#include "../Configs.h"

#include "Buffers.cuh"
#include "FastFourierTransformator.cuh"
#include "KernelCaller.cuh"

namespace mhd {
class Helper {
protected:
    // Simulation configurations
    Configs _configs;

    // Auxiliary classes
    FastFourierTransformator _transformator;
    KernelCaller _caller;
    // Steams
    cudaStream_t _stream1, _stream2;

    struct Fields {
        // Calculated Fields
        GpuComplexBuffer2D _vorticity;
        GpuComplexBuffer2D _stream;
        GpuComplexBuffer2D _current;
        GpuComplexBuffer2D _potential;

        // Auxiliary Fields: Equations
        GpuComplexBuffer2D _oldVorticity, _oldPotential, _rightPart;

        // Auxiliary Fields: Temporary GPU
        GpuComplexBuffer2D _complexBuffer;
        GpuDoubleBuffer2D _doubleBufferA, _doubleBufferB, _doubleBufferC;

        // Output buffer: Temporary CPU
        CpuDoubleBuffer2D _output;

        Fields(unsigned int gridLength)
            : _vorticity(gridLength),
              _stream(gridLength),
              _current(gridLength),
              _potential(gridLength),
              _oldVorticity(gridLength),
              _oldPotential(gridLength),
              _rightPart(gridLength),
              _complexBuffer(gridLength),
              _doubleBufferA(gridLength),
              _doubleBufferB(gridLength),
              _doubleBufferC(gridLength),
              _output(gridLength) {}
    } _fields;

    void normallize(GpuComplexBuffer2D& field, double ratio);
    double maxRotorAmplitude(const GpuComplexBuffer2D& field);
    double calcEnergy(cudaStream_t& stream, const GpuComplexBuffer2D& field);

public:
    Currents _currents;

    Helper(const Configs& configs);
    ~Helper();

    cudaStream_t& getStream1() { return _stream1; }
    cudaStream_t& getStream2() { return _stream2; }

    void fillNormally(unsigned long seed, int offset = 1);

    void updateStream();
    void updateVorticity();
    void updatePotential();
    void updateCurrent();

    void timeStep();
    void updateTimeStep();
    void updateEnergies();

    void saveOldFields();

    const GpuDoubleBuffer2D& getVorticity();
    const GpuDoubleBuffer2D& getStream();
    const GpuDoubleBuffer2D& getCurrent();
    const GpuDoubleBuffer2D& getPotential();

    GpuComplexBuffer2D& vorticity() { return _fields._vorticity; }
    GpuComplexBuffer2D& stream() { return _fields._stream; }
    GpuComplexBuffer2D& current() { return _fields._current; }
    GpuComplexBuffer2D& potential() { return _fields._potential; }
    GpuComplexBuffer2D& oldVorticity() { return _fields._oldVorticity; }
    GpuComplexBuffer2D& oldPotential() { return _fields._oldPotential; }
    GpuComplexBuffer2D& rightPart() { return _fields._rightPart; }
    GpuComplexBuffer2D& complexBuffer() { return _fields._complexBuffer; }
    GpuDoubleBuffer2D& doubleBufferA() { return _fields._doubleBufferA; }
    GpuDoubleBuffer2D& doubleBufferB() { return _fields._doubleBufferB; }
    GpuDoubleBuffer2D& doubleBufferC() { return _fields._doubleBufferC; }
    //CpuDoubleBuffer2D& Output();

    bool shouldContinue();
};

class CudaTimeCounter {
private:
    cudaEvent_t _start, _stop;
    float time;

public:
    CudaTimeCounter();
    CudaTimeCounter(const std::string& startMessage);
    ~CudaTimeCounter();

    void restart(const std::string& startMessage);
    void done(const std::string& stopMessage);

    void start();
    void stop();

    float getTime();
};
}  // namespace mhd
