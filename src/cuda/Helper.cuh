#pragma once

#include <filesystem>

#include "../params.h"

#include "Buffers.cuh"
#include "FastFourierTransformator.cuh"
#include "Writer.cuh"

namespace mhd {

class Helper {
protected:
    //Simulation parameters
    parameters::SimulationParameters _simulationParams;
    parameters::CurrentParameters _params;

    //Auxiliary classes
    FastFourierTransformator _transformator;
    Writer _writer;

    //Calculated Fields
    GpuComplexBuffer2D _vorticity;
    GpuComplexBuffer2D _stream;
    GpuComplexBuffer2D _current;
    GpuComplexBuffer2D _potential;

    //Auxiliary Fields: Equations
    GpuComplexBuffer2D _oldVorticity, _oldPotential, _rightPart;

    //Auxiliary Fields: Temporary
    GpuComplexBuffer2D _complexBuffer;
    GpuDoubleBuffer2D _doubleBufferA, _doubleBufferB, _doubleBufferC;

    //Auxiliary Fields: Temporary
    CpuDoubleBuffer1D _cpuLinearBufferX;
    CpuDoubleBuffer1D _cpuLinearBufferY;

    void normallize(GpuComplexBuffer2D& field, double ratio);
    double maxRotorAmplitude(const GpuComplexBuffer2D& field);
    double calcEnergy(const GpuComplexBuffer2D& field);

public:
    Helper();

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

    void saveData(const std::filesystem::path& outputDir);

    void saveDataLite(const std::filesystem::path& outputDir);

    void printCurrentParams();

    bool shouldContinue();
};

class CudaTimeCounter {
private:
    cudaEvent_t _start, _stop;
    float time;

public:
    CudaTimeCounter();
    ~CudaTimeCounter();

    void start();
    void stop();

    float getTime();
};
}  // namespace mhd
