#pragma once
#include "cuda_runtime.h"

#include <iostream>
#include <string>

#include <cufft.h>

#include "Params.h"
#include "SimpleFunctions.cuh"

#define CUDA_CALL(result) \
    checkCudaError(result, __FUNCTION__, __FILE__, __LINE__)

void checkCudaError(cudaError_t result, const std::string& functionName,
                    const std::string& fileName, int lineNumber) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << functionName << " at " << fileName
                  << ":" << lineNumber << " - " << cudaGetErrorString(result)
                  << std::endl;
    }
}

namespace mhd {
template <typename TData, bool IsTwoDimensional = true, bool IsHalf = true>
class CpuBuffer {
   private:
    TData* _buffer;
    unsigned int _gridLengthX;
    unsigned int _gridLengthY;
    unsigned int _gridLengthFull;
    unsigned int _bytes;

   public:
    CpuBuffer()
        : _gridLengthX(0),
          _gridLengthY(0),
          _gridLengthFull(0),
          _bytes(0),
          _buffer(nullptr) {}

    CpuBuffer(unsigned int gridLength) : _gridLengthX(gridLength) {
        if constexpr (IsTwoDimensional) {
            if constexpr (IsHalf) {
                _gridLengthY = gridLength / 2 + 1;
            } else {
                _gridLengthY = gridLength;
            }
        } else {
            _gridLengthY = 1U;
        }

        _gridLengthFull = _gridLengthX * _gridLengthY;
        _bytes = _gridLengthFull * sizeof(TData);

        _buffer = new TData[_gridLengthX * _gridLengthY];
    }

    ~CpuBuffer() { delete[] _buffer; }

    TData* data() { return _buffer; }

    const TData* data() const { return _buffer; }

    TData& operator[](unsigned int index) { return _buffer[index]; }

    const TData& operator[](unsigned int index) const { return _buffer[index]; }

    unsigned int size() const { return _bytes; }

    unsigned int length() const { return _gridLengthX; }

    unsigned int fullLength() const { return _gridLengthFull; }

    bool clear() { memset(_buffer, 0x0, _bytes); }

    void copyToDevice(TData* dst) const {
        CUDA_CALL(cudaMemcpy(dst, _buffer, _bytes, cudaMemcpyHostToDevice));
    }

    void copyFromDevice(const TData* src) {
        CUDA_CALL(cudaMemcpy(_buffer, src, _bytes, cudaMemcpyDeviceToHost));
    }
};

template <typename TData, bool IsHalf = true>
class GpuBuffer {
   private:
    TData* _buffer;
    unsigned int _gridLengthX;
    unsigned int _gridLengthY;
    unsigned int _gridLengthFull;
    unsigned int _bytes;

   public:
    GpuBuffer()
        : _gridLengthX(0),
          _gridLengthY(0),
          _gridLengthFull(0),
          _bytes(0),
          _buffer(nullptr) {}

    GpuBuffer(unsigned int gridLength) : _gridLengthX(gridLength) {
        if constexpr (IsHalf) {
            _gridLengthY = gridLength / 2 + 1;
        } else {
            _gridLengthY = gridLength;
        }

        _gridLengthFull = _gridLengthX * _gridLengthY;
        _bytes = _gridLengthFull * sizeof(TData);

        CUDA_CALL(cudaMalloc((void**)&_buffer, _bytes));
    }

    ~GpuBuffer() {
        if (_buffer != nullptr) {
            CUDA_CALL(cudaFree(_buffer));
        }
    }

    TData* data() { return _buffer; }

    const TData* data() const { return _buffer; }

    unsigned int size() const { return _bytes; }

    unsigned int length() const { return _gridLengthX; }

    unsigned int fullLength() const { return _gridLengthFull; }

    void clear() { CUDA_CALL(cudaMemset(_buffer, 0x0, _bytes)); }

    void copyToHost(TData* dst) const {
        CUDA_CALL(cudaMemcpy(dst, _buffer, _bytes, cudaMemcpyDeviceToHost));
    }

    void copyFromHost(const TData* src) {
        CUDA_CALL(cudaMemcpy(_buffer, src, _bytes, cudaMemcpyHostToDevice));
    }

    void copyToDevice(TData* dst) const {
        CUDA_CALL(cudaMemcpy(dst, _buffer, _bytes, cudaMemcpyDeviceToDevice));
    }

    void copyFromDevice(const TData* src) {
        CUDA_CALL(cudaMemcpy(_buffer, src, _bytes, cudaMemcpyDeviceToDevice));
    }
};
}  // namespace mhd

typedef mhd::CpuBuffer<double, false> CpuLinearDoubleBuffer;
typedef mhd::CpuBuffer<double, true, true> CpuHalfDoubleBuffer;
typedef mhd::CpuBuffer<double, true, false> CpuFullDoubleBuffer;

typedef mhd::GpuBuffer<cufftDoubleComplex, true> GpuComplexBuffer;
typedef mhd::GpuBuffer<double, false> GpuDoubleBuffer;

namespace mhd {
class FastFourierTransformator {
   private:
    cufftHandle planD2Z, planZ2D;

   public:
    FastFourierTransformator(unsigned int gridLength) {
        cufftPlan2d(&planD2Z, gridLength, gridLength, CUFFT_D2Z);
        cufftPlan2d(&planZ2D, gridLength, gridLength, CUFFT_Z2D);
    }

    ~FastFourierTransformator() {
        cufftDestroy(planD2Z);
        cufftDestroy(planZ2D);
    }

    void forwardFFT(double* input, cufftDoubleComplex* output) const {
        cufftExecD2Z(planD2Z, input, output);
    }

    void inverseFFT(cufftDoubleComplex* input, double* output) const {
        cufftExecZ2D(planZ2D, input, output);
    }

    void forward(GpuDoubleBuffer& input, GpuComplexBuffer& output) const {
        cufftExecD2Z(planD2Z, input.data(), output.data());
    }

    void inverse(GpuComplexBuffer& input, GpuDoubleBuffer& output) const {
        cufftExecZ2D(planZ2D, input.data(), output.data());
    }
};
}  // namespace mhd

typedef mhd::FastFourierTransformator FFTransformator;

struct CalculatedFields {
    static const unsigned int gridLength =
        mhd::parameters::SimulationParameters::gridLength;

    GpuComplexBuffer _vorticity;
    GpuComplexBuffer _streamFunction;

    GpuComplexBuffer _current;
    GpuComplexBuffer _magneticPotential;

    CalculatedFields(unsigned int gridLength)
        : _vorticity(gridLength),
          _streamFunction(gridLength),
          _current(gridLength),
          _magneticPotential(gridLength) {}
};

struct AuxiliaryFields {
    GpuComplexBuffer _oldOne;
    GpuComplexBuffer _oldTwo;

    GpuComplexBuffer _rightPart;

    GpuComplexBuffer _complexTmp;
    GpuDoubleBuffer _doubleTmpA, _doubleTmpB, _doubleTmpC;

    CpuLinearDoubleBuffer _bufferX;
    CpuLinearDoubleBuffer _bufferY;

    CpuFullDoubleBuffer _bufferOut;

    AuxiliaryFields(unsigned int gridLength, unsigned int linearGridLength)
        : _oldOne(gridLength),
          _oldTwo(gridLength),
          _rightPart(gridLength),
          _complexTmp(gridLength),
          _doubleTmpA(gridLength),
          _doubleTmpB(gridLength),
          _doubleTmpC(gridLength),
          _bufferX(linearGridLength),
          _bufferY(linearGridLength),
          _bufferOut(gridLength) {}
};

namespace mhd {
__global__ void Max_kernel(double* input, double* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double a[parameters::KernelRunParameters::blockSizeLinear];
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

__global__ void EnergyTransform_kernel(double* velocityX, double* velocityY,
                                       double* energy) {
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

class Helper {
   private:
    parameters::SimulationParameters _simulationParams;
    parameters::KernelRunParameters _kernelRunParams;

    FFTransformator _transformator;

    CalculatedFields _calc;

    GpuComplexBuffer _oldVorticity, _oldMagneticPotential, _rightPart;

    GpuComplexBuffer _complexBuffer;
    GpuDoubleBuffer _doubleBufferA, _doubleBufferB, _doubleBufferC;

    CpuLinearDoubleBuffer _cpuLinearBufferX;
    CpuLinearDoubleBuffer _cpuLinearBufferY;
    CpuFullDoubleBuffer _cpuOutBuffer;

    double maxRotorAmplitude(const GpuComplexBuffer& field) {
        unsigned int gridLength = _simulationParams.gridLength;
        unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
        double lambda = _simulationParams.lambda;

        CallKernel(DiffByX_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        CallKernelLinear(Max_kernel, _doubleBufferA.data(),
                         _doubleBufferB.data());
        _cpuLinearBufferX.copyFromDevice(_doubleBufferB.data());

        CallKernel(DiffByY_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);
        CallKernelLinear(Max_kernel, _doubleBufferA.data(),
                         _doubleBufferB.data());
        _cpuLinearBufferY.copyFromDevice(_doubleBufferB.data());

        double v = 0.0;
        for (unsigned int i = 0; i < gridLengthLinear; i++) {
            v = (fabs(_cpuLinearBufferX[i]) > v) ? fabs(_cpuLinearBufferX[i])
                                                 : v;
            v = (fabs(_cpuLinearBufferY[i]) > v) ? fabs(_cpuLinearBufferY[i])
                                                 : v;
        }

        return v * lambda;
    }

    double calcEnergy(const GpuComplexBuffer& field) {
        unsigned int gridLength = _simulationParams.gridLength;
        unsigned int gridLengthLinear = _simulationParams.gridLengthLinear;
        double lambda = _simulationParams.lambda;

        CallKernel(DiffByX_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferA);

        CallKernel(DiffByY_kernel, field.data(), gridLength,
                   _complexBuffer.data());
        _transformator.inverse(_complexBuffer, _doubleBufferB);

        CallKernelFull(EnergyTransform_kernel, _doubleBufferA.data(),
                       _doubleBufferB.data(), _doubleBufferC.data());
        CallKernelLinear(EnergyIntegrate_kernel, _doubleBufferC.data(),
                         _doubleBufferA.data());

        _cpuLinearBufferX.copyFromDevice(_doubleBufferA.data());

        double e = 0.0;
        for (int i = 0; i < gridLengthLinear; i++) {
            e += _cpuLinearBufferX[i];
        }

        return (4. * M_PI * M_PI) * lambda * e;
    }

   public:
    mhd::parameters::CurrentParameters _currentParams;

    Helper()
        : _transformator(_simulationParams.gridLength),
          _calc(_simulationParams.gridLength),
          _oldVorticity(_simulationParams.gridLength),
          _oldMagneticPotential(_simulationParams.gridLength),
          _rightPart(_simulationParams.gridLength),
          _complexBuffer(_simulationParams.gridLength),
          _doubleBufferA(_simulationParams.gridLength),
          _doubleBufferB(_simulationParams.gridLength),
          _doubleBufferC(_simulationParams.gridLength),
          _cpuLinearBufferX(_simulationParams.gridLengthLinear),
          _cpuLinearBufferY(_simulationParams.gridLengthLinear),
          _cpuOutBuffer(_simulationParams.gridLength) {}

    void updateTimeStep() {
        double cfl = mhd::parameters::SimulationParameters::cft;
        double gridStep = mhd::parameters::SimulationParameters::gridStep;
        double maxTimeStep = mhd::parameters::SimulationParameters::maxTimeStep;

        _currentParams.maxVelocityField =
            maxRotorAmplitude(_calc._streamFunction);
        _currentParams.maxMagneticField =
            maxRotorAmplitude(_calc._magneticPotential);
        _currentParams.timeStep =
            (_currentParams.maxMagneticField > _currentParams.maxVelocityField)
                ? cfl * gridStep / _currentParams.maxMagneticField
                : cfl * gridStep / _currentParams.maxVelocityField;
        if (_currentParams.timeStep > maxTimeStep)
            _currentParams.timeStep = maxTimeStep;
    }

    void updateEnergies() {
        _currentParams.kineticEnergy = calcEnergy(_calc._streamFunction);
        _currentParams.magneticEnergy = calcEnergy(_calc._magneticPotential);
    }
};
}  // namespace mhd