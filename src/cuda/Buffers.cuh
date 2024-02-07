#pragma once

#include <cufft.h>

namespace mhd {
class CpuDoubleBuffer1D {
private:
    double* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    CpuDoubleBuffer1D();
    CpuDoubleBuffer1D(unsigned int bufferLength);
    ~CpuDoubleBuffer1D();

    double* data();
    const double* data() const;

    double& operator[](unsigned int index);
    const double& operator[](unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(double* dst) const;
    void copyFromDevice(const double* src);
};

class CpuDoubleBuffer2D {
private:
    double* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    CpuDoubleBuffer2D();
    CpuDoubleBuffer2D(unsigned int sideLength);
    ~CpuDoubleBuffer2D();

    double* data();
    const double* data() const;

    double& operator[](unsigned int index);
    const double& operator[](unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(double* dst) const;
    void copyFromDevice(const double* src);
};

class GpuDoubleBuffer2D {
private:
    double* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuDoubleBuffer2D();
    GpuDoubleBuffer2D(unsigned int sideLength);
    ~GpuDoubleBuffer2D();

    double* data();
    const double* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(double* dst) const;
    void copyFromHost(const double* src);
    void copyToDevice(double* dst) const;
    void copyFromDevice(const double* src);
};

class GpuComplexBuffer2D {
private:
    cufftDoubleComplex* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuComplexBuffer2D();
    GpuComplexBuffer2D(unsigned int sideLength);
    ~GpuComplexBuffer2D();

    cufftDoubleComplex* data();
    const cufftDoubleComplex* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(cufftDoubleComplex* dst) const;
    void copyFromHost(const cufftDoubleComplex* src);
    void copyToDevice(cufftDoubleComplex* dst) const;
    void copyFromDevice(const cufftDoubleComplex* src);
};
}  // namespace mhd
