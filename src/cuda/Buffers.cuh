#pragma once

#include <cufft.h>
#include <curand_kernel.h>

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

class GpuStateBuffer2D {
private:
    curandState* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuStateBuffer2D();
    GpuStateBuffer2D(unsigned int sideLength);
    ~GpuStateBuffer2D();

    curandState* data();
    const curandState* data() const;

    unsigned int size() const;
    unsigned int length() const;
};
}  // namespace mhd

namespace graphics {
class CPUPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    CPUPixelBuffer2D();
    CPUPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~CPUPixelBuffer2D();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(unsigned char* dst) const;
    void copyFromDevice(const unsigned char* src);
};

class GPUPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    GPUPixelBuffer2D();
    GPUPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~GPUPixelBuffer2D();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(unsigned char* dst) const;
    void copyFromHost(const unsigned char* src);
};

class CPUColorMapBuffer {
private:
    unsigned char* _buffer;
    unsigned int _length = 256;
    unsigned int _channels = 3;
    unsigned int _bufferSize;

public:
    CPUColorMapBuffer();
    ~CPUColorMapBuffer();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned char& red(unsigned int index);
    unsigned char& green(unsigned int index);
    unsigned char& blue(unsigned int index);

    const unsigned char& red(unsigned int index) const;
    const unsigned char& green(unsigned int index) const;
    const unsigned char& blue(unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(unsigned char* dst) const;
    void copyFromDevice(const unsigned char* src);
};

class GPUColorMapBuffer {
private:
    unsigned char* _buffer;
    unsigned int _length = 256;
    unsigned int _channels = 3;
    unsigned int _bufferSize;

public:
    GPUColorMapBuffer();
    GPUColorMapBuffer(unsigned int sideLength, unsigned int channels = 3);
    ~GPUColorMapBuffer();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(unsigned char* dst) const;
    void copyFromHost(const unsigned char* src);
};
}  // namespace graphics
