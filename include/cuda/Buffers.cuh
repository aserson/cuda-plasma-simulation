#pragma once

#include <cufft.h>
#include <curand_kernel.h>

namespace mhd {

class CpuDoubleBuffer2D {
private:
    double* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    CpuDoubleBuffer2D();
    CpuDoubleBuffer2D(unsigned int sideLength);
    ~CpuDoubleBuffer2D();

    double* data() { return _buffer; }
    const double* data() const { return _buffer; }

    double& operator[](unsigned int index) { return _buffer[index]; }
    const double& operator[](unsigned int index) const {
        return _buffer[index];
    }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }

    void clear();
    void copyToDevice(cudaStream_t& stream, double* dst) const;
    void copyFromDevice(cudaStream_t& stream, const double* src);
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

    double* data() { return _buffer; }
    const double* data() const { return _buffer; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }
    unsigned int fullLength() const { return _sideLength * _sideLength; }

    void clear(cudaStream_t& stream);
    void copyToHost(cudaStream_t& stream, double* dst) const;
    void copyFromHost(cudaStream_t& stream, const double* src);
    void copyToDevice(cudaStream_t& stream, double* dst) const;
    void copyFromDevice(cudaStream_t& stream, const double* src);
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

    cufftDoubleComplex* data() { return _buffer; }
    const cufftDoubleComplex* data() const { return _buffer; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }

    void clear(cudaStream_t& stream);
    void copyToHost(cudaStream_t& stream, cufftDoubleComplex* dst) const;
    void copyFromHost(cudaStream_t& stream, const cufftDoubleComplex* src);
    void copyToDevice(cudaStream_t& stream, cufftDoubleComplex* dst) const;
    void copyFromDevice(cudaStream_t& stream, const cufftDoubleComplex* src);
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

    curandState* data() { return _buffer; }
    const curandState* data() const { return _buffer; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }
};
}  // namespace mhd

namespace graphics {
class CpuFloatBuffer {
private:
    float* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    CpuFloatBuffer();
    CpuFloatBuffer(unsigned int bufferLength);
    ~CpuFloatBuffer();

    float* data() { return _buffer; }
    const float* data() const { return _buffer; }

    float& operator[](unsigned int index) { return _buffer[index]; }
    const float& operator[](unsigned int index) const { return _buffer[index]; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _bufferLength; }

    void clear();
    void copyToDevice(cudaStream_t& stream, float* dst) const;
    void copyFromDevice(cudaStream_t& stream, const float* src);
};

class GpuFloatBuffer {
private:
    float* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    GpuFloatBuffer();
    GpuFloatBuffer(unsigned int bufferLength);
    ~GpuFloatBuffer();

    float* data() { return _buffer; }
    const float* data() const { return _buffer; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _bufferLength; }

    void clear(cudaStream_t& stream);
    void copyToDevice(cudaStream_t& stream, float* dst) const;
    void copyFromDevice(cudaStream_t& stream, const float* src);
};

class CpuPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    CpuPixelBuffer2D();
    CpuPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~CpuPixelBuffer2D();

    unsigned char* data() { return _buffer; }
    const unsigned char* data() const { return _buffer; }

    unsigned char& operator[](unsigned int index) { return _buffer[index]; }
    const unsigned char& operator[](unsigned int index) const {
        return _buffer[index];
    }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }

    void clear();
    void copyToDevice(cudaStream_t& stream, unsigned char* dst) const;
    void copyFromDevice(cudaStream_t& stream, const unsigned char* src);
};

class GpuPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    GpuPixelBuffer2D();
    GpuPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~GpuPixelBuffer2D();

    unsigned char* data() { return _buffer; }
    const unsigned char* data() const { return _buffer; }

    unsigned int size() const { return _bufferSize; }
    unsigned int length() const { return _sideLength; }

    void clear(cudaStream_t& stream);
    void copyToHost(cudaStream_t& stream, unsigned char* dst) const;
    void copyFromHost(cudaStream_t& stream, const unsigned char* src);
};

class CpuColorMapBuffer {
private:
    unsigned char _buffer[256 * 3];
    unsigned int _length = 256;
    unsigned int _channels = 3;

public:
    unsigned char* data() { return _buffer; }
    const unsigned char* data() const { return _buffer; }

    unsigned char& red(unsigned int index) { return _buffer[3 * index + 0]; }
    unsigned char& green(unsigned int index) { return _buffer[3 * index + 1]; }
    unsigned char& blue(unsigned int index) { return _buffer[3 * index + 2]; }

    const unsigned char& red(unsigned int index) const {
        return _buffer[3 * index + 0];
    }
    const unsigned char& green(unsigned int index) const {
        return _buffer[3 * index + 1];
    }
    const unsigned char& blue(unsigned int index) const {
        return _buffer[3 * index + 2];
    }

    unsigned int size() const { return sizeof(_buffer); }
    unsigned int length() const { return _length; }
};
}  // namespace graphics
