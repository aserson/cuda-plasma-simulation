#pragma once

#include <cufft.h>

#include "KernelCaller.cuh"

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

        CUDA_CALL(
            cudaHostAlloc((void**)&_buffer, _bytes, cudaHostAllocDefault));
    }

    ~CpuBuffer() { CUDA_CALL(cudaFreeHost(_buffer)); }

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
typedef mhd::CpuBuffer<double, true, false> CpuDoubleBuffer;

typedef mhd::GpuBuffer<cufftDoubleComplex, true> GpuComplexBuffer;
typedef mhd::GpuBuffer<double, false> GpuDoubleBuffer;
