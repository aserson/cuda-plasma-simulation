#include "cuda/Buffers.cuh"

#include "cuda/KernelCaller.cuh"

namespace mhd {

// CpuDoubleBuffer2D functions definitions
CpuDoubleBuffer2D::CpuDoubleBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

CpuDoubleBuffer2D::CpuDoubleBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize = _sideLength * _sideLength * sizeof(double);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuDoubleBuffer2D::~CpuDoubleBuffer2D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

void CpuDoubleBuffer2D::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuDoubleBuffer2D::copyToDevice(cudaStream_t& stream, double* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void CpuDoubleBuffer2D::copyFromDevice(cudaStream_t& stream,
                                       const double* src) {
    if (src != nullptr) {
        CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize,
                                  cudaMemcpyDeviceToHost, stream));
    } else {
        std::cerr << "Copying Error from device: Source buffer is nullptr!"
                  << std::endl;
    }
}

// GpuDoubleBuffer2D functions definitions
GpuDoubleBuffer2D::GpuDoubleBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

GpuDoubleBuffer2D::GpuDoubleBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize = _sideLength * _sideLength * sizeof(double);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuDoubleBuffer2D::~GpuDoubleBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}

void GpuDoubleBuffer2D::clear(cudaStream_t& stream) {
    CUDA_CALL(cudaMemsetAsync(_buffer, 0x0, _bufferSize, stream));
}

void GpuDoubleBuffer2D::copyToHost(cudaStream_t& stream, double* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

void GpuDoubleBuffer2D::copyFromHost(cudaStream_t& stream, const double* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void GpuDoubleBuffer2D::copyToDevice(cudaStream_t& stream, double* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize,
                              cudaMemcpyDeviceToDevice, stream));
}

void GpuDoubleBuffer2D::copyFromDevice(cudaStream_t& stream,
                                       const double* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize,
                              cudaMemcpyDeviceToDevice, stream));
}

// GpuComplexBuffer2D functions definitions
GpuComplexBuffer2D::GpuComplexBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

GpuComplexBuffer2D::GpuComplexBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize =
        (_sideLength / 2 + 1) * _sideLength * sizeof(cufftDoubleComplex);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuComplexBuffer2D::~GpuComplexBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}

void GpuComplexBuffer2D::clear(cudaStream_t& stream) {
    CUDA_CALL(cudaMemsetAsync(_buffer, 0x0, _bufferSize, stream));
}

void GpuComplexBuffer2D::copyToHost(cudaStream_t& stream,
                                    cufftDoubleComplex* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

void GpuComplexBuffer2D::copyFromHost(cudaStream_t& stream,
                                      const cufftDoubleComplex* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void GpuComplexBuffer2D::copyToDevice(cudaStream_t& stream,
                                      cufftDoubleComplex* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize,
                              cudaMemcpyDeviceToDevice, stream));
}

void GpuComplexBuffer2D::copyFromDevice(cudaStream_t& stream,
                                        const cufftDoubleComplex* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize,
                              cudaMemcpyDeviceToDevice, stream));
}

// GpuStateBuffer2D functions definitions

GpuStateBuffer2D::GpuStateBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

GpuStateBuffer2D::GpuStateBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize = (_sideLength / 2 + 1) * _sideLength * sizeof(curandState);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuStateBuffer2D::~GpuStateBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}
}  // namespace mhd

namespace graphics {

// CpuFloatBuffer functions definitions

CpuFloatBuffer::CpuFloatBuffer()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

CpuFloatBuffer::CpuFloatBuffer(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(float);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuFloatBuffer::~CpuFloatBuffer() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

void CpuFloatBuffer::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuFloatBuffer::copyToDevice(cudaStream_t& stream, float* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void CpuFloatBuffer::copyFromDevice(cudaStream_t& stream, const float* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

// GpuFloatBuffer functions definitions

GpuFloatBuffer::GpuFloatBuffer()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

GpuFloatBuffer::GpuFloatBuffer(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(float);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuFloatBuffer::~GpuFloatBuffer() {
    cudaFree(_buffer);
}

void GpuFloatBuffer::clear(cudaStream_t& stream) {
    CUDA_CALL(cudaMemsetAsync(_buffer, 0x0, _bufferSize, stream));
}

void GpuFloatBuffer::copyToDevice(cudaStream_t& stream, float* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void GpuFloatBuffer::copyFromDevice(cudaStream_t& stream, const float* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

// CpuPixelBuffer2D functions definitions

CpuPixelBuffer2D::CpuPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

CpuPixelBuffer2D::CpuPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuPixelBuffer2D::~CpuPixelBuffer2D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

void CpuPixelBuffer2D::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuPixelBuffer2D::copyToDevice(cudaStream_t& stream,
                                    unsigned char* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}

void CpuPixelBuffer2D::copyFromDevice(cudaStream_t& stream,
                                      const unsigned char* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

// GPUPixelBuffer2D functions definitions

GpuPixelBuffer2D::GpuPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

GpuPixelBuffer2D::GpuPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuPixelBuffer2D::~GpuPixelBuffer2D() {
    cudaFree(_buffer);
}

void GpuPixelBuffer2D::clear(cudaStream_t& stream) {
    CUDA_CALL(cudaMemsetAsync(_buffer, 0x0, _bufferSize, stream));
}

void GpuPixelBuffer2D::copyToHost(cudaStream_t& stream,
                                  unsigned char* dst) const {
    CUDA_CALL(cudaMemcpyAsync(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost,
                              stream));
}

void GpuPixelBuffer2D::copyFromHost(cudaStream_t& stream,
                                    const unsigned char* src) {
    CUDA_CALL(cudaMemcpyAsync(_buffer, src, _bufferSize, cudaMemcpyHostToDevice,
                              stream));
}
}  // namespace graphics