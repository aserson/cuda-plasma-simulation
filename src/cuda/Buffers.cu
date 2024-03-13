#include "Buffers.cuh"

#include "KernelCaller.cuh"

namespace mhd {

// CpuDoubleBuffer1D functions definitions
CpuDoubleBuffer1D::CpuDoubleBuffer1D()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

CpuDoubleBuffer1D::CpuDoubleBuffer1D(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(double);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuDoubleBuffer1D::~CpuDoubleBuffer1D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

double* CpuDoubleBuffer1D::data() {
    return _buffer;
}

const double* CpuDoubleBuffer1D::data() const {
    return _buffer;
}

double& CpuDoubleBuffer1D::operator[](unsigned int index) {
    return _buffer[index];
}

const double& CpuDoubleBuffer1D::operator[](unsigned int index) const {
    return _buffer[index];
}

unsigned int CpuDoubleBuffer1D::size() const {
    return _bufferSize;
}

unsigned int CpuDoubleBuffer1D::length() const {
    return _bufferLength;
}

void CpuDoubleBuffer1D::clear() {
    memset(_buffer, 0x0, _bufferSize);
}
void CpuDoubleBuffer1D::copyToDevice(double* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuDoubleBuffer1D::copyFromDevice(const double* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

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

double* CpuDoubleBuffer2D::data() {
    return _buffer;
}

const double* CpuDoubleBuffer2D::data() const {
    return _buffer;
}

double& CpuDoubleBuffer2D::operator[](unsigned int index) {
    return _buffer[index];
}

const double& CpuDoubleBuffer2D::operator[](unsigned int index) const {
    return _buffer[index];
}

unsigned int CpuDoubleBuffer2D::size() const {
    return _bufferSize;
}

unsigned int CpuDoubleBuffer2D::length() const {
    return _sideLength;
}

void CpuDoubleBuffer2D::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuDoubleBuffer2D::copyToDevice(double* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuDoubleBuffer2D::copyFromDevice(const double* src) {
    if (src != nullptr) {
        CUDA_CALL(
            cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
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

double* GpuDoubleBuffer2D::data() {
    return _buffer;
}

const double* GpuDoubleBuffer2D::data() const {
    return _buffer;
}

unsigned int GpuDoubleBuffer2D::size() const {
    return _bufferSize;
}

unsigned int GpuDoubleBuffer2D::length() const {
    return _sideLength;
}

void GpuDoubleBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuDoubleBuffer2D::copyToHost(double* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GpuDoubleBuffer2D::copyFromHost(const double* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

void GpuDoubleBuffer2D::copyToDevice(double* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToDevice));
}

void GpuDoubleBuffer2D::copyFromDevice(const double* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToDevice));
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

cufftDoubleComplex* GpuComplexBuffer2D::data() {
    return _buffer;
}

const cufftDoubleComplex* GpuComplexBuffer2D::data() const {
    return _buffer;
}

unsigned int GpuComplexBuffer2D::size() const {
    return _bufferSize;
}

unsigned int GpuComplexBuffer2D::length() const {
    return _sideLength;
}

void GpuComplexBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuComplexBuffer2D::copyToHost(cufftDoubleComplex* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GpuComplexBuffer2D::copyFromHost(const cufftDoubleComplex* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

void GpuComplexBuffer2D::copyToDevice(cufftDoubleComplex* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToDevice));
}

void GpuComplexBuffer2D::copyFromDevice(const cufftDoubleComplex* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToDevice));
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

curandState* GpuStateBuffer2D::data() {
    return _buffer;
}
const curandState* GpuStateBuffer2D::data() const {
    return _buffer;
}
unsigned int GpuStateBuffer2D::size() const {
    return _bufferSize;
}
unsigned int GpuStateBuffer2D::length() const {
    return _sideLength;
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

float* CpuFloatBuffer::data() {
    return _buffer;
}

const float* CpuFloatBuffer::data() const {
    return _buffer;
}

float& CpuFloatBuffer::operator[](unsigned int index) {
    return _buffer[index];
}

const float& CpuFloatBuffer::operator[](unsigned int index) const {
    return _buffer[index];
}

unsigned int CpuFloatBuffer::size() const {
    return _bufferSize;
}

unsigned int CpuFloatBuffer::length() const {
    return _bufferLength;
}

void CpuFloatBuffer::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuFloatBuffer::copyToDevice(float* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuFloatBuffer::copyFromDevice(const float* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
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

float* GpuFloatBuffer::data() {
    return _buffer;
}

const float* GpuFloatBuffer::data() const {
    return _buffer;
}

unsigned int GpuFloatBuffer::size() const {
    return _bufferSize;
}

unsigned int GpuFloatBuffer::length() const {
    return _bufferLength;
}

void GpuFloatBuffer::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuFloatBuffer::copyToDevice(float* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void GpuFloatBuffer::copyFromDevice(const float* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
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

unsigned char* CpuPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* CpuPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int CpuPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int CpuPixelBuffer2D::length() const {
    return _sideLength;
}

void CpuPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void CpuPixelBuffer2D::copyToDevice(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuPixelBuffer2D::copyFromDevice(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
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

unsigned char* GpuPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* GpuPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int GpuPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int GpuPixelBuffer2D::length() const {
    return _sideLength;
}

void GpuPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuPixelBuffer2D::copyToHost(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GpuPixelBuffer2D::copyFromHost(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

// CPUColorMapBuffer functions definitions

unsigned char* CpuColorMapBuffer::data() {
    return _buffer;
}

const unsigned char* CpuColorMapBuffer::data() const {
    return _buffer;
}

unsigned char& CpuColorMapBuffer::red(unsigned int index) {
    return _buffer[3 * index + 0];
}

unsigned char& CpuColorMapBuffer::green(unsigned int index) {
    return _buffer[3 * index + 1];
}

unsigned char& CpuColorMapBuffer::blue(unsigned int index) {
    return _buffer[3 * index + 2];
}

const unsigned char& CpuColorMapBuffer::red(unsigned int index) const {
    return _buffer[3 * index + 0];
}

const unsigned char& CpuColorMapBuffer::green(unsigned int index) const {
    return _buffer[3 * index + 1];
}

const unsigned char& CpuColorMapBuffer::blue(unsigned int index) const {
    return _buffer[3 * index + 2];
}

unsigned int CpuColorMapBuffer::size() const {
    return sizeof(_buffer);
}

unsigned int CpuColorMapBuffer::length() const {
    return _length;
}
}  // namespace graphics