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

// CPUPixelBuffer2D functions definitions

CPUPixelBuffer2D::CPUPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

CPUPixelBuffer2D::CPUPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CPUPixelBuffer2D::~CPUPixelBuffer2D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

unsigned char* CPUPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* CPUPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int CPUPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int CPUPixelBuffer2D::length() const {
    return _sideLength;
}

void CPUPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void CPUPixelBuffer2D::copyToDevice(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CPUPixelBuffer2D::copyFromDevice(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// GPUPixelBuffer2D functions definitions

GPUPixelBuffer2D::GPUPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

GPUPixelBuffer2D::GPUPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GPUPixelBuffer2D::~GPUPixelBuffer2D() {
    cudaFree(_buffer);
}

unsigned char* GPUPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* GPUPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int GPUPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int GPUPixelBuffer2D::length() const {
    return _sideLength;
}

void GPUPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GPUPixelBuffer2D::copyToHost(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GPUPixelBuffer2D::copyFromHost(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

// CPUColorMapBuffer functions definitions

CPUColorMapBuffer::CPUColorMapBuffer() {
    _bufferSize = _length * _channels * sizeof(unsigned char);
    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CPUColorMapBuffer::~CPUColorMapBuffer() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

unsigned char* CPUColorMapBuffer::data() {
    return _buffer;
}

const unsigned char* CPUColorMapBuffer::data() const {
    return _buffer;
}

unsigned char& CPUColorMapBuffer::red(unsigned int index) {
    return _buffer[3 * index + 0];
}

unsigned char& CPUColorMapBuffer::green(unsigned int index) {
    return _buffer[3 * index + 1];
}

unsigned char& CPUColorMapBuffer::blue(unsigned int index) {
    return _buffer[3 * index + 2];
}

const unsigned char& CPUColorMapBuffer::red(unsigned int index) const {
    return _buffer[3 * index + 0];
}

const unsigned char& CPUColorMapBuffer::green(unsigned int index) const {
    return _buffer[3 * index + 1];
}

const unsigned char& CPUColorMapBuffer::blue(unsigned int index) const {
    return _buffer[3 * index + 2];
}

unsigned int CPUColorMapBuffer::size() const {
    return _bufferSize;
}

unsigned int CPUColorMapBuffer::length() const {
    return _length;
}

void CPUColorMapBuffer::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void CPUColorMapBuffer::copyToDevice(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CPUColorMapBuffer::copyFromDevice(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// GPUColorMapBuffer functions definitions

GPUColorMapBuffer::GPUColorMapBuffer() {
    _bufferSize = _length * _channels * sizeof(unsigned char);
    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GPUColorMapBuffer::~GPUColorMapBuffer() {
    CUDA_CALL(cudaFree(_buffer));
}

unsigned char* GPUColorMapBuffer::data() {
    return _buffer;
}

const unsigned char* GPUColorMapBuffer::data() const {
    return _buffer;
}

unsigned int GPUColorMapBuffer::size() const {
    return _bufferSize;
}

unsigned int GPUColorMapBuffer::length() const {
    return _length;
}

void GPUColorMapBuffer::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GPUColorMapBuffer::copyToHost(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GPUColorMapBuffer::copyFromHost(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}
}  // namespace graphics