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
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
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
    _bufferSize = _sideLength * _sideLength * sizeof(cufftDoubleComplex);

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
}  // namespace mhd