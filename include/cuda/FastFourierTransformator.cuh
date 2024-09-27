#pragma once
#include <cufft.h>
#include <cufftXt.h>

#include "Buffers.cuh"

namespace mhd {
class FastFourierTransformator {
private:
    cufftHandle _planD2Z, _planZ2D;

public:
    FastFourierTransformator(unsigned int gridLength);

    ~FastFourierTransformator();

    void forwardFFT(cudaStream_t& stream, double* input,
                    cufftDoubleComplex* output) const;
    void inverseFFT(cudaStream_t& stream, cufftDoubleComplex* input,
                    double* output) const;

    void forward(cudaStream_t& stream, GpuDoubleBuffer2D& input,
                 GpuComplexBuffer2D& output) const;
    void inverse(cudaStream_t& stream, GpuComplexBuffer2D& input,
                 GpuDoubleBuffer2D& output) const;
};
}  // namespace mhd
