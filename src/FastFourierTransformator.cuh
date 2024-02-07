#pragma once
#include <cufft.h>

#include "Buffers.cuh"
#include "KernelCaller.cuh"

#define CUFFT_CALL(result) \
    checkCufftResult(result, __FUNCTION__, __FILE__, __LINE__)

void checkCufftResult(cufftResult_t result, const std::string& functionName,
                      const std::string& fileName, int lineNumber) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error " << result << " in " << functionName
                  << " at " << fileName << ":" << lineNumber << std::endl;
    }
}

namespace mhd {
class FastFourierTransformator {
private:
    cufftHandle planD2Z, planZ2D;

public:
    FastFourierTransformator(unsigned int gridLength) {
        CUFFT_CALL(cufftPlan2d(&planD2Z, gridLength, gridLength, CUFFT_D2Z));
        CUFFT_CALL(cufftPlan2d(&planZ2D, gridLength, gridLength, CUFFT_Z2D));
    }

    ~FastFourierTransformator() {
        CUFFT_CALL(cufftDestroy(planD2Z));
        CUFFT_CALL(cufftDestroy(planZ2D));
    }

    void forwardFFT(double* input, cufftDoubleComplex* output) const {
        CUFFT_CALL(cufftExecD2Z(planD2Z, input, output));
    }

    void inverseFFT(cufftDoubleComplex* input, double* output) const {
        CUFFT_CALL(cufftExecZ2D(planZ2D, input, output));
    }

    void forward(GpuDoubleBuffer& input, GpuComplexBuffer& output) const {
        CUFFT_CALL(cufftExecD2Z(planD2Z, input.data(), output.data()));
    }

    void inverse(GpuComplexBuffer& input, GpuDoubleBuffer& output) const {
        CUFFT_CALL(cufftExecZ2D(planZ2D, input.data(), output.data()));
    }
};
}  // namespace mhd

typedef mhd::FastFourierTransformator FFTransformator;
