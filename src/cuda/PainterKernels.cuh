#pragma once
#include "cuda_runtime.h"

namespace graphics {
__global__ static void DoubleToPixels_kernel(unsigned char* output,
                                             const double* input,
                                             unsigned int gridLength,
                                             double amplitude,
                                             const unsigned char* colorMap,
                                             unsigned int colorMapLength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;
    int pixelIdx = 3 * (gridLength * x + y);

    double value = 255. * (amplitude + input[idx]) / (2 * amplitude);
    uint8_t point = uint8_t(value);

    output[pixelIdx + 0] = colorMap[3 * point + 0];
    output[pixelIdx + 1] = colorMap[3 * point + 1];
    output[pixelIdx + 2] = colorMap[3 * point + 2];
}

}  // namespace graphics
