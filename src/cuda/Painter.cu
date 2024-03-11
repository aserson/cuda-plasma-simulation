#include "Painter.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "HelperKernels.cuh"
#include "fpng\fpng.h"

namespace graphics {
__global__ void DoubleToPixels_kernel(unsigned char* output,
                                      const double* input,
                                      unsigned int gridLength, double amplitude,
                                      unsigned char* colorMap,
                                      unsigned int colorMapLength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    double value = (amplitude + input[idx]) / (2 * amplitude);
    uint8_t point = static_cast<uint8_t>(value * (colorMapLength - 1));

    output[3 * idx + 0] = colorMap[3 * point + 0];
    output[3 * idx + 1] = colorMap[3 * point + 1];
    output[3 * idx + 2] = colorMap[3 * point + 2];
}

Painter::Painter(const mhd::Configs& configs)
    : _length(configs._gridLength),
      _cpuColorMap(),
      _gpuColorMap(),
      _cpuPixels(configs._gridLength),
      _gpuPixels(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength) {

    readColorMap(configs._colorMap);
    _gpuColorMap.copyFromHost(_cpuColorMap.data());
}

void Painter::readColorMap(const std::string& colorMapName) {
    std::filesystem::path filePath("res/colormaps");
    filePath /= colorMapName;

    float red, green, blue;
    unsigned int i = 0;

    std::ifstream file(filePath);
    if (file.is_open()) {
        while ((file >> red >> green >> blue) && (i < _cpuColorMap.length())) {
            _cpuColorMap.red(i) = (unsigned char)(float(255) * red);
            _cpuColorMap.green(i) = (unsigned char)(float(255) * green);
            _cpuColorMap.blue(i) = (unsigned char)(float(255) * blue);
            i++;
        }
    }
    file.close();
}

double Painter::findAmplitude(const mhd::GpuDoubleBuffer2D& src,
                              mhd::GpuDoubleBuffer2D& doubleBuffer,
                              mhd::CpuDoubleBuffer1D& cpuLinearBuffer) {
    _caller.callLinear(mhd::Max_kernel, src.data(), doubleBuffer.data());

    cpuLinearBuffer.copyFromDevice(doubleBuffer.data());

    double v = 0.;
    for (unsigned int i = 0; i < cpuLinearBuffer.length(); i++) {
        v = (fabs(cpuLinearBuffer[i]) > v) ? fabs(cpuLinearBuffer[i]) : v;
    }

    return v;
}

void Painter::doubleToPixels(const mhd::GpuDoubleBuffer2D& src,
                             mhd::GpuDoubleBuffer2D& doubleBuffer,
                             mhd::CpuDoubleBuffer1D& cpuLinearBuffer) {
    double amplitude = findAmplitude(src, doubleBuffer, cpuLinearBuffer);

    _caller.callFull(DoubleToPixels_kernel, _gpuPixels.data(), src.data(),
                     src.length(), amplitude, _gpuColorMap.data(),
                     _gpuColorMap.length());

    _cpuPixels.copyFromDevice(_gpuPixels.data());
}

void Painter::saveAsPNG(const std::filesystem::path& filePath) {
    fpng::fpng_encode_image_to_file(filePath.string().c_str(),
                                    _cpuPixels.data(), _length, _length, 3);
}
}  // namespace graphics