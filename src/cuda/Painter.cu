#include "cuda/Painter.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "cuda/HelperKernels.cuh"

namespace graphics {
static const unsigned int colorMapLength = 256;
__constant__ unsigned char colorMap[colorMapLength * 3];

__global__ void DoubleToPixels_kernel(unsigned char* output,
                                      const double* input,
                                      unsigned int gridLength,
                                      float amplitude) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridLength * x + y;

    float value = static_cast<float>(input[idx]);
    value = (amplitude + value) / (2 * amplitude);
    uint8_t point = static_cast<uint8_t>(value * (colorMapLength - 1));

    output[3 * idx + 0] = colorMap[3 * point + 0];
    output[3 * idx + 1] = colorMap[3 * point + 1];
    output[3 * idx + 2] = colorMap[3 * point + 2];
}

Painter::Painter(const mhd::Configs& configs,
                 const std::filesystem::path& resPath)
    : _length(configs._gridLength),
      _cpuPixels(configs._gridLength),
      _gpuPixels(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._dimBlock) {
    if (readColorMap(configs._colorMap, resPath))
        CUDA_CALL(
            cudaMemcpyToSymbol(colorMap, _colorMap.data(), _colorMap.size()));
}

bool Painter::readColorMap(const std::string& colorMapName,
                           const std::filesystem::path& resPath) {
    std::filesystem::path filePath = resPath / "colormaps" / colorMapName;
    if (!exists(filePath)) {
        std::cout << "Color Map by name " << colorMapName << " not exists"
                  << std::endl;
        return false;
    }

    float red, green, blue;
    unsigned int i = 0;

    std::ifstream file(filePath);
    if (file.is_open()) {
        while ((file >> red >> green >> blue) && (i < _colorMap.length())) {
            _colorMap.red(i) = (unsigned char)(float(255) * red);
            _colorMap.green(i) = (unsigned char)(float(255) * green);
            _colorMap.blue(i) = (unsigned char)(float(255) * blue);
            i++;
        }
    } else {
        std::cout << "File " << filePath.filename() << " can not be opened "
                  << std::endl;
        return false;
    }

    return true;
}

float Painter::findAmplitude(cudaStream_t& stream,
                             const mhd::GpuDoubleBuffer2D& src,
                             mhd::GpuDoubleBuffer2D& tmp) {

    _caller.callReduction(stream, src.fullLength(), mhd::Max_kernel, src.data(),
                          tmp.data());

    double maxValue = 0.;
    CUDA_CALL(cudaMemcpyAsync(&maxValue, tmp.data(), sizeof(double),
                              cudaMemcpyDeviceToHost, stream));

    return maxValue;
}

void Painter::doubleToPixels(cudaStream_t& stream,
                             const mhd::GpuDoubleBuffer2D& src,
                             mhd::GpuDoubleBuffer2D& tmp) {
    float amplitude = findAmplitude(stream, src, tmp);

    _caller.callFull(stream, DoubleToPixels_kernel, _gpuPixels.data(),
                     src.data(), src.length(), static_cast<float>(amplitude));

    cudaStreamSynchronize(stream);
    _cpuPixels.copyFromDevice(stream, _gpuPixels.data());
}
}  // namespace graphics