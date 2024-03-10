#include "Painter.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "HelperKernels.cuh"
#include "PainterKernels.cuh"

namespace graphics {
Painter::Painter(const std::string& colorMapName, const mhd::Configs& configs)
    : _length(configs._gridLength),
      _cpuColorMap(),
      _gpuColorMap(),
      _cpuPixels(configs._gridLength),
      _gpuPixels(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength) {

    readColorMap(colorMapName);
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

}  // namespace graphics