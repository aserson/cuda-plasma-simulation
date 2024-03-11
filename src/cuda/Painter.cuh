#pragma once

#include <string>

#include "../Configs.h"

#include "Buffers.cuh"
#include "KernelCaller.cuh"

namespace graphics {
class Painter {
protected:
    KernelCaller _caller;

    unsigned int _length;

    CPUColorMapBuffer _cpuColorMap;
    GPUColorMapBuffer _gpuColorMap;

    CPUPixelBuffer2D _cpuPixels;
    GPUPixelBuffer2D _gpuPixels;

    void readColorMap(const std::string& colorMapName);
    double findAmplitude(const mhd::GpuDoubleBuffer2D& src,
                         mhd::GpuDoubleBuffer2D& doubleBuffer,
                         mhd::CpuDoubleBuffer1D& cpuLinearBuffer);

public:
    Painter(const mhd::Configs& configs);

    void doubleToPixels(const mhd::GpuDoubleBuffer2D& src,
                        mhd::GpuDoubleBuffer2D& doubleBuffer,
                        mhd::CpuDoubleBuffer1D& cpuLinearBuffer);

    void saveAsPNG(const std::filesystem::path& filePath);

    const CPUPixelBuffer2D& getPixels() const { return _cpuPixels; }
    unsigned int getLength() const { return _length; }
};
}  // namespace graphics
