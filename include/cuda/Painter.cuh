#pragma once

#include <string>

#include "../Configs.h"

#include "Buffers.cuh"
#include "KernelCaller.cuh"

namespace graphics {

class Painter {
private:
    KernelCaller _caller;

    unsigned int _length;

    CpuColorMapBuffer _colorMap;

    CpuPixelBuffer2D _cpuPixels;
    GpuPixelBuffer2D _gpuPixels;

    bool readColorMap(const std::string& colorMapName,
                      const std::filesystem::path& resPath);
    float findAmplitude(cudaStream_t& stream, const mhd::GpuDoubleBuffer2D& src,
                        mhd::GpuDoubleBuffer2D& tmp);

public:
    Painter(const mhd::Configs& configs, const std::filesystem::path& resPath);

    void doubleToPixels(cudaStream_t& stream, const mhd::GpuDoubleBuffer2D& src,
                        mhd::GpuDoubleBuffer2D& tmp);

    const CpuPixelBuffer2D& getPixels() const { return _cpuPixels; }
    unsigned int getLength() const { return _length; }
};
}  // namespace graphics
