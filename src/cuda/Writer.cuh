#pragma once

#include <filesystem>
#include <string>

#include "../Configs.h"

#include "Buffers.cuh"

namespace mhd {
class KernelCaller;

class Writer {
private:
    CpuDoubleBuffer2D _output;

    std::string uintToStr(unsigned int value);
    bool memcpy(const double* field);
    void save(const double* field, const std::filesystem::path& filePath);
    void clear();

public:
    Writer(unsigned int gridLength);

    void saveVorticity(const double* buffer,
                       const std::filesystem::path& outputDir,
                       unsigned int outputNumber);

    void saveCurrent(const double* buffer,
                     const std::filesystem::path& outputDir,
                     unsigned int outputNumber);

    void saveStream(const double* buffer,
                    const std::filesystem::path& outputDir,
                    unsigned int outputNumber);

    void savePotential(const double* buffer,
                       const std::filesystem::path& outputDir,
                       unsigned int outputNumber);

    void saveCurrents(const Currents& currents,
                      const std::filesystem::path& outputDir,
                      unsigned int outputNumber);
};
}  // namespace mhd