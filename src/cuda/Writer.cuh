#pragma once

#include <filesystem>
#include <string>

#include "../params.h"

#include "Buffers.cuh"

enum FieldType { Vorticity = 0, Current, Stream, Potential };

namespace mhd {

class Writer {
private:
    CpuDoubleBuffer2D _output;

    std::string uintToStr(unsigned int value);
    bool memcpy(const double* field);
    void save(const double* field, const std::filesystem::path& filePath);
    void clear();

public:
    Writer();
    Writer(unsigned int gridLength);

    template <FieldType Type>
    void saveField(const GpuDoubleBuffer2D& field,
                   const std::filesystem::path& outputDir,
                   unsigned int outputNumber);
    void saveCurentParams(const mhd::parameters::CurrentParameters& params,
                          const std::filesystem::path& outputDir);

    void printField(const GpuComplexBuffer2D& field,
                    const std::string& message = "");
    template <bool IsNormalized = false>
    void printField(const GpuDoubleBuffer2D& field,
                    const std::string& message = "");
};
}  // namespace mhd