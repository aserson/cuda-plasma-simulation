#pragma once

#include <filesystem>
#include <string>

#include "Configs.h"

#include "cuda/Buffers.cuh"
#include "cuda/Helper.cuh"
#include "png/Painter.h"

namespace mhd {
class Writer {
private:
    CpuDoubleBuffer2D _output;
    void save(const double* field, const std::filesystem::path& filePath);
    void clear();

    std::filesystem::path _outputDir;

    double _outputTime;
    double _outputStep;
    double _outputStop;
    unsigned int _outputNumber;

    struct Settings {
        bool saveVorticity;
        bool saveCurrent;
        bool saveStream;
        bool savePotential;
        bool savePNG;
    } _settings;

public:
    Writer(const std::filesystem::path& outputDir, const mhd::Configs& configs);

    void saveData(mhd::Helper& writer, graphics::Painter& painter);

    void saveCurrents(const Currents& currents,
                      const std::filesystem::path& filePath);

    void printCurrents(const mhd::Currents& currents);

    bool shouldWrite(double time) {
        return ((time >= _outputTime) && (time <= _outputStop));
    }

    void step() {
        _outputNumber++;
        _outputTime += _outputStep;
    }

    static std::string uintToStr(unsigned int value);
};
}  // namespace mhd