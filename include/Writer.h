﻿#pragma once

#include <filesystem>
#include <string>

#include "Configs.h"

#include "cuda/Buffers.cuh"
#include "cuda/Helper.cuh"
#include "cuda/Painter.cuh"
#include "openGL/Creater.h"

namespace mhd {
class Writer {
private:
    CpuDoubleBuffer2D _output;
    graphics::Painter _painter;

    void save(const double* field, const std::filesystem::path& filePath);
    void clear();

    const std::filesystem::path _outputPath;

    double _outputTime;
    double _outputStep;
    double _outputStop;
    unsigned int _outputNumber;

    struct Settings {
        bool saveData;
        bool savePNG;
        bool saveVorticity;
        bool saveCurrent;
        bool saveStream;
        bool savePotential;
        bool showGraphics;
    } _settings;

public:
    Writer(const std::filesystem::path& outputPath, const mhd::Configs& configs,
           const std::filesystem::path& resPath);

    bool saveData(mhd::Helper& writer, opengl::Creater& creater);

    void saveCurrents(const Currents& currents,
                      const std::filesystem::path& filePath);

    void printCurrents(const mhd::Currents& currents);

    bool shouldWrite(double time) {
        return ((time >= _outputTime) && (time <= _outputStop));
    }

    bool shouldPaint(double time) {
        return time >= _outputTime;
    }

    void step() {
        _outputNumber++;
        _outputTime += _outputStep;
    }

    static std::string uintToStr(unsigned int value);
};
}  // namespace mhd