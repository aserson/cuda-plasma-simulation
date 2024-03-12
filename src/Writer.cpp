#include "Writer.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace mhd {
std::string Writer::uintToStr(unsigned int value) {
    std::ostringstream output;
    if (value < 10) {
        output << "00" << value;
    } else if ((10 <= value) && (value < 100)) {
        output << "0" << value;
    } else {
        output << value;
    }

    return output.str();
}

void Writer::save(const double* field, const std::filesystem::path& filePath) {
    _output.copyFromDevice(field);

    std::ofstream fData(filePath, std::ios::binary | std::ios::out);
    fData.write((char*)(_output.data()), _output.size());
    fData.close();
}

void Writer::clear() {
    memset(_output.data(), 0x0, _output.size());
}

Writer::Writer(const std::filesystem::path& outputDir,
               const mhd::Configs& configs)
    : _outputDir(outputDir),
      _painter(configs),
      _output(configs._gridLength),
      _outputTime(configs._outputStart),
      _outputStep(configs._outputStep),
      _outputStop(configs._outputStop),
      _outputNumber(0),
      _settings{configs._saveData,      configs._savePNG,
                configs._saveVorticity, configs._saveCurrent,
                configs._saveStream,    configs._savePotential,
                configs._showGraphics} {}

bool Writer::saveData(mhd::Helper& helper, opengl::Creater& creater) {
    if (shouldWrite(helper._currents.time)) {
        if (_settings.saveData) {
            std::filesystem::path currentDir =
                _outputDir / uintToStr(_outputNumber);
            std::filesystem::create_directory(currentDir);

            if (_settings.saveVorticity) {
                save(helper.getVorticity().data(),
                     _outputDir / "vorticity" / uintToStr(_outputNumber));
            }
            if (_settings.saveCurrent) {
                save(helper.getVorticity().data(),
                     _outputDir / "current" / uintToStr(_outputNumber));
            }
            if (_settings.saveStream) {
                save(helper.getVorticity().data(),
                     _outputDir / "stream" / uintToStr(_outputNumber));
            }
            if (_settings.savePotential) {
                save(helper.getVorticity().data(),
                     _outputDir / "potential" / uintToStr(_outputNumber));
            }

            saveCurrents(helper._currents, currentDir / "data.yaml");
        }

        if (_settings.showGraphics) {
            if (_settings.saveVorticity) {
                _painter.doubleToPixels(helper.getVorticity(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
            } else if (_settings.saveCurrent) {
                _painter.doubleToPixels(helper.getCurrent(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
            } else if (_settings.saveStream) {
                _painter.doubleToPixels(helper.getStream(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
            } else if (_settings.savePotential) {
                _painter.doubleToPixels(helper.getPotential(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
            }
            creater.AddTexture(_painter.getPixels().data(),
                               _painter.getLength(), _painter.getLength());
        }

        if (_settings.savePNG) {
            if (_settings.saveVorticity) {
                _painter.doubleToPixels(helper.getVorticity(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "vorticityPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.saveCurrent) {
                _painter.doubleToPixels(helper.getCurrent(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "currentPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.saveStream) {
                _painter.doubleToPixels(helper.getStream(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "streamPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.savePotential) {
                _painter.doubleToPixels(helper.getPotential(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "potentialPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
        }

        printCurrents(helper._currents);

        step();

        return true;
    }
    return false;
}

bool Writer::saveData(mhd::Helper& helper) {
    if (shouldWrite(helper._currents.time)) {
        if (_settings.saveData) {
            std::filesystem::path currentDir =
                _outputDir / uintToStr(_outputNumber);
            std::filesystem::create_directory(currentDir);

            if (_settings.saveVorticity) {
                save(helper.getVorticity().data(),
                     _outputDir / "vorticity" / uintToStr(_outputNumber));
            }
            if (_settings.saveCurrent) {
                save(helper.getVorticity().data(),
                     _outputDir / "current" / uintToStr(_outputNumber));
            }
            if (_settings.saveStream) {
                save(helper.getVorticity().data(),
                     _outputDir / "stream" / uintToStr(_outputNumber));
            }
            if (_settings.savePotential) {
                save(helper.getVorticity().data(),
                     _outputDir / "potential" / uintToStr(_outputNumber));
            }

            saveCurrents(helper._currents, currentDir / "data.yaml");
        }

        if (_settings.savePNG) {
            if (_settings.saveVorticity) {
                _painter.doubleToPixels(helper.getVorticity(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "vorticityPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.saveCurrent) {
                _painter.doubleToPixels(helper.getCurrent(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "currentPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.saveStream) {
                _painter.doubleToPixels(helper.getStream(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "streamPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
            if (_settings.savePotential) {
                _painter.doubleToPixels(helper.getPotential(),
                                        helper.DoubleBufferB(),
                                        helper.CpuLinearBufferX());
                _painter.saveAsPNG(_outputDir / "potentialPNG" /
                                   (uintToStr(_outputNumber) + ".png"));
            }
        }

        printCurrents(helper._currents);

        step();

        return true;
    }
    return false;
}

void Writer::saveCurrents(const Currents& currents,
                          const std::filesystem::path& filePath) {
    std::ofstream fParams(filePath);

    fParams << "T : " << currents.time << std::endl
            << "Nstep : " << currents.stepNumber << std::endl
            << "Ekin : " << currents.kineticEnergy << std::endl
            << "Emag : " << currents.magneticEnergy << std::endl
            << "Vmax : " << currents.maxVelocityField << std::endl
            << "Bnax : " << currents.maxMagneticField << std::endl;

    fParams.close();
}

void Writer::printCurrents(const mhd::Currents& currents) {
    if (currents.stepNumber == 0) {
        std::cout << std::left;
        std::cout << std::setw(6) << "Step:";
        std::cout << std::setw(6) << "Time:";
        std::cout << std::right;
        std::cout << std::setw(10) << "dTime:";
        std::cout << std::setw(11) << "Ekin:";
        std::cout << std::setw(12) << "Emag:";
        std::cout << std::setw(12) << "Esum:";
        std::cout << std::endl;

        std::cout
            << "_____________________________________________________________"
            << std::endl;
    }
    std::cout << " ";
    std::cout << std::left;
    std::cout << std::setw(8) << currents.stepNumber;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << std::setw(6) << currents.time;

    std::cout << std::fixed << std::setprecision(4) << std::right;
    std::cout << std::setw(10) << currents.timeStep;
    std::cout << std::setw(12) << currents.kineticEnergy;
    std::cout << std::setw(12) << currents.magneticEnergy;
    std::cout << std::setw(12)
              << currents.kineticEnergy + currents.magneticEnergy << std::endl;
}
}  // namespace mhd