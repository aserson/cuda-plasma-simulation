#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "KernelCaller.cuh"

enum FieldType { Vorticity = 0, Current, Stream, Potential };

namespace mhd {

class Writer {
private:
    CpuDoubleBuffer _output;

    std::string uintToStr(unsigned int value) {
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

    bool memcpy(const double* field) {
        cudaPointerAttributes attributes;
        CUDA_CALL(cudaPointerGetAttributes(&attributes, field));

        if (attributes.type == cudaMemoryTypeDevice) {
            CUDA_CALL(cudaMemcpy(_output.data(), field, _output.size(),
                                 cudaMemcpyDeviceToHost));
            return true;
        } else if (attributes.type == cudaMemoryTypeHost) {
            CUDA_CALL(cudaMemcpy(_output.data(), field, _output.size(),
                                 cudaMemcpyHostToHost));
            return true;
        } else {
            return false;
        }
    }

    void save(const double* field, const std::filesystem::path& filePath) {
        memcpy(field);

        std::ofstream fData(filePath, std::ios::binary | std::ios::out);
        fData.write((char*)(_output.data()), _output.size());
        fData.close();
    }

    void clear() { memset(_output.data(), 0x0, _output.size()); }

public:
    Writer() : _output(mhd::parameters::SimulationParameters::gridLength) {}

    Writer(unsigned int gridLength) : _output(gridLength) {}

    template <FieldType Type>
    void saveField(const GpuDoubleBuffer& field,
                   const std::filesystem::path& outputDir,
                   unsigned int outputNumber) {
        unsigned int gridLength =
            mhd::parameters::SimulationParameters::gridLength;
        double lambda = mhd::parameters::SimulationParameters::lambda;

        std::filesystem::path filePath;

        switch (Type) {
            case Vorticity:
                filePath =
                    outputDir / "vorticity" / ("out" + uintToStr(outputNumber));
                break;
            case Current:
                filePath =
                    outputDir / "current" / ("out" + uintToStr(outputNumber));
                break;
            case Stream:
                filePath =
                    outputDir / "stream" / ("out" + uintToStr(outputNumber));
                break;
            case Potential:
                filePath =
                    outputDir / "potential" / ("out" + uintToStr(outputNumber));
                break;
            default:
                break;
        }

        save(field.data(), filePath);
    }

    void saveCurentParams(const mhd::parameters::CurrentParameters& params,
                          const std::filesystem::path& outputDir) {
        std::filesystem::path filePath =
            outputDir / "params" /
            ("out" + uintToStr(params.stepNumberOut) + ".yaml");
        std::ofstream fParams(filePath);

        fParams << "T: " << params.time << std::endl
                << "Nstep: " << params.stepNumber << std::endl
                << "Ekin: " << params.kineticEnergy << std::endl
                << "Emag: " << params.magneticEnergy << std::endl
                << "E: " << params.kineticEnergy + params.magneticEnergy
                << std::endl
                << "Vmax: " << params.maxVelocityField << std::endl
                << "Bnax: " << params.maxMagneticField << std::endl;

        fParams.close();
    }

    void printField(const GpuComplexBuffer& field,
                    const std::string& message = "") {
        GpuComplexBuffer tmpComplex(field.length());
        tmpComplex.copyFromDevice(field.data());

        GpuDoubleBuffer tmpDouble(field.length());
        FFTransformator transformator(field.length());
        transformator.inverseFFT(tmpComplex.data(), tmpDouble.data());

        double lambda = mhd::parameters::SimulationParameters::lambda;
        CallKernelFull(MultDouble_kernel, tmpDouble.data(), tmpDouble.length(),
                       lambda, tmpDouble.data());

        clear();
        memcpy(tmpDouble.data());

        std::cout << "Field " << message << ":	" << _output[0] << "	"
                  << _output[1] << "	" << _output[2] << "	" << _output[3]
                  << "	" << _output[4] << std::endl
                  << std::endl;

        clear();
    }

    template <bool IsNormalized = false>
    void printField(const GpuDoubleBuffer& field,
                    const std::string& message = "") {
        GpuDoubleBuffer tmpDouble(field.length());
        tmpDouble.copyFromDevice(field.data());

        double lambda =
            (IsNormalized) ? 1. : mhd::parameters::SimulationParameters::lambda;
        CallKernelFull(MultDouble_kernel, tmpDouble.data(), tmpDouble.length(),
                       lambda, tmpDouble.data());

        clear();
        memcpy(tmpDouble.data());

        std::cout << "Field " << message << ":	" << _output[0] << "	"
                  << _output[1] << "	" << _output[2] << "	" << _output[3]
                  << "	" << _output[4] << std::endl
                  << std::endl;

        clear();
    }
};
}  // namespace mhd