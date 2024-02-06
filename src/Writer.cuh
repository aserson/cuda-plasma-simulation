#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "Helper.cuh"
#include "KernelCaller.cuh"

enum FieldType { Vorticity, Current, StreamFunction, MagneticPotential };

class Writer {
   private:
    double* _output;
    unsigned int _gridLength;
    unsigned int _outputSize;

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
            CUDA_CALL(cudaMemcpy(_output, field, _outputSize,
                                 cudaMemcpyDeviceToHost));
            return true;
        } else {
            CUDA_CALL(
                cudaMemcpy(_output, field, _outputSize, cudaMemcpyHostToHost));
            return true;
        }

        return false;
    }

   public:
    Writer() {
        _gridLength = mhd::parameters::SimulationParameters::gridLength;
        _outputSize = _gridLength * _gridLength * sizeof(double);
        CUDA_CALL(
            cudaHostAlloc((void**)&_output, _outputSize, cudaHostAllocDefault));
    }

    Writer(unsigned int girdLength) : _gridLength(girdLength) {
        _outputSize = _gridLength * _gridLength * sizeof(double);
        CUDA_CALL(
            cudaHostAlloc((void**)&_output, _outputSize, cudaHostAllocDefault));
    }

    ~Writer() {
        if (_output != nullptr) {
            CUDA_CALL(cudaFreeHost(_output));
        }
    }

    void save(const double* field, const std::filesystem::path& filePath) {
        memcpy(field);

        std::ofstream fData(filePath, std::ios::binary | std::ios::out);
        fData.write((char*)_output, _outputSize);
        fData.close();
    }

    template <FieldType Type>
    void saveField(const GpuComplexBuffer& field,
                   const std::filesystem::path& outputDir,
                   const mhd::FastFourierTransformator& transformator,
                   AuxiliaryFields& aux,
                   const mhd::parameters::CurrentParameters& params) {
        unsigned int gridLength =
            mhd::parameters::SimulationParameters::gridLength;
        double lambda = mhd::parameters::SimulationParameters::lambda;

        std::filesystem::path filePath;

        switch (Type) {
            case Vorticity:
                filePath = outputDir / "vorticity" /
                           ("out" + uintToStr(params.stepNumberOut));
                break;
            case Current:
                filePath = outputDir / "current" /
                           ("out" + uintToStr(params.stepNumberOut));
                break;
            case StreamFunction:
                filePath = outputDir / "streamFunction" /
                           ("out" + uintToStr(params.stepNumberOut));
                break;
            case MagneticPotential:
                filePath = outputDir / "magneticPotential" /
                           ("out" + uintToStr(params.stepNumberOut));
                break;
            default:
                break;
        }

        CallKernel(MultComplex_kernel, field.data(), gridLength, lambda,
                   aux._complexTmp.data());
        transformator.inverse(aux._complexTmp, aux._doubleTmpA);

        save(aux._doubleTmpA.data(), filePath);
    }

    void saveCurentParams(const mhd::parameters::CurrentParameters& params,
                          const std::filesystem::path& outputDir) {
        std::filesystem::path filePath =
            outputDir / "params" /
            ("out" + uintToStr(params.stepNumberOut) + ".yaml");
        std::ofstream fParams(filePath);

        fParams << "time: " << params.time << std::endl
                << "stepNumber: " << params.stepNumber << std::endl
                << "kineticEnergy: " << params.kineticEnergy << std::endl
                << "magneticEnergy: " << params.magneticEnergy << std::endl
                << "fullEnergy: "
                << params.kineticEnergy + params.magneticEnergy << std::endl
                << "maxVelocityField: " << params.maxVelocityField << std::endl
                << "naxMagnecitField: " << params.maxMagneticField << std::endl;

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
        CallKernelFull(MultDouble_kernel, tmpDouble.data(), _gridLength, lambda,
                       tmpDouble.data());

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
        CallKernelFull(MultDouble_kernel, tmpDouble.data(), _gridLength, lambda,
                       tmpDouble.data());

        clear();
        memcpy(tmpDouble.data());

        std::cout << "Field " << message << ":	" << _output[0] << "	"
                  << _output[1] << "	" << _output[2] << "	" << _output[3]
                  << "	" << _output[4] << std::endl
                  << std::endl;

        clear();
    }

    void clear() {
        if (_output != nullptr) {
            memset(_output, 0x0, _outputSize);
        }
    }
};
