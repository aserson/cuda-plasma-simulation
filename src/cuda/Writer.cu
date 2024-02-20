#include "Writer.cuh"

#include <cstring>
#include <fstream>
#include <sstream>

#include "FastFourierTransformator.cuh"
#include "HelperKernels.cuh"
#include "KernelCaller.cuh"

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

bool Writer::memcpy(const double* field) {
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

void Writer::save(const double* field, const std::filesystem::path& filePath) {
    memcpy(field);

    std::ofstream fData(filePath, std::ios::binary | std::ios::out);
    fData.write((char*)(_output.data()), _output.size());
    fData.close();
}

void Writer::clear() {
    memset(_output.data(), 0x0, _output.size());
}

Writer::Writer(unsigned int gridLength) : _output(gridLength) {}

void Writer::saveVorticity(const double* buffer,
                           const std::filesystem::path& outputDir,
                           unsigned int outputNumber) {
    std::filesystem::path filePath =
        outputDir / (uintToStr(outputNumber) + "_vorticity");
    save(buffer, filePath);
}

void Writer::saveCurrent(const double* buffer,
                         const std::filesystem::path& outputDir,
                         unsigned int outputNumber) {
    std::filesystem::path filePath =
        outputDir / (uintToStr(outputNumber) + "_current");
    save(buffer, filePath);
}

void Writer::saveStream(const double* buffer,
                        const std::filesystem::path& outputDir,
                        unsigned int outputNumber) {
    std::filesystem::path filePath =
        outputDir / (uintToStr(outputNumber) + "_stream");
    save(buffer, filePath);
}

void Writer::savePotential(const double* buffer,
                           const std::filesystem::path& outputDir,
                           unsigned int outputNumber) {
    std::filesystem::path filePath =
        outputDir / (uintToStr(outputNumber) + "_potential");
    save(buffer, filePath);
}

void Writer::saveCurrents(const Currents& currents,
                          const std::filesystem::path& outputDir,
                          unsigned int outputNumber) {
    std::filesystem::path filePath =
        outputDir / (uintToStr(outputNumber) + "_currents.yaml");
    std::ofstream fParams(filePath);

    fParams << "T : " << currents.time << std::endl
            << "Nstep : " << currents.stepNumber << std::endl
            << "Ekin : " << currents.kineticEnergy << std::endl
            << "Emag : " << currents.magneticEnergy << std::endl
            << "Vmax : " << currents.maxVelocityField << std::endl
            << "Bnax : " << currents.maxMagneticField << std::endl;

    fParams.close();
}
}  // namespace mhd