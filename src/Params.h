#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#define M_PI 3.141592653589793238462643

namespace mhd {
namespace parameters {
struct EquationCoefficients {
    static constexpr double nu = 1e-4;
    static constexpr double eta = 1e-4;
};

struct InitialCondition {
    static constexpr double kineticEnergy = 0.5;
    static constexpr double magneticEnergy = 0.5;

    static const unsigned int averageWN = 10;
    //static const unsigned int halfwidthWN = 3;
};

struct SimulationParameters {
    static const unsigned int gridLength = 2048;
    static constexpr double gridStep = 2. * M_PI / ((double)gridLength);
    static constexpr double lambda = 1. / ((double)(gridLength * gridLength));

    static const unsigned int dealaliasingWN = gridLength / 3;

    static constexpr double time = 0.1;
    static constexpr double cft = 0.2;
    static constexpr double maxTimeStep = 0.01;

    static const unsigned int bufferLengthLinear = 128;
    static const unsigned int gridLengthLinear =
        gridLength * gridLength / bufferLengthLinear;
};

struct OutputParameters {
    static constexpr double stepTime = 0.01;

    static constexpr double startTime = 0.0;
    static constexpr double stopTime = stepTime + 1000 * stepTime;
};

struct KernelRunParameters {
    static const unsigned int threadsPerDim = SimulationParameters::gridLength;

    static const unsigned int blockSizeX = 32;
    static const unsigned int blockSizeY = 16;

    static const unsigned int gridSizeX = threadsPerDim / blockSizeX;
    static const unsigned int gridSizeY = threadsPerDim / blockSizeY;

    static const unsigned int blockSizeLinear =
        SimulationParameters::bufferLengthLinear;
    static const unsigned int gridSizeLinear =
        SimulationParameters::gridLengthLinear;
};

struct CurrentParameters {
    double time = 0.0;
    unsigned int stepNumber = 0;
    double timeStep;

    double kineticEnergy;
    double magneticEnergy;

    double maxVelocityField;
    double maxMagneticField;

    double timeOut = mhd::parameters::OutputParameters::startTime;
    unsigned int stepNumberOut = 0;
    double timeStepOut = mhd::parameters::OutputParameters::stepTime;
};

void ParametersPrint() {
    unsigned int N = SimulationParameters::gridLength;
    double T = SimulationParameters::time;

    double kineticEnergy = InitialCondition::kineticEnergy;
    double magneticEnergy = InitialCondition::magneticEnergy;

    double nu = EquationCoefficients::nu;
    double eta = EquationCoefficients::eta;

    std::cout << "Simulation parameters:" << std::endl;
    std::cout << "  Grid Lenght = " << std::setw(13) << std::left << N
              << std::endl;
    std::cout << "  End Time = " << std::setw(12) << std::left << T
              << std::endl;
    std::cout << std::endl;

    std::cout << "Initial condition:" << std::endl;
    std::cout << "  Ekin = " << std::setw(13) << std::left << kineticEnergy
              << std::endl;
    std::cout << "  Emag = " << std::setw(13) << std::left << magneticEnergy
              << std::endl;
    std::cout << std::endl;

    std::cout << "Equation coefficients:" << std::endl;
    std::cout << "  nu = " << std::setw(13) << std::left << nu << std::endl;
    std::cout << "  eta = " << std::setw(13) << std::left << eta << std::endl;

    std::cout << std::endl;
}

void ParametersSave(const std::filesystem::path& outputDir) {
    unsigned int N = SimulationParameters::gridLength;
    double T = SimulationParameters::time;

    double kineticEnergy = InitialCondition::kineticEnergy;
    double magneticEnergy = InitialCondition::magneticEnergy;

    double nu = EquationCoefficients::nu;
    double eta = EquationCoefficients::eta;

    double ToutStep = OutputParameters::stepTime;
    double ToutStart = OutputParameters::startTime;
    double ToutStop = OutputParameters::stopTime;

    std::filesystem::path filePath = outputDir / "params.yaml";
    std::ofstream fParams(filePath);

    fParams << "gridLength: " << N << std::endl
            << "time: " << T << std::endl
            << "Ekin0: " << kineticEnergy << std::endl
            << "Emag0: " << magneticEnergy << std::endl
            << "nu: " << nu << std::endl
            << "eta: " << eta << std::endl
            << "outStep: " << ToutStep << std::endl
            << "outStart: " << ToutStart << std::endl
            << "outStop: " << ToutStop << std::endl;

    fParams.close();
}
}  // namespace parameters
}  // namespace mhd
