#include <iostream>

#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include "params.h"

#include "cuda/Solver.cuh"

std::filesystem::path CreateOutputDir(const std::filesystem::path& parentDir) {
    if (std::filesystem::exists(parentDir) == false)
        std::filesystem::create_directory(parentDir);

    auto currentFullTime = time(nullptr);

    std::ostringstream currenTimeStream;
    currenTimeStream << std::put_time(localtime(&currentFullTime), "%Y%m%d")
                     << "_"
                     << std::put_time(localtime(&currentFullTime), "%H%M%S");
    auto outputDir = parentDir / currenTimeStream.str();

    std::filesystem::create_directory(outputDir);
    std::filesystem::create_directory(outputDir / "vorticity");
    std::filesystem::create_directory(outputDir / "current");
    std::filesystem::create_directory(outputDir / "stream");
    std::filesystem::create_directory(outputDir / "potential");

    return outputDir;
}

void main() {
    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::cout << "Creating output directory... " << std::endl;

    const std::filesystem::path outputDir = CreateOutputDir("outputs");

    std::cout << "Output directory: " << outputDir << std::endl << std::endl;

    std::cout << "Printing parameters..." << std::endl;

    std::cout << mhd::parameters::ParametersPrint();
    mhd::parameters::ParametersSave(outputDir);

    mhd::CudaTimeCounter counter;

    counter.start();
    {
        mhd::Solver solver;

        // Initial Conditions
        solver.fillNormally(std::time(nullptr));

        // Initial Energy and Time Step
        solver.updateEnergies();
        solver.updateTimeStep();

        std::cout << "Simulation starts..." << std::endl;
        // Zero Data Output
        solver.saveData(outputDir);

        // Main Cycle of the Program
        while (solver.shouldContinue()) {
            // Saving fields from previous timelayer
            solver.saveOldFields();

            // Time Integration Scheme
            // Two-step Scheme

            // First step
            solver.calcKineticRigthPart();
            solver.timeSchemeKin();

            solver.calcMagneticRightPart();
            solver.timeSchemeMag();

            solver.updateStream();
            solver.updateCurrent();

            // Second step
            solver.calcKineticRigthPart();
            solver.timeSchemeKin();

            solver.calcMagneticRightPart();
            solver.timeSchemeMag();

            solver.updateStream();
            solver.updateCurrent();

            // Update Parameters (Energy and Time Step)
            solver.updateEnergies();
            solver.updateTimeStep();

            // Data Output
            solver.timeStep();
            solver.saveData(outputDir);
        }
    }
    counter.stop();

    // Calculation Time Output
    std::cout << std::endl
              << "Simulation time: " << counter.getTime() << std::endl;
}