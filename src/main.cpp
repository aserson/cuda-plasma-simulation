#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "Configs.h"
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

    return outputDir;
}

void main() {
    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::cout << "Reading configurations file... " << std::endl;

    mhd::Configs configs("configs/standart1024.yaml");

    std::cout << "Creating output directory... " << std::endl;

    const std::filesystem::path outputDir = CreateOutputDir("outputs");

    std::cout << "Output directory: " << outputDir << std::endl << std::endl;

    std::cout << "Printing parameters..." << std::endl;

    std::cout << configs.ParametersPrint();
    configs.ParametersSave(outputDir);

    mhd::CudaTimeCounter counter;

    counter.start();
    {
        mhd::Solver solver(configs);

        // Initial Conditions
        solver.fillNormally(std::time(nullptr));

        // Initial Energy and Time Step
        solver.updateEnergies();
        solver.updateTimeStep();

        std::cout << "Simulation starts..." << std::endl;

        // Initial Data Output
        solver.saveDataLite(outputDir);

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
            solver.timeStep();

            // Data Output
            solver.saveDataLite(outputDir);
        }
    }
    counter.stop();

    // Calculation Time Output
    std::cout << std::endl
              << "Simulation time: " << counter.getTime() << std::endl;
}