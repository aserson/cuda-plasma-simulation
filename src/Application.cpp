#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "Configs.h"
#include "Writer.h"
#include "cuda/Solver.cuh"
#include "openGL/Creater.h"

std::filesystem::path CreateOutputDir(const mhd::Configs& configs) {
    std::filesystem::path parentDir = "outputs";

    if (std::filesystem::exists(parentDir) == false)
        std::filesystem::create_directory(parentDir);

    auto currentFullTime = time(nullptr);

    std::ostringstream currenTimeStream;
    currenTimeStream << std::put_time(localtime(&currentFullTime), "%Y%m%d")
                     << "_"
                     << std::put_time(localtime(&currentFullTime), "%H%M%S");
    auto outputDir = parentDir / currenTimeStream.str();

    std::filesystem::create_directory(outputDir);

    if (configs._saveData) {
        if (configs._saveVorticity)
            std::filesystem::create_directory(outputDir / "vorticity");
        if (configs._saveCurrent)
            std::filesystem::create_directory(outputDir / "current");
        if (configs._saveStream)
            std::filesystem::create_directory(outputDir / "stream");
        if (configs._savePotential)
            std::filesystem::create_directory(outputDir / "potential");
    }

    if (configs._savePNG) {
        if (configs._saveVorticity)
            std::filesystem::create_directory(outputDir / "vorticityPNG");
        if (configs._saveCurrent)
            std::filesystem::create_directory(outputDir / "currentPNG");
        if (configs._saveStream)
            std::filesystem::create_directory(outputDir / "streamPNG");
        if (configs._savePotential)
            std::filesystem::create_directory(outputDir / "potentialPNG");
    }

    return outputDir;
}

int main(int argc, char* argv[]) {
    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::string confisFile;

    if (argc > 1) {
        confisFile += argv[1];
    } else {
        confisFile += "configs\\standart1024.yaml";
    }

    std::cout << "Reading configurations file " << confisFile << "... "
              << std::endl;

    mhd::Configs configs(confisFile);

    std::cout << "Creating output directories... " << std::endl;

    const std::filesystem::path outputDir = CreateOutputDir(configs);

    std::cout << "Output directory: " << outputDir.string() << std::endl
              << std::endl;

    std::cout << "Printing parameters..." << std::endl;

    std::cout << configs.ParametersPrint();
    configs.ParametersSave(outputDir);

    mhd::CudaTimeCounter writerCounter;
    std::cout << "Creating writer... ";
    writerCounter.start();
    mhd::Writer writer(outputDir, configs);
    writerCounter.stop();
    std::cout << "Done. Time: " << writerCounter.getTime() << std::endl;

    mhd::CudaTimeCounter windowCounter;
    std::cout << "Creating window... ";
    windowCounter.start();
    opengl::Creater creater(configs);
    windowCounter.stop();
    std::cout << "Done. Time: " << windowCounter.getTime() << std::endl;

    mhd::CudaTimeCounter runCounter;
    runCounter.start();
    {
        mhd::CudaTimeCounter solverCounter;
        std::cout << "Creating solver... ";
        solverCounter.start();
        mhd::Solver solver(configs);
        solverCounter.stop();
        std::cout << "Done. Time: " << solverCounter.getTime() << std::endl;

        // Initial Conditions
        solver.fillNormally(static_cast<unsigned long>(std::time(nullptr)));

        // Saving fields from previous timelayer
        solver.saveOldFields();

        // Initial Energy and Time Step
        solver.updateEnergies();
        solver.updateTimeStep();

        std::cout << std::endl
                  << "Simulation starts..." << std::endl
                  << std::endl;

        // Initial Data Output
        writer.saveData(solver, creater);
        creater.PrepareToRun();

        // Main Cycle of the Program
        while (solver.shouldContinue() && creater.ShouldOpen()) {
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

            // Saving fields from previous timelayer
            solver.saveOldFields();

            // Update Parameters (Energy and Time Step)
            solver.updateEnergies();
            solver.updateTimeStep();
            solver.timeStep();

            // Data Output
            creater.Render(writer.saveData(solver, creater));

            creater.WindowUpdate();
        }
    }
    runCounter.stop();
    std::cout << std::endl
              << "Simulation time: " << runCounter.getTime() << std::endl;
}
