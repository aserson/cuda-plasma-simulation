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
#include "png/Painter.h"

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

int main() {
    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::cout << "Reading configurations file... " << std::endl;

    mhd::Configs configs("configs/standart1024.yaml");

    std::cout << "Creating output directory... " << std::endl;

    const std::filesystem::path outputDir = CreateOutputDir(configs);

    std::cout << "Output directory: " << outputDir << std::endl << std::endl;

    std::cout << "Printing parameters..." << std::endl;

    std::cout << configs.ParametersPrint();
    configs.ParametersSave(outputDir);

    std::cout << "Creating writer..." << std::endl;
    mhd::Writer writer(outputDir, configs, "plasma");

    std::cout << "Creating painter..." << std::endl;
    png::Painter painter(configs._gridLength, "plasma");

    std::cout << "Creating window..." << std::endl;
    opengl::Creater creater(32, 1000, 1000);

    mhd::CudaTimeCounter counter;

    counter.start();
    {
        mhd::Solver solver(configs);

        // Initial Conditions
        solver.fillNormally(static_cast<unsigned long>(std::time(nullptr)));

        // Initial Energy and Time Step
        solver.updateEnergies();
        solver.updateTimeStep();

        std::cout << "Simulation starts..." << std::endl;

        // Initial Data Output
        writer.saveData(solver, painter, creater);
        creater.PrepareToRun();

        // Main Cycle of the Program
        while (solver.shouldContinue() && creater.ShouldOpen()) {
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
            if (configs._showGraphics) {
                creater.Render(writer.saveData(solver, painter, creater));
            } else {
                writer.saveData(solver, painter, creater);
            }

            creater.WindowUpdate();
        }
    }
    counter.stop();

    // Calculation Time Output
    std::cout << std::endl
              << "Simulation time: " << counter.getTime() << std::endl;
}
