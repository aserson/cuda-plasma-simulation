#include "Application.cuh"

#include <ctime>

#include "Params.h"
#include "Solver.cuh"

void cuda_main(const std::filesystem::path& outputDir) {
    cudaSetDevice(0);

    std::cout << "Printing parameters..." << std::endl;
    mhd::parameters::ParametersPrint();
    mhd::parameters::ParametersSave(outputDir);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    {
        mhd::Solver solver;

        // Initial Conditions
        solver.fillNormally(std::time(nullptr));

        // Initial Energy and Time Step
        solver.updateEnergies();
        solver.updateTimeStep();

        // Zero Data Output
        solver.saveData(outputDir);
        solver.printCurrentParams();

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
            solver.printCurrentParams();
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    std::cout << std::endl;

    // CalculationTime Output
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Calculation time: " << elapsedTime / 1000 << std::endl;

    // Memory Clearing
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
