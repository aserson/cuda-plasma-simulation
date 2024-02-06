#include "Application.cuh"

#include <cufft.h>

#include "EquationSolver.cuh"
#include "Helper.cuh"
#include "InitialFields.cuh"
#include "IntegralFunctions.cuh"
#include "KernelCaller.cuh"
#include "Params.h"
#include "SimpleFunctions.cuh"
#include "Writer.cuh"

void cuda_main(const std::filesystem::path& outputDir) {
    cudaSetDevice(0);

    std::cout << "Printing parameters..." << std::endl;
    mhd::parameters::ParametersPrint();
    mhd::parameters::ParametersSave(outputDir);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int gridLength = mhd::parameters::SimulationParameters::gridLength;
    unsigned int minorGridLength =
        mhd::parameters::KernelRunParameters::gridSizeLinear;

    cudaEventRecord(start, 0);

    mhd::parameters::CurrentParameters params;
    FFTransformator transformator(gridLength);
    CalculatedFields calc(gridLength);
    AuxiliaryFields aux(gridLength, minorGridLength);
    Writer writer(gridLength);

    // Initial Conditions: Kinetic Part
    FillNormally(calc._streamFunction,
                 mhd::parameters::InitialCondition::averageWaveNumber,
                 time(NULL));

    double value =
        sqrt(mhd::parameters::InitialCondition::initialKineticEnergy /
             Energy(transformator, aux, calc._streamFunction));

    Normallize(calc._streamFunction, value);

    UpdateVorticity(calc);

    // Initial Conditions: Magnetic Part
    FillNormally(calc._magneticPotential,
                 mhd::parameters::InitialCondition::averageWaveNumber,
                 time(NULL) + 1);

    value = sqrt(mhd::parameters::InitialCondition::initialMagneticEnergy /
                 Energy(transformator, aux, calc._magneticPotential));

    Normallize(calc._magneticPotential, value);

    UpdateCurrent(calc);

    // Initial Energy
    params.kineticEnergy =
        EnergyDirty(transformator, aux, calc._streamFunction);
    params.magneticEnergy =
        EnergyDirty(transformator, aux, calc._magneticPotential);

    // Zero Time Step
    double cfl = mhd::parameters::SimulationParameters::cft;
    double gridStep = mhd::parameters::SimulationParameters::gridStep;
    double maxTimeStep = mhd::parameters::SimulationParameters::maxTimeStep;

    params.maxVelocityField = Max(transformator, aux, calc._streamFunction);
    params.maxMagneticField = Max(transformator, aux, calc._magneticPotential);

    params.timeStep = (params.maxMagneticField > params.maxVelocityField)
                          ? cfl * gridStep / params.maxMagneticField
                          : cfl * gridStep / params.maxVelocityField;

    if (params.timeStep > maxTimeStep)
        params.timeStep = maxTimeStep;

    // Zero Data Output
    writer.saveField<Vorticity>(calc._vorticity, outputDir, transformator, aux,
                                params);
    writer.saveField<StreamFunction>(calc._streamFunction, outputDir,
                                     transformator, aux, params);
    writer.saveField<Current>(calc._current, outputDir, transformator, aux,
                              params);
    writer.saveField<MagneticPotential>(calc._magneticPotential, outputDir,
                                        transformator, aux, params);
    writer.saveCurentParams(params, outputDir);

    params.stepNumberOut++;

    // Zero Consile Output
    std::cout
        << "---------------------------------------------------------------"
           "-------------"
        << std::endl;
    printf("%f\t%d\t%f\t%f\t%f\t%f\n", params.time, params.stepNumber,
           params.timeStep, params.kineticEnergy, params.magneticEnergy,
           params.kineticEnergy + params.magneticEnergy);

    // Main Cycle of the Program

    params.timeOut = mhd::parameters::OutputParameters::startTime;

    while (params.time < mhd::parameters::SimulationParameters::time) {
        params.stepNumber++;
        params.time += params.timeStep;

        // Saving fields from previous timelayer
        calc._vorticity.copyToDevice(aux._oldOne.data());
        calc._magneticPotential.copyToDevice(aux._oldTwo.data());

        // Time Integration Scheme
        // Two-step Scheme

        // First step
        KineticRightSideCalc(aux._rightPart, transformator, calc, aux);
        TimeScheme(calc._vorticity, aux._oldOne, aux._rightPart, params);

        MagneticRightSideCalc(aux._rightPart, transformator, calc, aux);
        TimeScheme(calc._magneticPotential, aux._oldTwo, aux._rightPart,
                   params);

        UpdateStreamFunction(calc);
        UpdateCurrent(calc);

        // Second step
        KineticRightSideCalc(aux._rightPart, transformator, calc, aux);
        TimeScheme(calc._vorticity, aux._oldOne, aux._rightPart, params);

        MagneticRightSideCalc(aux._rightPart, transformator, calc, aux);
        TimeScheme(calc._magneticPotential, aux._oldTwo, aux._rightPart,
                   params);

        UpdateStreamFunction(calc);
        UpdateCurrent(calc);

        // New Time Step
        params.maxVelocityField = Max(transformator, aux, calc._streamFunction);
        params.maxMagneticField =
            Max(transformator, aux, calc._magneticPotential);
        params.timeStep = (params.maxMagneticField > params.maxVelocityField)
                              ? cfl * gridStep / params.maxMagneticField
                              : cfl * gridStep / params.maxVelocityField;
        if (params.timeStep > maxTimeStep)
            params.timeStep = maxTimeStep;

        // Actual Eneries
        params.kineticEnergy = Energy(transformator, aux, calc._streamFunction);
        params.magneticEnergy =
            Energy(transformator, aux, calc._magneticPotential);

        // Data Output
        if (params.time >= params.timeOut) {
            writer.saveField<Vorticity>(calc._vorticity, outputDir,
                                        transformator, aux, params);
            writer.saveField<StreamFunction>(calc._streamFunction, outputDir,
                                             transformator, aux, params);
            writer.saveField<Current>(calc._current, outputDir, transformator,
                                      aux, params);
            writer.saveField<MagneticPotential>(
                calc._magneticPotential, outputDir, transformator, aux, params);
            writer.saveCurentParams(params, outputDir);

            params.timeOut += params.timeStepOut;
            params.stepNumberOut++;
        }

        printf("%f\t%d\t%f\t%f\t%f\t%f\n", params.time, params.stepNumber,
               params.timeStep, params.kineticEnergy, params.magneticEnergy,
               params.kineticEnergy + params.magneticEnergy);
    }

    // ProgramTime Output
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%3.1f\n", elapsedTime / 1000);

    std::ofstream globalOut("outputs/time.txt", std::ios::app);

    if (globalOut.is_open()) {
        globalOut << outputDir << ":" << std::endl
                  << "	Time = " << params.time << std::endl
                  << "	TimeStepsCount = " << params.stepNumber << std::endl
                  << "	GridLength = " << gridLength << std::endl
                  << "	Real Time = " << elapsedTime / 1000 << std::endl
                  << std::endl;
    }
    globalOut.close();

    // Memory Clearing

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
