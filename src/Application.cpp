#include <iostream>

// This libraries needs to create folder with date name
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include "Application.cuh"

std::string CreateOutputDir(const std::filesystem::path& parentDir) {
    if (std::filesystem::exists(parentDir) == false)
        std::filesystem::create_directory(parentDir);

    auto currentFullTime = std::time(nullptr);

    std::ostringstream currenTimeStream;
    currenTimeStream << std::put_time(std::localtime(&currentFullTime),
                                      "%Y%m%d")
                     << "_"
                     << std::put_time(std::localtime(&currentFullTime),
                                      "%H%M%S");
    auto outputDir = parentDir / currenTimeStream.str();

    std::filesystem::create_directory(outputDir);
    std::filesystem::create_directory(outputDir / "vorticity");
    std::filesystem::create_directory(outputDir / "current");
    std::filesystem::create_directory(outputDir / "stream");
    std::filesystem::create_directory(outputDir / "potential");

    return outputDir.string();
}

void main() {
    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::cout << "Creating output directory... " << std::endl;

    const std::string outputDir = CreateOutputDir("outputs");

    std::cout << "Output directory: " << outputDir << std::endl << std::endl;

    // std::cout << "Done" << std::endl;

    cuda_main(outputDir);
}