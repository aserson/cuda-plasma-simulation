#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <filesystem>

void cuda_main(const std::filesystem::path& outputDir);
