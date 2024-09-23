cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0146 NEW)

project(cuda-mhd-plasma-simulation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR}/bin)

file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")
file(GLOB GL_SOURCES "${CMAKE_SOURCE_DIR}/src/opengl/*.cpp")

source_group("Source Files" FILES ${CPP_SOURCES})
source_group("Source Files\\Cuda" FILES ${CUDA_SOURCES})
source_group("Source Files\\OpenGL" FILES ${GL_SOURCES})

file(GLOB CPP_HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB CUDA_HEADERS "${CMAKE_SOURCE_DIR}/include/cuda/*.cuh")
file(GLOB GL_HEADERS "${CMAKE_SOURCE_DIR}/include/opengl/*.h")

source_group("Headers Files" FILES ${CPP_HEADERS})
source_group("Headers Files\\Cuda" FILES ${CUDA_HEADERS})
source_group("Headers Files\\OpenGL" FILES ${GL_HEADERS})

add_executable(simulation ${CPP_SOURCES} ${CUDA_SOURCES} ${GL_SOURCES} ${CPP_HEADERS} ${CUDA_HEADERS} ${GL_HEADERS})

target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glew/include)
target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glfw/include)
target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/yaml/include)

target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glew/lib)
target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glfw/lib)
target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/yaml/lib)

target_compile_definitions(simulation PRIVATE YAML_CPP_STATIC_DEFINE)
target_compile_definitions(simulation PRIVATE GLEW_STATIC)

target_link_libraries(simulation
    glew32s.lib
    glfw3.lib
    opengl32.lib
    cufft.lib
    debug yaml-cppd.lib
    optimized yaml-cpp.lib
)

set_target_properties(simulation PROPERTIES CUDA_ARCHITECTURES "86")
target_link_options(simulation PRIVATE /NODEFAULTLIB:LIBCMT)
