cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0146 NEW)

project(cuda-mhd-plasma-simulation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")
file(GLOB GL_SOURCES "${CMAKE_SOURCE_DIR}/src/openGL/*.cpp")

file(GLOB CPP_HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB CUDA_HEADERS "${CMAKE_SOURCE_DIR}/include/cuda/*.cuh")
file(GLOB GL_HEADERS "${CMAKE_SOURCE_DIR}/include/openGL/*.h")

file(GLOB CONFIGS "${CMAKE_SOURCE_DIR}/configs/*.yaml")

add_executable(simulation ${CPP_SOURCES} ${CUDA_SOURCES} ${GL_SOURCES} ${CPP_HEADERS} ${CUDA_HEADERS} ${GL_HEADERS} ${CONFIGS})

target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(simulation PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(WIN32)
    source_group("Source Files" FILES ${CPP_SOURCES})
    source_group("Source Files\\cuda" FILES ${CUDA_SOURCES})
    source_group("Source Files\\openGL" FILES ${GL_SOURCES})

    source_group("Headers Files" FILES ${CPP_HEADERS})
    source_group("Headers Files\\cuda" FILES ${CUDA_HEADERS})
    source_group("Headers Files\\openGL" FILES ${GL_HEADERS})

    source_group("Configuration Files" FILES ${CONFIGS})

    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glew/include)
    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glfw/include)
    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/yaml/include)

    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glew/lib)
    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glfw/lib)
    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/yaml/lib)

    target_link_libraries(simulation
        glew32s.lib
        glfw3.lib
        opengl32.lib
        cufft.lib
        debug yaml-cppd.lib
        optimized yaml-cpp.lib
    )

    target_link_options(simulation PRIVATE /NODEFAULTLIB:LIBCMT)
elseif(UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED glfw3)
    pkg_check_modules(GLEW REQUIRED glew)
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

    target_include_directories(simulation PRIVATE ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
    target_include_directories(simulation PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    target_link_libraries(simulation
        ${GLEW_LIBRARIES}
        ${GLFW_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        GL
        cufft
    )
endif()

set_target_properties(simulation PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_definitions(simulation PRIVATE YAML_CPP_STATIC_DEFINE)
target_compile_definitions(simulation PRIVATE GLEW_STATIC)
