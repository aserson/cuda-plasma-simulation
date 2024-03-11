#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <array>
#include <filesystem>
#include <string>

#include "Renderer.h"
#include "Shader.h"
#include "Textures.h"
#include "VertexArray.h"
#include "VertexBuffer.h"

namespace opengl {

class Creater {
public:
    static const std::array<float, 16> _positions;
    static const std::array<unsigned int, 6> _indices;

    Creater(unsigned int numOutputs, unsigned int width, unsigned int height);
    ~Creater();

    bool ShouldOpen();
    void AddTexture(const unsigned char* buffer, unsigned int width,
                    unsigned int height);
    void PrepareToRun();
    void Render(bool shouldUpdateTexture);
    void WindowUpdate();

private:
    unsigned int _numOutputs;
    int _currentTextureIndex;
    double _lastTextureChangeTime;
    unsigned int _frames = 0;
    double _lastTime;

    int Create(unsigned int width, unsigned int height);
    void CreateWindowData();
    void UpdateFPSCounter();

    GLFWwindow* _window;
    Shader* _shader;
    Textures* _textures;
    VertexArray* _va;
    VertexBuffer* _vb;
    IndexBuffer* _ib;

    Renderer* _renderer;
};

inline const std::array<float, 16> Creater::_positions = {
    -1.f, -1.f, 0.0f, 0.0f,  // 0
    1.f,  -1.f, 1.0f, 0.0f,  // 1
    1.f,  1.f,  1.0f, 1.0f,  // 2
    -1.f, 1.f,  0.0f, 1.0f   // 3
};

inline const std::array<unsigned int, 6> Creater::_indices = {
    0, 1, 2,  // first triangle
    2, 3, 0   // second triangle
};
}  // namespace opengl
