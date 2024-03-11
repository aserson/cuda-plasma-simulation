#include "Creater.h"

#include <iostream>

#include "VertexBufferLayout.h"

namespace opengl {

Creater::Creater(unsigned int numOutputs, unsigned int width,
                 unsigned int height)
    : _numOutputs(numOutputs),
      _currentTextureIndex(0),
      _lastTextureChangeTime(0) {
    Create(width, height);

    /* Make the window's context current */
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GLCall(glEnable(GL_BLEND));

    if (glewInit() != GLEW_OK)
        std::cout << "Error!" << std::endl;

    _vb = new VertexBuffer(_positions.data(), _positions.size());
    _va = new VertexArray();
    _ib = new IndexBuffer(_indices.data(), _indices.size());

    CreateWindowData();

    _shader = new Shader("res/shaders/Basic.shader");
    _textures = new Textures(numOutputs);
    _renderer = new Renderer();
}

Creater::~Creater() {
    delete _vb;
    delete _va;
    delete _ib;

    delete _shader;
    delete _textures;

    delete _renderer;

    glfwTerminate();
}

int Creater::Create(unsigned int width, unsigned int height) {
    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    _window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!_window) {
        glfwTerminate();
        return -1;
    }
}

void Creater::CreateWindowData() {
    VertexBufferLayout layout;
    layout.Push<float>(2);
    layout.Push<float>(2);
    _va->AddBuffer(*_vb, layout);
    _vb->Unbing();
    _va->Unbind();
    _ib->Unbing();
}

void Creater::PrepareToRun() {
    _shader->Bind();
    _textures->Bind(0);
    _shader->SetUniform1i("u_Texture", 0);
    _lastTime = glfwGetTime();
}

void Creater::UpdateFPSCounter() {
    double currentTime = glfwGetTime();
    double delta = currentTime - _lastTime;
    _frames++;

    if (delta >= 1.0) {
        double fps = double(_frames) / delta;

        std::string title = "FPS: " + std::to_string(fps);
        glfwSetWindowTitle(_window, title.c_str());

        _frames = 0;
        _lastTime = currentTime;
    }
}

void Creater::AddTexture(const unsigned char* buffer, unsigned int width,
                         unsigned int height) {
    _textures->LoadTexture(buffer, width, height, _currentTextureIndex % 32);
}

bool Creater::ShouldOpen() {
    return !glfwWindowShouldClose(_window);
}

void Creater::Render(bool shouldUpdateTexture) {
    _renderer->Clear();
    _shader->Bind();

    if (shouldUpdateTexture) {
        _currentTextureIndex = _currentTextureIndex % 32;

        _textures->Bind(_currentTextureIndex);
        _shader->SetUniform1i("u_Texture", _currentTextureIndex);
        _currentTextureIndex++;
    }

    _renderer->Draw(*_va, *_ib, *_shader);

    UpdateFPSCounter();
}

void Creater::WindowUpdate() {
    /* Swap front and back buffers */
    glfwSwapBuffers(_window);

    /* Poll for and process events */
    glfwPollEvents();
}
}  // namespace opengl
