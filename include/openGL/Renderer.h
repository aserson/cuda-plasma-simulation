#pragma once

#include <GL/glew.h>

#include "IndexBuffer.h"
#include "Shader.h"
#include "VertexArray.h"

#ifdef _MSC_VER
#include <intrin.h>
#define break() __debugbreak()
#else
#include <csignal>
#define break() raise(SIGTRAP);
#endif

#define ASSERT(x) \
    if (!(x))     \
        break();

#define GLCall(x)   \
    GLClearError(); \
    x;              \
    ASSERT(GLLogError(#x, __FILE__, __LINE__));

void GLClearError();

bool GLLogError(const char* function, const char* file, int line);

class Renderer {
private:
public:
    void Clear() const;
    void Draw(const VertexArray& va, const IndexBuffer& ib,
              const Shader& shader) const;
};