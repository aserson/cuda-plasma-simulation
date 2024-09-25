#include "openGL/VertexBuffer.h"

#include "openGL/Renderer.h"

VertexBuffer::VertexBuffer(const void* data, unsigned int count) {
    GLCall(glGenBuffers(1, &m_RendererID));
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
    GLCall(glBufferData(GL_ARRAY_BUFFER, count * sizeof(float), data,
                        GL_STATIC_DRAW));
}

VertexBuffer::~VertexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererID));
}

void VertexBuffer::Bind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
}

void VertexBuffer::Unbing() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
