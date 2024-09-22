#include "IndexBuffer.h"

#include "Renderer.h"

IndexBuffer::IndexBuffer(const unsigned int* data, size_t count)
    : m_Count(count) {
    ASSERT(sizeof(size_t) == sizeof(GLuint));

    GLCall(glGenBuffers(1, &m_RendererID));
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int),
                        data, GL_STATIC_DRAW));
}

IndexBuffer::~IndexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererID));
}

void IndexBuffer::Bind() const {
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
}

void IndexBuffer::Unbing() const {
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}
