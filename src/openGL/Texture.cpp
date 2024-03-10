#include "Texture.h"

#include <sstream>

#include "vendor/stb_image/stb_image.h"

Texture::Texture(unsigned int numOutputs)
    : m_LocalBuffer(nullptr), m_Width(0), m_Height(0), m_BPP(0) {

    GLCall(glGenTextures(numOutputs, m_RendererID));
}

Texture::~Texture() {
    GLCall(glDeleteTextures(32, m_RendererID));
}

void Texture::Bind(unsigned int slot) {
    GLCall(glActiveTexture(GL_TEXTURE0 + slot));
    GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID[slot]));
}

void Texture::Unbind() {
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}

void Texture::LoadTexture(const unsigned char* buffer, unsigned int width,
                          unsigned int height, unsigned int i) {
    GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID[i]));

    // Should be for all textures
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGB,
                        GL_UNSIGNED_BYTE, buffer));
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}
