#pragma once

#include "Renderer.h"

class Texture {
private:
    unsigned int m_RendererID[32];
    unsigned char* m_LocalBuffer;
    int m_Width, m_Height, m_BPP;

public:
    Texture(unsigned int numOutputs);
    ~Texture();

    void Bind(unsigned int slot);
    void Unbind();

    void LoadTexture(const unsigned char* buffer, unsigned int width,
                     unsigned int height, unsigned int i);

    inline int GetWidth() const { return m_Width; }
    inline int GetHeight() const { return m_Height; }
};