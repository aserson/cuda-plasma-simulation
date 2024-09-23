#pragma once

class IndexBuffer {
private:
    unsigned int m_RendererID;
    size_t m_Count;

public:
    IndexBuffer(const unsigned int* data, unsigned int count);
    ~IndexBuffer();

    void Bind() const;
    void Unbing() const;

    inline size_t GetCount() const { return m_Count; }
};
