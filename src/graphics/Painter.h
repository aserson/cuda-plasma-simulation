#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace graphics {
struct RGB8 {
    uint8_t red;
    uint8_t green;
    uint8_t blue;

    RGB8() : red(0), green(0), blue(0) {}

    RGB8(uint8_t r, uint8_t g, uint8_t b) : red(r), green(g), blue(b) {}

    void setRed(uint8_t r) { red = r; }
    void setGreen(uint8_t g) { green = g; }
    void setBlue(uint8_t b) { blue = b; }
};

class Painter {
private:
    std::vector<uint8_t> _image;

    unsigned int _sideLength;

    static const unsigned int _cmLength = 255;
    RGB8 _colorMap[_cmLength];

    FILE* fp;

public:
    Painter(unsigned int sideLength, const std::string& colorMapName);
    ~Painter();

    void readColorMap(const std::string& colorMapName);

    void fillBuffer(const double* src);

    void saveAsPNG(const std::filesystem::path& path);
};

}  // namespace graphics