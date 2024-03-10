#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace png {
struct RGB8 {
    unsigned char red;
    unsigned char green;
    unsigned char blue;

    RGB8() : red(0), green(0), blue(0) {}

    RGB8(unsigned char r, unsigned char g, unsigned char b)
        : red(r), green(g), blue(b) {}

    void setRed(unsigned char r) { red = r; }
    void setGreen(unsigned char g) { green = g; }
    void setBlue(unsigned char b) { blue = b; }
};

class Painter {
private:
    unsigned char* _pixels;

    unsigned int _sideLength;

    static const unsigned int _cmLength = 256;
    RGB8 _colorMap[_cmLength];

    FILE* fp;

public:
    Painter(unsigned int sideLength, const std::string& colorMapName);
    ~Painter();

    void readColorMap(const std::string& colorMapName);

    void fillBuffer(const double* src);
    void copyBuffer(unsigned char* dst);

    void saveAsPNG(const std::filesystem::path& path);

    unsigned int getSideLength() { return _sideLength; }
    unsigned char* getPixels() { return _pixels; }
};

}  // namespace png