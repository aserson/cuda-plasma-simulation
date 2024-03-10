#include "Painter.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "fpng.h"

namespace png {
Painter::Painter(unsigned int sideLength, const std::string& colorMapName)
    : _sideLength(sideLength) {

    _pixels = new unsigned char[_sideLength * _sideLength * 3];
    readColorMap(colorMapName);
}

Painter::~Painter() {
    delete[] _pixels;
}

void Painter::readColorMap(const std::string& colorMapName) {
    std::filesystem::path filePath("res/colormaps");
    filePath /= colorMapName;

    float red, green, blue;
    unsigned int i = 0;

    std::ifstream file(filePath);
    if (file.is_open()) {
        while ((file >> red >> green >> blue) && (i < _cmLength)) {
            _colorMap[i].red = (unsigned char)(float(255) * red);
            _colorMap[i].green = (unsigned char)(float(255) * green);
            _colorMap[i].blue = (unsigned char)(float(255) * blue);
            i++;
        }
    }
    file.close();
}

void Painter::fillBuffer(const double* src) {
    double amplitude = fabs(src[0]);
    for (unsigned int i = 1; i < _sideLength * _sideLength; i++) {
        amplitude = std::max(amplitude, fabs(src[i]));
    }

    for (unsigned int i = 0; i < _sideLength; i++) {
        for (unsigned int j = 0; j < _sideLength; j++) {
            double value =
                255. * (amplitude + src[i * _sideLength + j]) / (2 * amplitude);
            uint8_t point = uint8_t(value);

            _pixels[i * (_sideLength * 3) + j * 3 + 0] = _colorMap[point].red;
            _pixels[i * (_sideLength * 3) + j * 3 + 1] = _colorMap[point].green;
            _pixels[i * (_sideLength * 3) + j * 3 + 2] = _colorMap[point].blue;
        }
    }
}

void Painter::copyBuffer(unsigned char* dst) {
    for (unsigned int i = 0; i < _sideLength * _sideLength * 3; ++i) {
        dst[i] = _pixels[i];
    }
}

void Painter::saveAsPNG(const std::filesystem::path& filePath) {
    fpng::fpng_encode_image_to_file(filePath.string().c_str(), _pixels,
                                    _sideLength, _sideLength, 3);
}
}  // namespace png