#include "Painter.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "fpng.h"

namespace graphics {
Painter::Painter(unsigned int sideLength, const std::string& colorMapName)
    : _sideLength(sideLength) {
    _image.resize(_sideLength * _sideLength * 3);

    readColorMap(colorMapName);
}

Painter::~Painter() {}

void Painter::readColorMap(const std::string& colorMapName) {
    std::filesystem::path filePath("res/colormaps");
    filePath /= colorMapName;

    float red, green, blue;
    unsigned int i = 0;

    std::ifstream file(filePath);
    if (file.is_open()) {
        while ((file >> red >> green >> blue) && (i < _cmLength)) {
            _colorMap[i].red = uint8_t(float(255) * red);
            _colorMap[i].green = uint8_t(float(255) * green);
            _colorMap[i].blue = uint8_t(float(255) * blue);
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

            _image[i * (_sideLength * 3) + j * 3 + 0] = _colorMap[point].red;
            _image[i * (_sideLength * 3) + j * 3 + 1] = _colorMap[point].green;
            _image[i * (_sideLength * 3) + j * 3 + 2] = _colorMap[point].blue;
        }
    }
}

void Painter::saveAsPNG(const std::filesystem::path& filePath) {
    fpng::fpng_encode_image_to_file(filePath.string().c_str(), _image.data(),
                                    _sideLength, _sideLength, 3);
}
}  // namespace graphics