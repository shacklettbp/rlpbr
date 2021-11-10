#include <OpenImageIO/imageio.h>
#include <iostream>

using namespace std;

namespace RLpbr {

void saveSDR(const char *fname, uint32_t width, uint32_t height,
             const uint8_t *data)
{
    using namespace OIIO;

    auto png_out = ImageOutput::create(fname);
    if (!png_out) {
        cerr << "Failed to write SDR image" << endl;
        return;
    }

    ImageSpec spec(width, height, 3, TypeDesc::UINT8);
    png_out->open(fname, spec);
    png_out->write_image(TypeDesc::UINT8, data);
    png_out->close();
}

void saveHDR(const char *fname, uint32_t width, uint32_t height,
             const void *data, bool half)
{
    using namespace OIIO;

    auto hdr_out = ImageOutput::create(fname);
    if (!hdr_out) {
        cerr << "Failed to write SDR image" << endl;
        return;
    }

    TypeDesc type;
    if (half) {
        type = TypeDesc::HALF;
    } else {
        type = TypeDesc::FLOAT;
    }

    ImageSpec spec(width, height, 3, type);
    hdr_out->open(fname, spec);
    hdr_out->write_image(type, data);
    hdr_out->close();
}

}
