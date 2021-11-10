#pragma once

namespace RLpbr {

void saveSDR(const char *fname, uint32_t width, uint32_t height,
             const uint8_t *data);

void saveHDR(const char *fname, uint32_t width, uint32_t height,
             const void *data, bool half);

}
