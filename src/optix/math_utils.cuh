#pragma once

#include "device.cuh"

namespace RLpbr {
namespace optix {

namespace Constants {

static constexpr float nearestOne = float(0x1.fffffep-1);

}

// https://nullprogram.com/blog/2018/07/31/
constexpr inline uint32_t mix32(uint32_t v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;

    return v;
}

// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
constexpr inline uint2 mix32x2(const uint32_t v)
{
    uint64_t vbig = v;
    vbig ^= (vbig >> 31);
    vbig *= 0x7fb5d329728ea185;
    vbig ^= (vbig >> 27);
    vbig *= 0x81dadef4bc2dd44d;
    vbig ^= (vbig >> 33);

    return {
        uint32_t(vbig >> 32),
        uint32_t(vbig),
    };
}

constexpr inline uint32_t morton2D32(uint32_t x, uint32_t y) {
    // Space out bottom 16 bits with 0s
    auto space16 = [](uint32_t v) {
        v = (v ^ (v << 8)) & 0x00ff00ff;
        v = (v ^ (v << 4)) & 0x0f0f0f0f;
        v = (v ^ (v << 2)) & 0x33333333;
        v = (v ^ (v << 1)) & 0x55555555;
        return v;
    };

    return (space16(y) << 1) | space16(x);
}

inline uint32_t log2Down(uint32_t v)
{
    return 32 - __clz(v) - 1;
}

// NVCC doesn't support __builtin_clz and __clz isn't constexpr :(
constexpr inline uint32_t log2DownConst(uint32_t v)
{
    return v == 1 ? 0 : 1 + log2DownConst(v >> 1);
}

constexpr inline uint32_t maxConst(uint32_t a, uint32_t b) {
    if (a > b) return a;
    return b;
}


}
}
