#pragma once

#include "device.cuh"

namespace RLpbr {
namespace optix {
namespace sobol {

static constexpr int matrixSize = 32;
static constexpr int maxDims = 2;

// S. Joe and F. Y. Kuo: "Constructing Sobol sequences with better
// two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008).
// These numbers are reversed to save a reverse instruction before scrambling
static const uint32_t generatorMatrix[maxDims * matrixSize] = {
    // Dim 1
    0x80000000, 0x40000000, 0x20000000, 0x10000000,
    0x08000000, 0x04000000, 0x02000000, 0x01000000,
    0x00800000, 0x00400000, 0x00200000, 0x00100000,
    0x00080000, 0x00040000, 0x00020000, 0x00010000,
    0x00008000, 0x00004000, 0x00002000, 0x00001000,
    0x00000800, 0x00000400, 0x00000200, 0x00000100,
    0x00000080, 0x00000040, 0x00000020, 0x00000010,
    0x00000008, 0x00000004, 0x00000002, 0x00000001,
    // Dim 2
    0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
    0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
    0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
    0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
    0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
    0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
    0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
    0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
};

inline uint32_t sample(uint32_t sobol_idx, uint32_t dim)
{
    uint32_t v = 0;
    for (int mat_idx = dim * matrixSize; sobol_idx != 0;
         sobol_idx >>= 1, mat_idx++) {
        if (sobol_idx & 1) v ^= generatorMatrix[mat_idx];
    }

    return v;
}

}
}
}
