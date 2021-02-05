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
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    // Dim 2
    0x00000001, 0x00000003, 0x00000005, 0x0000000f,
    0x00000011, 0x00000033, 0x00000055, 0x000000ff,
    0x00000101, 0x00000303, 0x00000505, 0x00000f0f,
    0x00001111, 0x00003333, 0x00005555, 0x0000ffff,
    0x00010001, 0x00030003, 0x00050005, 0x000f000f,
    0x00110011, 0x00330033, 0x00550055, 0x00ff00ff,
    0x01010101, 0x03030303, 0x05050505, 0x0f0f0f0f,
    0x11111111, 0x33333333, 0x55555555, 0xffffffff,
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
