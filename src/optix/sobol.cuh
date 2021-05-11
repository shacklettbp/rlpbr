#pragma once

#include "device.cuh"

namespace RLpbr {
namespace optix {
namespace sobol {

static constexpr int matrixSize = 32;
static constexpr int maxDims = 2;

template <int dim>
inline uint32_t sample(uint32_t sobol_idx)
{
    uint32_t v = 0;

    [[maybe_unused]] uint32_t prev_vec;
    if constexpr (dim == 1) {
        prev_vec = 0x80000000;
    }
    for (int mat_idx = 0; mat_idx < matrixSize;
         sobol_idx >>= 1, mat_idx++) {

        uint32_t cur_vec;
        if constexpr (dim == 0) {
            cur_vec = 1 << (matrixSize - 1 - mat_idx);
        } else if constexpr (dim == 1) {
            cur_vec = prev_vec;

            prev_vec = prev_vec ^ (prev_vec >> 1);
        }

        if (sobol_idx & 1) v ^= cur_vec;
    }

    return v;
}

}
}
}
