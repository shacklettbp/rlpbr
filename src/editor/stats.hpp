#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace RLpbr {

struct MeanVarContext {
    int64_t numPixels;
    int numBlocks;
    double *scratch;
    cudaStream_t strm;
};

MeanVarContext getMeanVarContext(uint32_t batch_size,
                                 uint32_t res_x,
                                 uint32_t res_y);

std::pair<std::array<float, 3>, std::array<float, 3>> computeMeanAndVar(
    half *batch, const MeanVarContext &ctx);

}
