#pragma once

#include <optix.h>
#include <cuda_fp16.h>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef __CUDACC__
#include <array>
#endif

namespace RLpbr {
namespace optix {

struct PackedVertex {
    float4 data[2];
};

struct alignas(16) PackedEnv {
#ifdef __CUDACC__
    float4 camData[3];
#else
    std::array<float4, 3> camData;
#endif
    OptixTraversableHandle tlas;
    const PackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
};

struct alignas(16) LaunchInput {
    uint32_t baseBatchOffset;
    uint32_t baseFrameCounter;
};

}
}
