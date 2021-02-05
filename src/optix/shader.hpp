#pragma once

#include <optix.h>
#include <cuda_fp16.h>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace RLpbr {
namespace optix {

struct PackedVertex {
    float4 data[2];
};

struct CameraParams {
    float4 data[3];
};

struct ClosestHitEnv {
    const PackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
};

struct PackedEnv {
    float4 camData[3];
    const PackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
    OptixTraversableHandle tlas;
};

struct alignas(16) LaunchInput {
    uint32_t baseBatchOffset;
    uint32_t baseFrameCounter;
};

}
}
