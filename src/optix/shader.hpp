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

struct PackedMaterial {
    // Actually a mix of integers and floats
    float4 data[2];
};

struct PackedInstance {
    uint32_t materialIdx;
};

struct PackedLight {
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
    const PackedMaterial *materialBuffer;
    const cudaTextureObject_t *textureHandles;

    // FIXME Turn instance and light pointers
    // into uint32_t offsets into constant 64 bit
    // pointer
    const PackedInstance *instances;
    const PackedLight *lights;
    uint32_t numLights;
};

struct alignas(16) LaunchInput {
    uint32_t baseBatchOffset;
    uint32_t baseFrameCounter;
};

}
}
