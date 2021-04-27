#pragma once

#include <optix.h>
#include <cuda_fp16.h>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef __CUDACC__
#include <array>
#endif

#include <rlpbr_core/device.hpp>

namespace RLpbr {
namespace optix {

struct DevicePackedVertex {
    float4 data[2];
};

struct PackedMaterial {
    uint4 data[2];
};

struct PackedMeshInfo {
    uint4 data;
};

struct PackedInstance {
    uint32_t materialOffset;
    uint32_t meshOffset;
};

struct PackedLight {
    float4 data[2];
};

struct PackedTransforms {
    float4 data[6];
};

struct TextureSize {
    uint32_t width;
    uint32_t height;
};

struct alignas(16) PackedEnv {
#ifdef __CUDACC__
    float4 camData[3];
#else
    std::array<float4, 3> camData;
#endif
    OptixTraversableHandle tlas;
    const DevicePackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
    const PackedMaterial *materialBuffer;
    const cudaTextureObject_t *textureHandles;
    const TextureSize *textureDims;
    const PackedMeshInfo *meshInfos;

    // FIXME Turn instance and light pointers
    // into uint32_t offsets into constant 64 bit
    // pointer
    const PackedInstance *instances;
    const uint32_t *instanceMaterials;
    const PackedLight *lights;
    const PackedTransforms *transforms;
    uint32_t numLights;
};

struct BSDFPrecomputed {
    cudaTextureObject_t diffuseAverage;
    cudaTextureObject_t diffuseDirectional;
    cudaTextureObject_t microfacetAverage;
    cudaTextureObject_t microfacetDirectional;
    cudaTextureObject_t microfacetDirectionalInverse;
};

struct alignas(16) LaunchInput {
    uint32_t baseBatchOffset;
    uint32_t baseFrameCounter;
    BSDFPrecomputed precomputed;
};

}
}
