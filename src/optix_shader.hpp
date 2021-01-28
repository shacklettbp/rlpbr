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

struct ShaderParams {
    half *outputBuffer;
    OptixTraversableHandle *accelStructs;
    CameraParams *cameras;

    ClosestHitEnv *envs;
};

}
}
