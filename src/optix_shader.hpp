#pragma once

#include <optix.h>
#include <cuda_fp16.h>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace RLpbr {
namespace optix {

struct CameraParams {
    float4 data[3];
};

struct ShaderParams {
    half *outputBuffer;
    OptixTraversableHandle *accelStructs;
    CameraParams *cameras;
};

}
}
