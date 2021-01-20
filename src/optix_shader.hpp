#pragma once

#include <optix.h>

#include <cuda_runtime.h>

namespace RLpbr {
namespace optix {

struct CameraParams {
    float4 data[3];
};

struct ShaderParams {
    float *outputBuffer;
    OptixTraversableHandle *accelStructs;
    CameraParams *cameras;
};

}
}
