#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace RLpbr {
namespace optix {

__host__ inline dim3 computeBlockDims(dim3 global_dims, dim3 tpb)
{
    return dim3 {
        (global_dims.x + tpb.x - 1) / tpb.x,
        (global_dims.y + tpb.y - 1) / tpb.y,
        (global_dims.z + tpb.z - 1) / tpb.z,
    };
}

__device__ inline dim3 globalThreadIdx()
{
    return dim3 {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z,
    };
}

}
}
