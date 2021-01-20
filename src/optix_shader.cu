#include <optix.h>

#include "optix_shader.hpp"
#include "optix_device.cuh"

using namespace RLpbr::optix;

extern "C" {
__constant__ ShaderParams params;
}

struct CameraRay {
    float3 origin;
    float3 direction;
};

__device__ __forceinline__ CameraRay computeCameraRay(
    const CameraParams &camera, uint3 idx, uint3 dim)
{
    float4 data0 = camera.data[0];
    float4 data1 = camera.data[1];
    float4 data2 = camera.data[2];

    float3 origin = make_float3(data0.x, data0.y, data0.z);
    float3 view = make_float3(data0.w, data1.x, data1.y);
    float3 up = make_float3(data1.z, data1.w, data2.x);
    float3 right = make_float3(data2.y, data2.z, data2.w);

    float2 screen = make_float2((2.f * idx.x + 1) / dim.x - 1,
                                (2.f * idx.y + 1) / dim.y - 1);

    float3 direction = right * screen.x + up * screen.y + view;

    return CameraRay {
        origin,
        direction,
    };
}

__device__ __forceinline__ float3 computeBarycentrics()
{
    float2 attrs  = optixGetTriangleBarycentrics();

    return make_float3(1.f - attrs.x - attrs.y, attrs.x, attrs.y);
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    uint batch_idx = idx.z;

    const CameraParams &cam = params.cameras[batch_idx];

    auto [ray_origin, ray_dir] = computeCameraRay(cam, idx, dim);

    // Trace the ray against our scene hierarchy
    unsigned int payload_0;
    optixTrace(
            params.accelStructs[batch_idx],
            ray_origin,
            ray_dir,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            payload_0);

    float depth = int_as_float(payload_0);

    // Record results in our output raster
    params.outputBuffer[idx.z * dim.y * dim.x + idx.y * dim.x + idx.x] = depth;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(float_as_int(0.f));
}

extern "C" __global__ void __closesthit__ch()
{
    float3 scaled_dir = optixGetWorldRayDirection() * optixGetRayTmax();

    float depth = length(scaled_dir);

    optixSetPayload_0(float_as_int(depth));
}
