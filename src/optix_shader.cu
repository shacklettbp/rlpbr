#include <optix.h>
#include "optix_device.cuh"

#include "optix_shader.hpp"

static constexpr int spp = (16);

struct HalfVec2 {
    half a;
    half b;
};

using namespace RLpbr::optix;
using namespace std;

extern "C" {
__constant__ ShaderParams params;
}

struct CameraRay {
    float3 origin;
    float3 direction;
};

struct DeviceVertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct Triangle {
    DeviceVertex a;
    DeviceVertex b;
    DeviceVertex c;
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

__device__ __forceinline__ float computeDepth()
{
    float3 scaled_dir = optixGetWorldRayDirection() * optixGetRayTmax();
    return length(scaled_dir);
}

__device__ __forceinline__ float3 computeBarycentrics()
{
    float2 attrs  = optixGetTriangleBarycentrics();

    return make_float3(1.f - attrs.x - attrs.y, attrs.x, attrs.y);
}

__device__ __forceinline__ unsigned int packHalfs(half a, half b)
{
    return (((unsigned int)__half_as_ushort(a)) << 16) + __half_as_ushort(b);
}

__device__ __forceinline__ HalfVec2 unpackHalfs(unsigned int v)
{
    uint16_t a = v >> 16;
    uint16_t b = v;

    return {
        __ushort_as_half(a),
        __ushort_as_half(b),
    };
}

__device__ __forceinline__ void setPayload(float r, float g, float b)
{
    half hr = __float2half(r);
    half hg = __float2half(g);
    half hb = __float2half(b);

    optixSetPayload_0(packHalfs(hr, hg));
    optixSetPayload_1(packHalfs(0, hb));
}

__device__ __forceinline__ void setOutput(half *base_output, 
                                          unsigned int payload_0,
                                          unsigned int payload_1)
{
    auto [r, g] = unpackHalfs(payload_0);
    auto [unused, b] = unpackHalfs(payload_1);

    base_output[0] = r;
    base_output[1] = g;
    base_output[2] = b;
}

__device__ __forceinline__ DeviceVertex unpackVertex(
    const PackedVertex &packed)
{
    float4 a = packed.data[0];
    float4 b = packed.data[1];

    return DeviceVertex {
        make_float3(a.x, a.y, a.z),
        make_float3(a.w, b.x, b.y),
        make_float2(b.z, b.w),
    };
}

__device__ __forceinline__ Triangle fetchTriangle(
    const PackedVertex *vertex_buffer,
    const uint32_t *index_start)
{
    return Triangle {
        unpackVertex(vertex_buffer[index_start[0]]),
        unpackVertex(vertex_buffer[index_start[1]]),
        unpackVertex(vertex_buffer[index_start[2]]),
    };
}

__device__ __forceinline__ DeviceVertex interpolateTriangle(
    const Triangle &tri, float3 barys)
{
    return DeviceVertex {
        tri.a.position * barys.x +
            tri.b.position * barys.y + 
            tri.c.position * barys.z,
        tri.a.normal * barys.x +
            tri.b.normal * barys.y +
            tri.c.normal * barys.z,
        tri.a.uv * barys.x +
            tri.b.uv * barys.y +
            tri.c.uv * barys.z,
    };
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    size_t base_out_offset = 
        3 * (idx.z * dim.y * dim.x + idx.y * dim.x + idx.x);

    uint batch_idx = idx.z;

    const CameraParams &cam = params.cameras[batch_idx];

    auto [ray_origin, ray_dir] = computeCameraRay(cam, idx, dim);

    // Trace the ray against our scene hierarchy
    unsigned int payload_0;
    unsigned int payload_1;
    optixTrace(
            params.accelStructs[batch_idx],
            ray_origin,
            ray_dir,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(0xff), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            0,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            payload_0,
            payload_1);

    setOutput(params.outputBuffer + base_out_offset, payload_0, payload_1);
}

extern "C" __global__ void __miss__ms()
{
    setPayload(0, 0, 0);
}

extern "C" __global__ void __closesthit__ch()
{
    const ClosestHitEnv &ch_env = params.envs[optixGetLaunchIndex().z];
    uint32_t index_offset =
        optixGetInstanceId() + 3 * optixGetPrimitiveIndex();
    Triangle hit_tri = fetchTriangle(ch_env.vertexBuffer,
                                     ch_env.indexBuffer + index_offset);
    float3 barys = computeBarycentrics();
    DeviceVertex interpolated = interpolateTriangle(hit_tri, barys);

    float3 world_normal = 
        optixTransformNormalFromObjectToWorldSpace(interpolated.normal);

    setPayload(world_normal.x,
               world_normal.y,
               world_normal.z);
}
