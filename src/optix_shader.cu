#include <optix.h>
#include "optix_device.cuh"

#include "optix_shader.hpp"

#include <cuda/std/tuple>

using namespace RLpbr::optix;
using namespace cuda::std;

extern "C" {
__constant__ ShaderParams params;
}

struct RTParams {
    static constexpr int spp = (SPP);
    static constexpr int maxDepth = (MAX_DEPTH);
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

class RNG {
public:
    __inline__ RNG(uint seed, uint frame_idx)
        : v_(seed)
    {
        uint v1 = frame_idx;
        uint s0 = 0;

        for (int n = 0; n < 4; n++) {
            s0 += 0x9e3779b9;
            v_ += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
            v1 += ((v_<<4)+0xad90777d)^(v_+s0)^((v_>>5)+0x7e95761e);
        }
    }

    __inline__ float sample1D()
    {
        return (float)next() / (float)0x01000000;
    }

    __inline__ float2 sample2D()
    {
        return make_float2(sample1D(), sample1D());
    }

private:
    __inline__ uint next()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        v_ = (LCG_A * v_ + LCG_C);
        return v_ & 0x00FFFFFF;
    }

    uint v_;
};

static RNG initRNG(uint3 idx, uint3 dim, uint sample_idx, uint path_idx,
                   uint frame_idx)
{
    uint seed = ((idx.z * dim.y + (idx.y * dim.x + dim.x)) * 
        RTParams::spp + sample_idx) * RTParams::maxDepth + path_idx;

    return RNG(seed, frame_idx);
}

__forceinline__ pair<float3, float3> computeCameraRay(
    const CameraParams &camera, uint3 idx, uint3 dim, RNG &rng)
{
    float4 data0 = camera.data[0];
    float4 data1 = camera.data[1];
    float4 data2 = camera.data[2];

    float3 origin = make_float3(data0.x, data0.y, data0.z);
    float3 view = make_float3(data0.w, data1.x, data1.y);
    float3 up = make_float3(data1.z, data1.w, data2.x);
    float3 right = make_float3(data2.y, data2.z, data2.w);

    float2 jittered_raster = make_float2(idx.x, idx.y) + rng.sample2D();

    float2 screen = make_float2((2.f * jittered_raster.x) / dim.x - 1,
                                (2.f * jittered_raster.y) / dim.y - 1);

    float3 direction = right * screen.x + up * screen.y + view;

    return {
        origin,
        direction,
    };
}

__forceinline__ float computeDepth()
{
    float3 scaled_dir = optixGetWorldRayDirection() * optixGetRayTmax();
    return length(scaled_dir);
}

__forceinline__ float3 computeBarycentrics()
{
    float2 attrs  = optixGetTriangleBarycentrics();

    return make_float3(1.f - attrs.x - attrs.y, attrs.x, attrs.y);
}

__forceinline__ unsigned int packHalfs(half a, half b)
{
    return (((unsigned int)__half_as_ushort(a)) << 16) + __half_as_ushort(b);
}

__forceinline__ pair<half, half> unpackHalfs(unsigned int v)
{
    uint16_t a = v >> 16;
    uint16_t b = v;

    return {
        __ushort_as_half(a),
        __ushort_as_half(b),
    };
}

__forceinline__ void setPayload(float r, float g, float b)
{
    half hr = __float2half(r);
    half hg = __float2half(g);
    half hb = __float2half(b);

    optixSetPayload_0(packHalfs(hr, hg));
    optixSetPayload_1(packHalfs(0, hb));
}

__forceinline__ void setOutput(half *base_output, float3 rgb)
{
    base_output[0] = __float2half(rgb.x);
    base_output[1] = __float2half(rgb.y);
    base_output[2] = __float2half(rgb.z);
}

__forceinline__ DeviceVertex unpackVertex(
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

__forceinline__ Triangle fetchTriangle(
    const PackedVertex *vertex_buffer,
    const uint32_t *index_start)
{
    return Triangle {
        unpackVertex(vertex_buffer[index_start[0]]),
        unpackVertex(vertex_buffer[index_start[1]]),
        unpackVertex(vertex_buffer[index_start[2]]),
    };
}

__forceinline__ DeviceVertex interpolateTriangle(
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
    
    float3 pixel_radiance = make_float3(0.f);

#if SPP != 1
#pragma unroll 1
#endif
    for (int32_t sample_idx = 0; sample_idx < RTParams::spp; sample_idx++) {
        float3 sample_radiance = make_float3(0.f);

#if MAX_DEPTH != 1
#pragma unroll 1
#endif
        for (int32_t path_depth = 0; path_depth < RTParams::maxDepth;
             path_depth++) {
            RNG rng = initRNG(idx, dim, sample_idx, path_depth, 0);

            float3 ray_origin;
            float3 ray_dir;
            if (path_depth == 0) {
                tie(ray_origin, ray_dir) = computeCameraRay(cam, idx, dim, rng);
            } else {
                // Compute bounce
            }

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

            auto [r, g] = unpackHalfs(payload_0);
            auto [unused, b] = unpackHalfs(payload_1);

            sample_radiance.x += __half2float(r);
            sample_radiance.y += __half2float(g);
            sample_radiance.z += __half2float(b);
        }

        pixel_radiance += sample_radiance / RTParams::spp;
    }

    setOutput(params.outputBuffer + base_out_offset, pixel_radiance);
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
