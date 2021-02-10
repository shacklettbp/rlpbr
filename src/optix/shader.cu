#include "device.cuh"
#include "sampler.cuh"
#include "shader.hpp"

#include <optix.h>
#include <cuda/std/tuple>
#include <math_constants.h>

#define INV_PI (1.f / CUDART_PI_F)

using namespace RLpbr::optix;
using namespace cuda::std;

extern "C" {
__constant__ LaunchInput launchInput;
}

struct DeviceVertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct Camera {
    float3 origin;
    float3 view;
    float3 up;
    float3 right;
};

struct Environment {
    OptixTraversableHandle tlas;
    const PackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
    const PackedMaterial *materialBuffer;
    const cudaTextureObject_t *textureHandles;
    const PackedInstance *instances;
    const PackedLight *lights;
    uint32_t numLights;
};

struct Instance {
    uint32_t materialIdx;
};

struct Material {
    uint32_t textureIdx;
    float3 albedoBase;
};

struct Light {
    float3 rgb;
    float3 position;
};

struct Triangle {
    DeviceVertex a;
    DeviceVertex b;
    DeviceVertex c;
};

__forceinline__ Camera unpackCamera(const float4 *packed)
{
    float3 origin = make_float3(packed[0].x, packed[0].y, packed[0].z);
    float3 view = make_float3(packed[0].w, packed[1].x, packed[1].y);
    float3 up = make_float3(packed[1].z, packed[1].w, packed[2].x);
    float3 right = make_float3(packed[2].y, packed[2].z, packed[2].w);

    return Camera {
        origin,
        view,
        up,
        right,
    };
}

// The CUDA compiler incorrectly tries to fit a 64bit immediate offset
// into the 32 bits allowed by the ISA. Use PTX to force the address
// computation into the t1 register rather than a huge immediate
// offset from batch_idx
__forceinline__ pair<Camera, Environment> unpackEnv(uint32_t batch_idx)
{
    PackedEnv packed;

    uint64_t lights_padded;
    asm ("{\n\t"
        ".reg .u64 t1;\n\t"
        "mad.wide.u32 t1, %20, %21, %22;\n\t"
        "ld.global.v4.f32 {%0, %1, %2, %3}, [t1];\n\t"
        "ld.global.v4.f32 {%4, %5, %6, %7}, [t1 + 16];\n\t"
        "ld.global.v4.f32 {%8, %9, %10, %11}, [t1 + 32];\n\t"
        "ld.global.v2.u64 {%12, %13}, [t1 + 48];\n\t"
        "ld.global.v2.u64 {%14, %15}, [t1 + 64];\n\t"
        "ld.global.v2.u64 {%16, %17}, [t1 + 80];\n\t"
        "ld.global.v2.u64 {%18, %19}, [t1 + 96];\n\t"
        "}\n\t"
        : "=f" (packed.camData[0].x), "=f" (packed.camData[0].y),
          "=f" (packed.camData[0].z), "=f" (packed.camData[0].w),
          "=f" (packed.camData[1].x), "=f" (packed.camData[1].y),
          "=f" (packed.camData[1].z), "=f" (packed.camData[1].w),
          "=f" (packed.camData[2].x), "=f" (packed.camData[2].y),
          "=f" (packed.camData[2].z), "=f" (packed.camData[2].w),
          "=l" (packed.tlas), "=l" (packed.vertexBuffer),
          "=l" (packed.indexBuffer), "=l" (packed.materialBuffer),
          "=l" (packed.textureHandles), "=l" (packed.instances),
          "=l" (packed.lights), "=l" (lights_padded)
        : "r" (batch_idx), "n" (sizeof(PackedEnv)), "n" (ENV_PTR)
    );

    return {
        unpackCamera(packed.camData),
        Environment {
            packed.tlas,
            packed.vertexBuffer,
            packed.indexBuffer,
            packed.materialBuffer,
            packed.textureHandles,
            packed.instances,
            packed.lights,
            uint32_t(lights_padded),
        },
    };
}

// Similarly to unpackEnv, work around broken 64bit constant pointer support
__forceinline__ void setOutput(uint32_t base_offset, float3 rgb)
{
    uint16_t r = __half_as_ushort(__float2half(rgb.x));
    uint16_t g = __half_as_ushort(__float2half(rgb.y));
    uint16_t b = __half_as_ushort(__float2half(rgb.z));

    asm ("{\n\t"
        ".reg .u64 t1;\n\t"
        "mad.wide.u32 t1, %3, %4, %5;\n\t"
        "st.global.u16 [t1], %0;\n\t"
        "st.global.u16 [t1 + %4], %1;\n\t"
        "st.global.u16 [t1 + 2 * %4], %2;\n\t"
        "}\n\t"
        :
        : "h" (r), "h" (g), "h" (b),
          "r" (base_offset), "n" (sizeof(half)), "n" (OUTPUT_PTR)
        : "memory"
    );
}

__forceinline__ Instance unpackInstance(const PackedInstance &packed)
{
    return Instance {
        packed.materialIdx,
    };
}

__forceinline__ Material unpackMaterial(const PackedMaterial &packed)
{
    float4 data0 = packed.data0;
    uint4 data1 = packed.data1;

    return Material {
        data1.x,
        make_float3(data0.x, data0.y, data0.z),
    };
}

__forceinline__ Light unpackLight(const PackedLight &packed)
{
    float4 data0 = packed.data[0];
    float4 data1 = packed.data[0];

    return Light {
        make_float3(data0.x, data0.y, data0.z),
        make_float3(data0.w, data1.x, data1.y),
    };
}

__forceinline__ pair<float3, float3> computeCameraRay(
    const Camera &camera, uint3 idx, uint3 dim, Sampler &sampler)
{
    float2 jitter = sampler.get2D();

    float2 jittered_raster = make_float2(idx.x, idx.y) + jitter;

    float2 screen = make_float2((2.f * jittered_raster.x) / dim.x - 1,
                                (2.f * jittered_raster.y) / dim.y - 1);

    float3 direction = camera.right * screen.x + camera.up * screen.y +
        camera.view;

    return {
        camera.origin,
        direction,
    };
}

__forceinline__ float computeDepth()
{
    float3 scaled_dir = optixGetWorldRayDirection() * optixGetRayTmax();
    return length(scaled_dir);
}

__forceinline__ float3 computeBarycentrics(float2 raw)
{
    return make_float3(1.f - raw.x - raw.y, raw.x, raw.y);
}

__forceinline__ uint32_t unormFloat2To32(float2 a)
{
    auto conv = [](float v) { return (uint32_t)trunc(v * 65535.f + 0.5f); };

    return conv(a.x) << 16 | conv(a.y);
}

__forceinline__ float2 unormFloat2From32(uint32_t a)
{
     return make_float2(a >> 16, a & 0xffff) * (1.f / 65535.f);
}

__forceinline__ void setHitPayload(float2 barycentrics,
                                   uint32_t triangle_index,
                                   OptixTraversableHandle inst_hdl,
                                   uint32_t instance_index)
{
    optixSetPayload_0(unormFloat2To32(barycentrics));
    optixSetPayload_1(triangle_index);
    optixSetPayload_2(inst_hdl >> 32);
    optixSetPayload_3(inst_hdl & 0xFFFFFFFF);
    optixSetPayload_4(instance_index);
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

// Returns *unnormalized vector*
inline float3 computeGeometricNormal(const Triangle &tri)
{
    float3 v1 = tri.b.position - tri.a.position;
    float3 v2 = tri.c.position - tri.a.position;

    return cross(v1, v2);
}

__forceinline__ float3 transformNormal(const float4 *w2o, float3 n)
{
    float4 r1 = w2o[0];
    float4 r2 = w2o[1];
    float4 r3 = w2o[2];

    return make_float3(
        r1.x * n.x + r2.x * n.y + r3.x * n.z,
        r1.y * n.x + r2.y * n.y + r3.y * n.z,
        r1.z * n.x + r2.z * n.y + r3.z * n.z);

}

__forceinline__ float3 transformPosition(const float4 *o2w, float3 p)
{
    float4 r1 = o2w[0];
    float4 r2 = o2w[1];
    float4 r3 = o2w[2];

    return make_float3(
        r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w,
        r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w,
        r3.x * p.x + r3.y * p.y + r3.z * p.z + r3.w);
}

inline float3 faceforward(const float3 &n, const float3 &i, const float3 &nref)
{
  return n * copysignf(1.0f, dot(i, nref));
}

// Ray Tracing Gems Chapter 6 (avoid self intersections)
inline float3 offsetRayOrigin(const float3 &o, const float3 &geo_normal)
{
    constexpr float global_origin = 1.f / 32.f;
    constexpr float float_scale = 1.f / 65536.f;
    constexpr float int_scale = 256.f;

    int3 int_offset = make_int3(geo_normal.x * int_scale,
        geo_normal.y * int_scale, geo_normal.z * int_scale);

    float3 o_integer = make_float3(
        __int_as_float(
            __float_as_int(o.x) + ((o.x < 0) ? -int_offset.x : int_offset.x)),
        __int_as_float(
            __float_as_int(o.y) + ((o.y < 0) ? -int_offset.y : int_offset.y)),
        __int_as_float(
            __float_as_int(o.z) + ((o.z < 0) ? -int_offset.z : int_offset.z)));

    return make_float3(
        fabsf(o.x) < global_origin ?
            o.x + float_scale * geo_normal.x : o_integer.x,
        fabsf(o.y) < global_origin ?
            o.y + float_scale * geo_normal.y : o_integer.y,
        fabsf(o.z) < global_origin ?
            o.z + float_scale * geo_normal.z : o_integer.z);
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    uint32_t batch_idx = launchInput.baseBatchOffset + idx.z;

    uint32_t base_out_offset = 
        3 * (batch_idx * dim.y * dim.x + idx.y * dim.x + idx.x);

    const auto [cam, env] = unpackEnv(batch_idx);
    
    float3 pixel_radiance = make_float3(0.f);

    const float intensity = 10.f;

#if SPP != (1u)
#pragma unroll 1
#endif
    for (int32_t sample_idx = 0; sample_idx < SPP; sample_idx++) {
        Sampler sampler(idx.x, idx.y, sample_idx,
                        launchInput.baseFrameCounter + idx.z);

        float3 sample_radiance = make_float3(0.f);
        float3 path_prob = make_float3(1.f);

        auto [ray_origin, ray_dir] =
            computeCameraRay(cam, idx, dim, sampler);

#if MAX_DEPTH != (1u)
#pragma unroll 1
#endif
        for (int32_t path_depth = 0; path_depth < MAX_DEPTH;
             path_depth++) {
            // Trace shade ray
            unsigned int payload_0;
            unsigned int payload_1;
            unsigned int payload_2;

            // Need to overwrite the register so miss detection works
            // payload_3 is guaranteed to not be 0 on a hit
            unsigned int payload_3 = 0;
            unsigned int payload_4;

            // FIXME Min T for both shadow and this ray
            optixTrace(
                    env.tlas,
                    ray_origin,
                    ray_dir,
                    0.f, // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(0xff), // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload_0,
                    payload_1,
                    payload_2,
                    payload_3,
                    payload_4);

            // Miss, hit env map
            if (payload_3 == 0) {
                //sample_radiance += intensity * path_prob;
                break;
            }

            float2 raw_barys = unormFloat2From32(payload_0);
            uint32_t index_offset = payload_1;
            OptixTraversableHandle inst_hdl = (OptixTraversableHandle)(
                (uint64_t)payload_2 << 32 | (uint64_t)payload_3);

            uint32_t instance_idx = payload_4;
            Instance inst = unpackInstance(env.instances[instance_idx]);

            const float4 *o2w =
                optixGetInstanceTransformFromHandle(inst_hdl);
            const float4 *w2o =
                optixGetInstanceInverseTransformFromHandle(inst_hdl);

            float3 barys = computeBarycentrics(raw_barys);

            Triangle hit_tri = fetchTriangle(env.vertexBuffer,
                                             env.indexBuffer + index_offset);
            DeviceVertex interpolated = interpolateTriangle(hit_tri, barys);
            float3 obj_geo_normal = computeGeometricNormal(hit_tri);

            float3 world_position =
                transformPosition(o2w, interpolated.position);
            float3 world_normal =
                transformNormal(w2o, interpolated.normal);
            float3 world_geo_normal =
                transformNormal(w2o, obj_geo_normal);

            world_normal = faceforward(world_normal, -ray_dir, world_normal);

            world_normal = normalize(world_normal);
            world_geo_normal = normalize(world_geo_normal);

            float3 up = make_float3(0, 0, 1);
            float3 up_alt = make_float3(0, 1, 0);

            float3 binormal = cross(world_normal, up);
            if (length(binormal) < 1e-3f) {
                binormal = cross(world_normal, up_alt);
            }
            binormal = normalize(binormal);

            float3 tangent = normalize(cross(binormal, world_normal));

            auto randomDirection = [&sampler] (const float3 &tangent,
                                           const float3 &binormal,
                                           const float3 &normal) {
                float2 uv = sampler.get2D();
                const float r = sqrtf(uv.x);
                const float phi = 2.0f * (CUDART_PI_F) * uv.y;
                float2 disk = r * make_float2(cosf(phi), sinf(phi));
                float3 hemisphere = make_float3(disk.x, disk.y,
                    sqrtf(fmaxf(0.0f, 1.0f - dot(disk, disk))));

                return normalize(hemisphere.x * tangent +
                    hemisphere.y * binormal +
                    hemisphere.z * normal);
            };

            float3 shadow_origin =
                offsetRayOrigin(world_position, world_geo_normal);

            uint32_t light_idx = sampler.get1D() * env.numLights;

            Light light = unpackLight(env.lights[light_idx]);

            float3 to_light = light.position - shadow_origin;
            float3 shadow_direction = normalize(to_light);

            payload_0 = 1;
            optixTrace(
                    env.tlas,
                    shadow_origin,
                    shadow_direction,
                    0.f,                // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(0xff), // Specify always visible
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload_0,
                    payload_1,
                    payload_2,
                    payload_3,
                    payload_4);

            Material mat =
                unpackMaterial(env.materialBuffer[inst.materialIdx]);

            float3 albedo;
            if (mat.textureIdx != ~0u) {
                cudaTextureObject_t tex = env.textureHandles[mat.textureIdx];
                float4 tex_value = tex2DLod<float4>(tex, interpolated.uv.x,
                                                   interpolated.uv.y, 0);
                albedo.x = tex_value.x;
                albedo.y = tex_value.y;
                albedo.z = tex_value.z;
            } else {
                albedo = mat.albedoBase;
            }

            if (payload_0 == 0) {
                // Shade
                float light_r2 = dot(to_light, to_light);
                float3 irradiance = light.rgb / light_r2;
                float inv_light_pdf = env.numLights;

                float dir_prob = fabsf(dot(world_normal, shadow_direction));
                float3 brdf = albedo * INV_PI * dir_prob;

                sample_radiance +=
                    path_prob * brdf * irradiance * inv_light_pdf;
            }

            // Start setup for next bounce
            ray_origin = shadow_origin;
            ray_dir = randomDirection(tangent, binormal, world_normal);

            // nDwi & inverse PI cancel out
            path_prob *= albedo;
        }

        pixel_radiance += sample_radiance / SPP;
    }

    setOutput(base_out_offset, pixel_radiance);
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__ch()
{
    uint32_t base_index = optixGetInstanceId() + 3 * optixGetPrimitiveIndex();
    setHitPayload(optixGetTriangleBarycentrics(), base_index,
                  optixGetTransformListHandle(0),
                  optixGetInstanceIndex());
}
