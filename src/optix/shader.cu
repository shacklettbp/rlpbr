#include "device.cuh"
#include "sampler.cuh"
#include "shader.hpp"

#include <optix.h>
#include <math_constants.h>

#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2
#include <cuda/std/tuple>
#else

template <typename X, typename Y>
struct pair {
    X first;
    Y second;
};

#endif

#define INV_PI (1.f / CUDART_PI_F)
#define M_PI CUDART_PI_F
#define M_1_PI INV_PI

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

#ifdef METALLIC_ROUGHNESS
struct Material {
    float3 baseColor;
    float metallic;
    float roughness;
    uint32_t baseColorIdx;
    uint32_t metallicRoughnessIdx;
};
#endif

struct MaterialParams {
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float alpha;
    float metallic;
    float roughness;
};

struct BSDFParams {
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float alpha;
    float roughness;
    float diffuseProb;
    float specularProb;
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

__forceinline__ float rgbExtractY(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.722f * rgb.z;
}

template <typename T>
__forceinline__ T computeFresnel(T f0, T f90, float cos_theta)
{
    return f0 + (f90 - f0) * pow(max(1.f - cos_theta, 0.f), 5.f);
}

#ifdef METALLIC_ROUGHNESS
__forceinline__ Material unpackMaterial(const PackedMaterial &packed)
{
    float4 data0 = packed.data[0];
    float4 data1 = packed.data[1];

    return Material {
        make_float3(data0.x, data0.y, data0.z),
        data0.w,
        data1.x,
        __float_as_uint(data1.y),
        __float_as_uint(data1.z),
    };
}

__forceinline__ MaterialParams computeMaterialParams(const Material &mat,
    const cudaTextureObject_t *textures, float2 uv)
{
    float3 base_color = mat.baseColor;
    if (mat.baseColorIdx != ~0u) {
        cudaTextureObject_t tex = textures[mat.baseColorIdx];
        float4 tex_value = tex2DLod<float4>(tex, uv.x, uv.y, 0);

        base_color.x *= tex_value.x;
        base_color.y *= tex_value.y;
        base_color.z *= tex_value.z;
    }

    float metallic = mat.metallic;
    float roughness = mat.roughness;

    if (mat.metallicRoughnessIdx != ~0u) {
        cudaTextureObject_t tex = textures[mat.metallicRoughnessIdx];
        float4 tex_value = tex2DLod<float4>(tex, uv.x, uv.y, 0);

        metallic *= tex_value.z;
        roughness *= tex_value.y;
    }

    float3 diffuse = lerp(base_color, make_float3(0), metallic);

    float F0 = 0.04; // Assume constant IoR 1.5

    float3 specular = lerp(make_float3(F0), base_color, metallic);

    float alpha = roughness * roughness;

    static constexpr float min_alpha = 0.0064f;
    if (alpha < min_alpha) {
        alpha = 0.f;
    }

    return MaterialParams {
        diffuse,
        specular,
        alpha,
        metallic,
        roughness,
    };
}
#endif

__forceinline__ BSDFParams computeBSDFParams(const MaterialParams &mat_params,
                                             float3 wo)
{
    float dielectric_weight = 1.f - mat_params.metallic;

    float3 specular_fresnel = computeFresnel(mat_params.specularAlbedo,
                                             make_float3(1.f), wo.z);

    float diffuse_luma = rgbExtractY(mat_params.diffuseAlbedo);
    float specular_luma = rgbExtractY(specular_fresnel);

    float diffuse_prob = diffuse_luma * dielectric_weight;
    float specular_prob = specular_luma;

    float total_prob = diffuse_prob + specular_prob;
    if (total_prob > 0.f) {
        diffuse_prob /= total_prob;
        specular_prob /= total_prob;
    }

    BSDFParams out;
    out.diffuseAlbedo = mat_params.diffuseAlbedo;
    out.specularAlbedo = mat_params.specularAlbedo;
    out.alpha = mat_params.alpha;
    out.roughness = mat_params.roughness;
    out.diffuseProb = diffuse_prob;
    out.specularProb = specular_prob;

    return out;
}

struct ShadeResult {
    float3 color;
    float3 bounceDir;
    float3 bounceProb;
};

__forceinline__ float3 diffuseWeight(const BSDFParams &bsdf_params,
                                     const float3 &wo, const float3 &wi,
                                     const float3 &half_vec)
{
    float cos_in_half = dot(wi, half_vec);
    float bias = lerp(0.f, 0.5f, bsdf_params.roughness);
    float energy_coeff = lerp(1.f, 1.f / 1.51f, bsdf_params.roughness);
    float fd90 = bias + 2.f * cos_in_half * cos_in_half * bsdf_params.roughness;
    constexpr float fd0 = 1.f;
    float out_scatter = computeFresnel(fd0, fd90, wo.z);
    float in_scatter = computeFresnel(fd0, fd90, wi.z);

    return bsdf_params.diffuseAlbedo * out_scatter * in_scatter * energy_coeff;
}

__forceinline__ float3 diffuseBSDF(const BSDFParams &bsdf_params,
                                   const float3 &wo, const float3 &wi,
                                   const float3 &half_vec)
{
    return diffuseWeight(bsdf_params, wo, wi, half_vec) * M_1_PI * wi.z;
}

__forceinline__ float ggxLambda(float cos_theta, float a2)
{
    if (cos_theta <= 0.f) return 0.f;

    float cos2 = cos_theta * cos_theta;
    float tan2 = max(1.f - cos2, 0.f) / cos2;
    return 0.5f * (-1.f + sqrt(1.f + a2 * tan2));
}

__forceinline__ float ggxNDF(float alpha, float cos_theta)
{
    float a2 = alpha * alpha;
    float d = ((cos_theta * a2 - cos_theta) * cos_theta + 1.f);
    return a2 / (d * d * M_PI);
}

__forceinline__ float ggxMasking(float a2, float out_cos, float in_cos)
{
    float in_lambda = ggxLambda(in_cos, a2);
    float out_lambda = ggxLambda(out_cos, a2);
    return 1.f / (1.f + in_lambda + out_lambda);
}

__forceinline__ float3 specularBSDF(const BSDFParams &bsdf_params,
                                    const float3 &wo, const float3 &wi,
                                    const float3 &half_vec)
{
    float a2 = bsdf_params.alpha * bsdf_params.alpha;
    float D = ggxNDF(bsdf_params.alpha, half_vec.z);
    float G = ggxMasking(a2, wo.z, wi.z);

    float cos_out_half = dot(wo, half_vec);

    float3 F = computeFresnel(bsdf_params.specularAlbedo, make_float3(1.f),
                              cos_out_half);

    float3 specular = 0.25f * F * D * G / wo.z;

    return (bsdf_params.diffuseProb == 0.f || bsdf_params.alpha == 0 ||
            min(wo.z, wi.z) < 1e-6f) ? make_float3(0.f) : specular;
}

__forceinline__ float3 evalBSDF(const BSDFParams &bsdf_params,
                                const float3 &wo, const float3 &wi)
{
    float3 half_vec = normalize(wo + wi);
    return diffuseBSDF(bsdf_params, wo, wi, half_vec) +
        specularBSDF(bsdf_params, wo, wi, half_vec);
}

__forceinline__ float3 concentricHemisphere(float2 uv)
{
    float2 c = 2.f * uv - 1.f;
    float2 d;
    if (c.x == 0.f && c.y == 0.f) {
        d = make_float2(0.f);
    } else {
        float phi, r;
        if (abs(c.x) > abs(c.y))
        {
            r = c.x;
            phi = (c.y / c.x) * (M_PI / 4.f);
        } else {
            r = c.y;
            phi = (M_PI / 2.f) - (c.x / c.y) * (M_PI / 4.f);
        }

        d = r * make_float2(cosf(phi), sinf(phi));
    }

    float z = sqrt(max(0.f, 1.f - dot(d, d)));

    return make_float3(d.x, d.y, z);
}

__forceinline__ float3 cosineHemisphere(float2 uv)
{
    const float r = sqrtf(uv.x);
    const float phi = 2.0f * M_PI * uv.y;
    float2 disk = r * make_float2(cosf(phi), sinf(phi));
    float3 hemisphere = make_float3(disk.x, disk.y,
        sqrtf(fmaxf(0.0f, 1.0f - dot(disk, disk))));

    return hemisphere;
}

__forceinline__ pair<float3, float3> sampleDiffuse(const BSDFParams &bsdf_params,
                                                   const float3 &wo,
                                                   const float2 &sample_uv)
{
    float3 wi = concentricHemisphere(sample_uv);
    float3 h = normalize(wo + wi);
    float3 weight = diffuseWeight(bsdf_params, wo, wi, h);

    if (min(wo.z, wi.z) < 1e-6f) {
        weight = make_float3(0.f);
    } 

    return {
        wi,
        weight,
    };
}

__forceinline__ float3 sampleGGX(float alpha, const float3 &wo,
                                              const float2 &sample_uv)
{
    float3 Vh = normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));

    // Construct orthonormal basis (Vh,T1,T2).
    float3 T1 = (Vh.z < 0.9999f) ?
        normalize(cross(make_float3(0.f, 0.f, 1.f), Vh)) :
        make_float3(1.f, 0.f, 0.f);

    float3 T2 = cross(Vh, T1);

    float r = sqrtf(sample_uv.x);
    float phi = (2.f * M_PI) * sample_uv.y;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    float3 h = normalize(make_float3(alpha * Nh.x, alpha * Nh.y, max(0.f, Nh.z)));

    return h;
}

__forceinline__ pair<float3, float3> sampleSpecular(const BSDFParams &bsdf_params,
                                                    const float3 &wo,
                                                    const float2 &sample_uv)
{
    float3 wi;
    float3 weight;

    if (wo.z < 1e-6f) {
        wi = make_float3(0.f);
        weight = make_float3(0.f);
    } else if (bsdf_params.alpha == 0.f) {
        wi = make_float3(-wo.x, -wo.y, wo.z);
        weight = computeFresnel(bsdf_params.specularAlbedo, make_float3(1.f), wo.z);
    } else {
        float3 h = sampleGGX(bsdf_params.alpha, wo, sample_uv);
        float cos_half_out = dot(wo, h);
        wi = 2.f * cos_half_out * h - wo;

        if (wi.z < 1e-6f) {
            weight = make_float3(0.f);
        } else {
            float a2 = bsdf_params.alpha * bsdf_params.alpha;

            float G = ggxMasking(a2, wo.z, wi.z);
            float GG1_out = G * (1.f + ggxLambda(wo.z, a2));
            float3 F = computeFresnel(bsdf_params.specularAlbedo, make_float3(1.f),
                                      cos_half_out);

            weight = F * GG1_out;
        }
    }

    return {
        wi,
        weight,
    };
}

__forceinline__ pair<float3, float3> sampleBSDF(Sampler &sampler,
                                                const BSDFParams &bsdf_params,
                                                const float3 &wo)
{
    float3 wi;
    float3 weight;
    float selector = sampler.get1D();
    float2 sample2D = sampler.get2D();

    if (selector < bsdf_params.diffuseProb) {
        tie(wi, weight) = sampleDiffuse(bsdf_params, wo, sample2D);
        weight /= bsdf_params.diffuseProb;
    } else if (selector < bsdf_params.diffuseProb + bsdf_params.specularProb) {
        tie(wi, weight) = sampleSpecular(bsdf_params, wo, sample2D);
        weight /= bsdf_params.specularProb;
    } else {
        wi = make_float3(0.f);
        weight = make_float3(0.f);
    }

    return {
        wi,
        weight,
    };
}

__forceinline__ float3 worldToLocal(float3 vec, float3 normal, float3 tangent,
                                    float3 bitangent)
{
    return make_float3(dot(vec, tangent), dot(vec, bitangent),
        dot(vec, normal));
}

__forceinline__ float3 localToWorld(float3 vec, float3 normal, float3 tangent,
                                  float3 bitangent)
{
    return vec.x * tangent + vec.y * bitangent + vec.z * normal;
}

__forceinline__ ShadeResult shade(Sampler &sampler, const Material &mat,
                                  const cudaTextureObject_t *textures, const Light &light, 
                                  float3 outgoing, float3 to_light,
                                  float dist_to_light2, float inv_light_pdf,
                                  float2 uv, float3 normal,
                                  float3 tangent, float3 bitangent)
{
    float3 wo = worldToLocal(outgoing, normal, tangent, bitangent);
    float3 wi = worldToLocal(to_light, normal, tangent, bitangent);

    // FIXME, make divergence less bad here
    if (wi.z < 0.f) {
        return {
            make_float3(0.f),
            make_float3(0.f),
            make_float3(0.f),
        };
    }

    MaterialParams mat_params = computeMaterialParams(mat, textures, uv);
    BSDFParams bsdf_params = computeBSDFParams(mat_params, wo);

    float3 bsdf = evalBSDF(bsdf_params, wo, wi);

    // Lighting FIXME move light calculations to before this function
    float3 irradiance = light.rgb / dist_to_light2;
    float3 nee = bsdf * irradiance * inv_light_pdf;

    // Compute bounce
    auto [sample_dir, sample_prob] = sampleBSDF(sampler, bsdf_params, wo);

    return {
        nee,
        localToWorld(sample_dir, normal, tangent, bitangent),
        sample_prob,
    };
}

__forceinline__ Light unpackLight(const PackedLight &packed)
{
    float4 data0 = packed.data[0];
    float4 data1 = packed.data[1];

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

            if (path_prob.x == 0.f && path_prob.y == 0.f && path_prob.z == 0.f) {
                break;
            }

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

            float3 shadow_origin =
                offsetRayOrigin(world_position, world_geo_normal);

            uint32_t light_idx = sampler.get1D() * env.numLights;

            Light light = unpackLight(env.lights[light_idx]);

            float3 to_light = light.position - shadow_origin;
            float3 to_light_norm = normalize(to_light);
            float light_dist2 = dot(to_light, to_light);
            float light_dist = sqrtf(light_dist2);

            payload_0 = 1;
            optixTrace(
                    env.tlas,
                    shadow_origin,
                    to_light_norm,
                    0.f,                // Min intersection distance
                    light_dist,               // Max intersection distance
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

            auto [color, bounce_dir, bounce_prob] =
                shade(sampler, mat, env.textureHandles, light, -ray_dir, to_light_norm,
                      light_dist2, env.numLights,
                      interpolated.uv, world_normal, tangent, binormal);


            if (payload_0 == 0) {
                sample_radiance += path_prob * color;
            }

            // Start setup for next bounce
            ray_origin = shadow_origin;
            ray_dir = bounce_dir;

            path_prob *= bounce_prob;
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
