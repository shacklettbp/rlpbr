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
#define M_PI_4 CUDART_PIO4_F
#define INFINITY CUDART_INF_F

using namespace RLpbr;
using namespace RLpbr::optix;
using namespace cuda::std;

extern "C" {
__constant__ LaunchInput launchInput;
}

struct DeviceVertex {
    float3 position;
    float3 normal;
    float4 tangentAndSign;
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
    const DevicePackedVertex *vertexBuffer;
    const uint32_t *indexBuffer;
    const PackedMaterial *materialBuffer;
    const cudaTextureObject_t *textureHandles;
    const TextureSize *textureDims;
    const PackedMeshInfo *meshInfos;
    const PackedInstance *instances;
    const uint32_t *instanceMaterials;
    const PackedLight *lights;
    const PackedTransforms *transforms;
    uint32_t numLights;
};

struct MeshInfo {
    uint32_t indexOffset;
};

struct Instance {
    uint32_t materialOffset;
    uint32_t meshOffset;
};

struct TransformMat {
    float4 rows[3];
};

struct MaterialParams {
    float3 baseColor;
    float baseTransmission;
    float3 baseSpecular;
    float specularScale;
    float ior;
    float baseMetallic;
    float baseRoughness;
    uint32_t flags;

#ifdef ADVANCED_MATERIAL
    float clearcoat;
    float clearcoatRoughness;
    float3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    float anisoRotation;
    float3 baseEmittance;
#endif
};

struct Material {
    float3 rho;
    float transmission;
    float3 rhoSpecular;
    float specularScale;
    float ior;
    float metallic;
    float roughness;
    float transparencyMask;

#ifdef ADVANCED_MATERIAL
    float clearcoatScale;
    float clearcoatRoughness;
    float3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    float anisoRotation;
    float3 emittance;
#endif
};

struct BSDFParams {
    float3 rhoDiffuse;
    float transparencyMask;
    float3 rhoTransmissive;
    float transmission;
    float3 sharedF0;
    float sharedF90;
    float3 transmissiveF0;
    float transmissiveF90;
    float alpha;
    float roughness;
    float diffuseLambertScale;
    float diffuseLookupF0;
    float diffuseAverageAlbedo;
    float3 microfacetMSWeight;
    float microfacetMSAvgAlbedoComplement;

    // Sampling probabilities for each sub component of BSDF
    float diffuseProb;
    float microfacetProb;
    float microfacetMSProb;
    float transmissionProb;

#ifdef ADVANCED_MATERIAL
    float clearcoatScale;
    float clearcoatAlpha;
    float clearcoatProb;
#endif
};

enum class LightType : uint32_t {
    Point,
    Portal,
    Environment,
};

struct PointLight {
    float3 rgb;
    float3 position;
};

struct PortalLight {
    float3 corners[4];
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
        "mad.wide.u32 t1, %24, %25, %26;\n\t"
        "ld.global.v4.f32 {%0, %1, %2, %3}, [t1];\n\t"
        "ld.global.v4.f32 {%4, %5, %6, %7}, [t1 + 16];\n\t"
        "ld.global.v4.f32 {%8, %9, %10, %11}, [t1 + 32];\n\t"
        "ld.global.v2.u64 {%12, %13}, [t1 + 48];\n\t"
        "ld.global.v2.u64 {%14, %15}, [t1 + 64];\n\t"
        "ld.global.v2.u64 {%16, %17}, [t1 + 80];\n\t"
        "ld.global.v2.u64 {%18, %19}, [t1 + 96];\n\t"
        "ld.global.v2.u64 {%20, %21}, [t1 + 112];\n\t"
        "ld.global.v2.u64 {%22, %23}, [t1 + 128];\n\t"
        "}\n\t"
        : "=f" (packed.camData[0].x), "=f" (packed.camData[0].y),
          "=f" (packed.camData[0].z), "=f" (packed.camData[0].w),
          "=f" (packed.camData[1].x), "=f" (packed.camData[1].y),
          "=f" (packed.camData[1].z), "=f" (packed.camData[1].w),
          "=f" (packed.camData[2].x), "=f" (packed.camData[2].y),
          "=f" (packed.camData[2].z), "=f" (packed.camData[2].w),
          "=l" (packed.tlas), "=l" (packed.vertexBuffer),
          "=l" (packed.indexBuffer), "=l" (packed.materialBuffer),
          "=l" (packed.textureHandles), "=l" (packed.textureDims),
          "=l" (packed.meshInfos), "=l" (packed.instances),
          "=l" (packed.instanceMaterials), "=l" (packed.lights),
          "=l" (packed.transforms), "=l" (lights_padded)
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
            packed.textureDims,
            packed.meshInfos,
            packed.instances,
            packed.instanceMaterials,
            packed.lights,
            packed.transforms,
            uint32_t(lights_padded),
        },
    };
}

// Similarly to unpackEnv, work around broken 64bit constant pointer support
__forceinline__ void setOutput(uint32_t base_offset, float3 rgb,
                               uint16_t instance_id)
{
    // FP16 cannot represent numbers over this, and get converted
    // to infinity, clamp instead
    rgb = fminf(rgb, make_float3(65504.f));

    uint16_t r = __half_as_ushort(__float2half(rgb.x));
    uint16_t g = __half_as_ushort(__float2half(rgb.y));
    uint16_t b = __half_as_ushort(__float2half(rgb.z));

    asm ("{\n\t"
        ".reg .u64 t1;\n\t"
        "mad.wide.u32 t1, %4, %5, %6;\n\t"
        "st.global.v4.u16 [t1], {%0, %1, %2, %3};\n\t"
        "}\n\t"
        :
        : "h" (r), "h" (g), "h" (b), "h" (instance_id)
          "r" (base_offset), "n" (sizeof(half)), "n" (OUTPUT_PTR)
        : "memory"
    );
}

#ifdef AUXILIARY_OUTPUTS
__forceinline__ void setAuxiliaries(uint32_t base_offset, float3 normal,
                                    float3 albedo)
{
    uint16_t nx = __half_as_ushort(__float2half(normal.x));
    uint16_t ny = __half_as_ushort(__float2half(normal.y));
    uint16_t nz = __half_as_ushort(__float2half(normal.z));

    uint16_t ax = __half_as_ushort(__float2half(albedo.x));
    uint16_t ay = __half_as_ushort(__float2half(albedo.y));
    uint16_t az = __half_as_ushort(__float2half(albedo.z));

    asm ("{\n\t"
         ".reg .u64 t1;\n\t"
         ".reg .u64 t2;\n\t"
         "mad.wide.u32 t1, %6, %7, %8;\n\t"
         "mad.wide.u32 t2, %6, %7, %9;\n\t"
         "st.global.u16 [t1], %0;\n\t"
         "st.global.u16 [t1 + %7], %1;\n\t"
         "st.global.u16 [t1 + 2 * %7], %2;\n\t"
         "st.global.u16 [t2], %3;\n\t"
         "st.global.u16 [t2 + %7], %4;\n\t"
         "st.global.u16 [t2 + 2 * %7], %5;\n\t"
         "}\n\t"
         :
         : "h" (nx), "h" (ny), "h" (nz),
           "h" (ax), "h" (ay), "h" (az),
           "r" (base_offset), "n" (sizeof(half)),
           "n" (NORMAL_PTR), "n" (ALBEDO_PTR)
         : "memory"
    );
}
#endif

__forceinline__ MeshInfo unpackMeshInfo(const PackedMeshInfo &packed)
{
    return MeshInfo {
        packed.data.x,
    };
}

__forceinline__ Instance unpackInstance(const PackedInstance &packed)
{
    return Instance {
        packed.materialOffset,
        packed.meshOffset,
    };
}

__forceinline__ pair<TransformMat, TransformMat> unpackTransforms(
    const PackedTransforms &txfms)
{
    float4 o2w0 = txfms.data[0];
    float4 o2w1 = txfms.data[1];
    float4 o2w2 = txfms.data[2];
    float4 w2o0 = txfms.data[3];
    float4 w2o1 = txfms.data[4];
    float4 w2o2 = txfms.data[5];

    return {
        TransformMat {
            {
                make_float4(o2w0.x, o2w0.w, o2w1.z, o2w2.y),
                make_float4(o2w0.y, o2w1.x, o2w1.w, o2w2.z),
                make_float4(o2w0.z, o2w1.y, o2w2.x, o2w2.w),
            },
        },
        TransformMat {
            {
                make_float4(w2o0.x, w2o0.w, w2o1.z, w2o2.y),
                make_float4(w2o0.y, w2o1.x, w2o1.w, w2o2.z),
                make_float4(w2o0.z, w2o1.y, w2o2.x, w2o2.w),
            },
        },
    };
}

__forceinline__ float unpackUnorm8(uint8_t x)
{
    return float(x) / 255.f;
}

__forceinline__ float unpackNonlinearUnorm8(uint8_t x)
{
    float f = unpackUnorm8(x); 
    return f * f;
}

__forceinline__ bool checkMaterialFlag(uint32_t flags, MaterialFlags check)
{
    return flags & static_cast<uint32_t>(check);
}

__forceinline__ MaterialParams unpackMaterialParams(const PackedMaterial &packed)
{
    uint4 data0 = packed.data[0];

    MaterialParams params {};
    params.baseColor = make_float3(
        unpackNonlinearUnorm8(uint8_t(data0.x)),
        unpackNonlinearUnorm8(uint8_t(data0.x >> 8)),
        unpackNonlinearUnorm8(uint8_t(data0.x >> 16)));
    params.baseTransmission = unpackUnorm8(uint8_t(data0.x >> 24));
    params.specularScale = unpackUnorm8(uint8_t(data0.y));
    params.ior = float(uint8_t(data0.y >> 8)) / 170.f + 1.f;
    params.baseMetallic = unpackUnorm8(uint8_t(data0.y >> 16));
    params.baseRoughness  = unpackUnorm8(uint8_t(data0.y >> 24));
    params.baseSpecular = make_float3(
        __half2float(__ushort_as_half(uint16_t(data0.z))),
        __half2float(__ushort_as_half(uint16_t(data0.z >> 16))),
        __half2float(__ushort_as_half(uint16_t(data0.w))));
    params.flags = uint16_t(data0.w >> 16);

#ifdef ADVANCED_MATERIAL
    if (checkMaterialFlag(params.flags, MaterialFlags::Complex)) {
        uint4 data1 = packed.data[1];

        params.clearcoat = unpackUnorm8(uint8_t(data1.x));
        params.clearcoatRoughness = unpackUnorm8(uint8_t(data1.x >> 8));
        params.attenuationColor = make_float3(
            unpackNonlinearUnorm8(uint8_t(data1.x >> 16)),
            unpackNonlinearUnorm8(uint8_t(data1.x >> 24)),
            unpackNonlinearUnorm8(uint8_t(data1.y)));
        params.anisoScale = unpackUnorm8(uint8_t(data1.y >> 8));
        params.anisoRotation = unpackUnorm8(uint8_t(data1.y >> 16));
        // data1.y >> 24 currently unused
        params.attenuationDistance =
            __half2float(__ushort_as_half(uint16_t(data1.z)));
        params.baseEmittance = make_float3(
            __half2float(__ushort_as_half(uint16_t(data0.z >> 16))),
            __half2float(__ushort_as_half(uint16_t(data0.w))),
            __half2float(__ushort_as_half(uint16_t(data0.w >> 16))));
    } else {
        params.clearcoat = 0.f;
        params.clearcoatRoughness = 0.f;
        params.attenuationColor = make_float3(1.f, 1.f, 1.f);
        params.anisoScale = 0.f;
        params.anisoRotation = 0.f;
        params.attenuationDistance = INFINITY;
        params.baseEmittance = make_float3(0.f, 0.f, 0.f);
    }
#endif

    return params;
}

__forceinline__ float rgbToLuminance(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

template <typename T>
__forceinline__ T computeFresnel(T f0, T f90, float cos_theta)
{
    float complement = max(1.f - cos_theta, 0.f);
    return f0 + (f90 - f0) * complement * complement * complement *
        complement * complement;

}

template <typename T> __forceinline__ T makeZeroVec();

template <>
float makeZeroVec()
{
    return 0.f;
}

template <>
float2 makeZeroVec()
{
    return make_float2(0.f);
}

template <>
float3 makeZeroVec()
{
    return make_float3(0.f);
}

template <>
float4 makeZeroVec()
{
    return make_float4(0.f);
}

template <typename T> __forceinline__ T trimFloat4(float4 v);

template <>
float trimFloat4(float4 v)
{
    return v.x;
}

template <>
float2 trimFloat4(float4 v)
{
    return make_float2(v.x, v.y);
}

template <>
float4 trimFloat4(float4 v)
{
    return v;
}

template <typename T>
__forceinline__ T textureFetch1D(cudaTextureObject_t tex, float ux, float mip)
{
    float4 out;
    asm ("{\n\t"
        "tex.level.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, %5], %6;\n\t"
        "}\n\t"
        : "=f" (out.x), "=f" (out.y), "=f" (out.z), "=f" (out.w)
        : "l" (tex), "f" (ux), "f" (mip)
    );

    return trimFloat4<T>(out);
}

template <typename T>
__forceinline__ T textureFetch2D(cudaTextureObject_t tex, float ux, float uy, float mip)
{
    float4 out;
    asm ("{\n\t"
        "tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;\n\t"
        "}\n\t"
        : "=f" (out.x), "=f" (out.y), "=f" (out.z), "=f" (out.w)
        : "l" (tex), "f" (ux), "f" (uy), "f" (mip)
    );

    return trimFloat4<T>(out);
}

template <typename T>
__forceinline__ T textureFetch3D(cudaTextureObject_t tex, float ux, float uy, float uz, float mip)
{
    float4 out;
    asm ("{\n\t"
        "tex.level.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}], %8;\n\t"
        "}\n\t"
        : "=f" (out.x), "=f" (out.y), "=f" (out.z), "=f" (out.w)
        : "l" (tex), "f" (ux), "f" (uy), "f" (uz), "f" (mip)
    );

    return trimFloat4<T>(out);
}

__forceinline__ Material processMaterial(const MaterialParams &params,
                                         const cudaTextureObject_t *textures,
                                         float2 uv, float mip_level)
{
    Material mat;

    mat.rho = params.baseColor;
    mat.transparencyMask = 1.f;
    if (checkMaterialFlag(params.flags, MaterialFlags::HasBaseTexture)) {
        float4 tex_value = textureFetch2D<float4>(
            textures[TextureConstants::baseOffset], uv.x, uv.y, mip_level);

        mat.rho.x *= tex_value.x;
        mat.rho.y *= tex_value.y;
        mat.rho.z *= tex_value.z;
        mat.transparencyMask *= tex_value.w;
    }

    mat.metallic = params.baseMetallic;
    mat.roughness = params.baseRoughness;
    if (checkMaterialFlag(params.flags, MaterialFlags::HasMRTexture)) {
        float2 tex_value = textureFetch2D<float2>(
            textures[TextureConstants::mrOffset], uv.x, uv.y, mip_level);

        mat.roughness *= tex_value.x;
        mat.metallic *= tex_value.y;
    }
    
    mat.rhoSpecular = params.baseSpecular;
    mat.specularScale = params.specularScale;
    if (checkMaterialFlag(params.flags, MaterialFlags::HasSpecularTexture)) {
        float4 tex_value = textureFetch2D<float4>(
            textures[TextureConstants::specularOffset], uv.x, uv.y, mip_level);

        mat.rhoSpecular.x *= tex_value.x;
        mat.rhoSpecular.y *= tex_value.y;
        mat.rhoSpecular.z *= tex_value.z;
        mat.specularScale *= tex_value.w;
    }

    mat.transmission = params.baseTransmission;
    if (checkMaterialFlag(params.flags,
                          MaterialFlags::HasTransmissionTexture)) {
        cudaTextureObject_t tex =
            textures[TextureConstants::transmissionOffset];
        float tex_value = textureFetch2D<float>(tex, uv.x, uv.y, mip_level);

        mat.transmission *= tex_value;
    }
    mat.ior = params.ior;

#ifdef ADVANCED_MATERIAL
    mat.emittance = params.baseEmittance;
    if (checkMaterialFlag(params.flags, MaterialFlags::HasEmittanceTexture)) {
        cudaTextureObject_t tex = textures[TextureConstants::emittanceOffset];
        float4 tex_value = textureFetch2D<float4>(tex, uv.x, uv.y, mip_level);

        mat.emittance.x *= tex_value.x;
        mat.emittance.y *= tex_value.y;
        mat.emittance.z *= tex_value.z;
    }

    mat.clearcoatScale = params.clearcoat;
    mat.clearcoatRoughness = params.clearcoatRoughness;
    if (checkMaterialFlag(params.flags,
                          MaterialFlags::HasClearcoatTexture)) {
        float2 tex_value = textureFetch2D<float2>(
            textures[TextureConstants::clearcoatOffset], uv.x, uv.y,
            mip_level);

        mat.clearcoatScale *= tex_value.x;
        mat.clearcoatRoughness *= tex_value.y;
    }

    mat.anisoScale = params.anisoScale;
    mat.anisoRotation = params.anisoRotation;
    if (checkMaterialFlag(params.flags,
                          MaterialFlags::HasAnisotropicTexture)) {
        float2 tex_value = textureFetch2D<float2>(
            textures[TextureConstants::anisoOffset], uv.x, uv.y, mip_level);
        float2 aniso_v = tex_value * 2.f - 1.f;

        mat.anisoScale = length(aniso_v);
        mat.anisoRotation = atan2f(aniso_v.y, aniso_v.x);
    }

    mat.attenuationColor = params.attenuationColor;
    mat.attenuationDistance = params.attenuationDistance;
#endif

    return mat;
}

__forceinline__ float fetchDiffuseAverageAlbedo(float lookup_f0,
                                                float roughness)
{
    return textureFetch2D<float>(launchInput.precomputed.diffuseAverage,
                           roughness, lookup_f0, 0);
}

__forceinline__ float fetchDiffuseDirectionalAlbedo(float lookup_f0,
                                                    float roughness,
                                                    float cos_theta)
{
    return textureFetch3D<float>(launchInput.precomputed.diffuseDirectional,
                           cos_theta, roughness, lookup_f0, 0);
}

__forceinline__ float fetchMicrofacetMSAverageAlbedo(float roughness)
{
    return textureFetch1D<float>(launchInput.precomputed.microfacetAverage,
                           roughness, 0);
}

__forceinline__ float fetchMicrofacetMSDirectionalAlbedo(float roughness,
                                                         float cos_theta)
{
    return textureFetch2D<float>(launchInput.precomputed.microfacetDirectional,
                           cos_theta, roughness, 0);
}

__forceinline__ float sampleMSMicrofacetAngle(float roughness, float u)
{
    return textureFetch2D<float>(
        launchInput.precomputed.microfacetDirectionalInverse, u, roughness, 0);
}

__forceinline__ BSDFParams buildBSDF(const Material &material, float3 wo)
{
    static constexpr float min_alpha = 0.0064f;

    static constexpr float prior_ior = 1.f;
    float ior_ratio = (material.ior - prior_ior) / (material.ior + prior_ior);
    float base_f0 = ior_ratio * ior_ratio;

    float3 dielectric_f0 = fminf(make_float3(1.f), base_f0 * 
        material.rhoSpecular) * material.specularScale;
    float dielectric_f90 = material.specularScale;

    // Core weights
    float transmission_weight = material.transmission;
    float opaque_weight = (1.f - material.transmission);
    float dielectric_weight = (1.f - material.metallic);

    float3 base_dielectric = material.rho * dielectric_weight;

    // Microfacet params
    // Scale between specular and metallic fresnel
    float3 shared_f0 = lerp(material.rho, dielectric_f0, dielectric_weight);
    float shared_f90 = fmaf(dielectric_weight, dielectric_f90,
                            material.metallic);

    float3 ss_fresnel_estimate =
        computeFresnel(shared_f0, make_float3(shared_f90), wo.z);
    float3 ss_transmissive_fresnel_estimate =
        1.f - computeFresnel(dielectric_f0, make_float3(dielectric_f90), wo.z);

    float alpha = material.roughness * material.roughness;
    if (alpha < min_alpha) {
        alpha = 0.f;
    }
    
    // Multiscattering / energy conservation params
    // FIXME, it is pretty ambiguous whether the lookup tables wants
    // shared_f0 or dielectric_f0 here
    float diffuse_lookup_f0 = max(shared_f0.x, max(shared_f0.y, shared_f0.z));

    float ms_microfacet_avg_albedo =
        fetchMicrofacetMSAverageAlbedo(material.roughness);
    float3 ms_fresnel_avg = 1.f / 21.f * shared_f90 + 20.f / 21.f * shared_f0;

    float ms_avg_albedo_compl = 1.f - ms_microfacet_avg_albedo;
    float ms_dir_albedo_compl =
        (1.f - fetchMicrofacetMSDirectionalAlbedo(material.roughness, wo.z));

    float3 ms_fresnel =
        (ms_fresnel_avg * ms_fresnel_avg * ms_microfacet_avg_albedo) /
        (1.f - ms_fresnel_avg * ms_avg_albedo_compl);

    float3 ms_microfacet_weight = ms_fresnel * ms_dir_albedo_compl;

#ifdef ADVANCED_MATERIAL
    // Clearcoat params
    float clearcoat_reflectance_estimate =
        material.clearcoatScale * computeFresnel(0.04f, 1.f, wo.z);
    float clearcoat_alpha =
        material.clearcoatRoughness * material.clearcoatRoughness;
    if (clearcoat_alpha < min_alpha) {
        clearcoat_alpha = 0.f;
    }

    float clearcoat_prob = clearcoat_reflectance_estimate;
    float not_clearcoat_prob = 1.f - clearcoat_prob;
#else
    constexpr float not_clearcoat_prob = 1.f;
#endif


    // Compute importance sampling weights
    float dielectric_luminance =
        not_clearcoat_prob * rgbToLuminance(base_dielectric);
    float diffuse_prob =
        not_clearcoat_prob * dielectric_luminance * opaque_weight;
    float microfacet_prob =
        not_clearcoat_prob * rgbToLuminance(ss_fresnel_estimate);
    float microfacet_ms_prob =
        not_clearcoat_prob * rgbToLuminance(ms_fresnel);
    float transmission_prob = not_clearcoat_prob * dielectric_luminance *
        transmission_weight * rgbToLuminance(ss_transmissive_fresnel_estimate);

    float prob_sum = 
#ifdef ADVANCED_MATERIAL
        clearcoat_prob + 
#endif
        diffuse_prob + microfacet_prob +
        microfacet_ms_prob + transmission_prob;
    if (prob_sum > 0.f) {
        float inv_prob = 1.f / prob_sum;
        diffuse_prob *= inv_prob;
        microfacet_prob *= inv_prob;
        microfacet_ms_prob *= inv_prob;
        transmission_prob *= inv_prob;

#ifdef ADVANCED_MATERIAL
        clearcoat_prob *= inv_prob;
#endif
    }

    return BSDFParams {
        base_dielectric * opaque_weight,
        material.transparencyMask,
        base_dielectric * transmission_weight,
        transmission_weight,
        shared_f0,
        shared_f90,
        dielectric_f0,
        dielectric_f90,
        alpha,
        material.roughness,
        material.specularScale,
        diffuse_lookup_f0,
        fetchDiffuseAverageAlbedo(diffuse_lookup_f0, material.roughness),
        ms_microfacet_weight,
        ms_avg_albedo_compl,
        diffuse_prob,
        microfacet_prob,
        microfacet_ms_prob,
        transmission_prob,
#ifdef ADVANCED_MATERIAL
        material.clearcoatScale,
        clearcoat_alpha,
        clearcoat_prob,
#endif
    };
}

enum class BSDFFlags : uint32_t {
    Invalid = 1 << 0,
    Delta = 1 << 1,
    Diffuse = 1 << 2,
    MicrofacetReflection = 1 << 3,
    MicrofacetTransmission = 1 << 4,
};

BSDFFlags & operator|=(BSDFFlags &a, BSDFFlags b)
{
    a = BSDFFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

bool operator&(BSDFFlags a, BSDFFlags b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}

template <typename T>
struct SampleResult {
    float3 dir;
    T weight;
    BSDFFlags flags;
};

struct ShadeResult {
    float3 color;
    float3 bounceDir;
    float3 bounceProb;
    BSDFFlags flags;
};

__forceinline__ float3 cosineHemisphere(float2 uv)
{
    const float r = sqrtf(uv.x);
    const float phi = 2.0f * M_PI * uv.y;
    float2 disk = r * make_float2(cosf(phi), sinf(phi));
    float3 hemisphere = make_float3(disk.x, disk.y,
        sqrtf(fmaxf(0.0f, 1.0f - dot(disk, disk))));

    return hemisphere;
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

// Enterprise PBR diffuse BRDF
// diffuseWeight and diffuseBSDF are separated to allow concentricHemisphere
// sampling which divides out the M_1_PI * wi_dot_n factor when sampling
__forceinline__ float3 diffuseWeight(const BSDFParams &params,
                                     float wo_dot_n, float wi_dot_n)
{
    float E_diffuse_o = fetchDiffuseDirectionalAlbedo(params.diffuseLookupF0,
                                                      params.roughness,
                                                      wo_dot_n);

    float E_diffuse_i = fetchDiffuseDirectionalAlbedo(params.diffuseLookupF0,
                                                      params.roughness,
                                                      wi_dot_n);

    float Bc = ((1.f - E_diffuse_o) * (1.f - E_diffuse_i)) /
            (1.f - params.diffuseAverageAlbedo);

    float weight = lerp(1.f, Bc, params.diffuseLambertScale);

    if (min(wo_dot_n, wi_dot_n) < 1e-6f) {
        return make_float3(0.f);
    } else {
        return weight * params.rhoDiffuse;
    }
}

__forceinline__ float3 diffuseBSDF(const BSDFParams &bsdf_params,
                                   float wo_dot_n, float wi_dot_n)
{
    return diffuseWeight(bsdf_params, wo_dot_n, wi_dot_n) * M_1_PI * wi_dot_n;
}

__forceinline__ SampleResult<float3> sampleDiffuse(
    const BSDFParams &bsdf_params, float3 wo, float2 sample_uv)
{
    float3 wi = concentricHemisphere(sample_uv);
    float3 weight = diffuseWeight(bsdf_params, wo.z, wi.z);

    return {
        wi,
        weight,
        BSDFFlags::Diffuse,
    };
}

// Single scattering GGX Microfacet BRDF
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

template <typename T>
__forceinline__ T evalGGX(float wo_dot_n, float wi_dot_n, float n_dot_h,
                          T F, float alpha)
{
    float a2 = alpha * alpha;
    float D = ggxNDF(alpha, n_dot_h);
    float G = ggxMasking(a2, wo_dot_n, wi_dot_n);

    T specular = 0.25f * F * D * G / wo_dot_n;

    if (alpha == 0.f || min(wo_dot_n, wi_dot_n) < 1e-6f) {
        specular = makeZeroVec<T>();
    }
    
    return specular;
}

__forceinline__ float3 microfacetBSDF(const BSDFParams &params,
                                      float wo_dot_n, float wi_dot_n,
                                      float n_dot_h, float dir_dot_h)
{
    float3 F = computeFresnel(params.sharedF0, make_float3(params.sharedF90),
                              dir_dot_h);

    return evalGGX(wo_dot_n, wi_dot_n, n_dot_h, F, params.alpha);
}

__forceinline__ float3 microfacetTransmissiveBSDF(const BSDFParams &params,
                                                  float3 wo,
                                                  float3 wi)
{
    wi.z *= -1.f;

    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = rsqrtf(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    float3 F = 1.f - computeFresnel(params.transmissiveF0,
                                    make_float3(params.transmissiveF90),
                                    dir_dot_h);
    float3 microfacet_response = evalGGX(wo.z, wi.z,
        n_dot_h, F, params.alpha);

    return microfacet_response * params.rhoTransmissive;
}

__forceinline__ float3 sampleGGX(float alpha, float3 wo, float2 sample_uv)
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

template <typename T, bool transmission = false>
__forceinline__ SampleResult<T> sampleMicrofacet(
    float3 wo, float2 sample_uv, T F0, T F90, float alpha)
{
    float3 wi {};
    T weight {};
    BSDFFlags flags {};

    if (wo.z < 1e-6f) {
        wi = make_float3(0.f);
        weight = makeZeroVec<T>();

        flags |= BSDFFlags::Invalid;
    } else if (alpha == 0.f) {
        wi = make_float3(-wo.x, -wo.y, wo.z);
        T F = computeFresnel(F0, F90, wo.z);
        if constexpr (transmission) {
            weight = 1.f - F;
        } else {
            weight = F;
        }

        flags |= BSDFFlags::Delta;
    } else {
        float3 h = sampleGGX(alpha, wo, sample_uv);
        float cos_half_out = dot(wo, h);
        wi = 2.f * cos_half_out * h - wo;

        if (wi.z < 1e-6f) {
            weight = makeZeroVec<T>();
        } else {
            float a2 = alpha * alpha;

            float G = ggxMasking(a2, wo.z, wi.z);
            float GG1_out = G * (1.f + ggxLambda(wo.z, a2));
            T F = computeFresnel(F0, F90, cos_half_out);

            if constexpr (transmission) {
                weight = (1.f - F) * GG1_out;
            } else {
                weight = F * GG1_out;
            }
        }
    }

    if constexpr (transmission) {
        flags |= BSDFFlags::MicrofacetTransmission;
    } else {
        flags |= BSDFFlags::MicrofacetReflection;
    }

    return {
        wi,
        weight,
        flags,
    };
}

__forceinline__ SampleResult<float3> sampleMicrofacetShared(
    const BSDFParams &params, float3 wo, float2 sample_uv)
{
    return sampleMicrofacet(wo, sample_uv, params.sharedF0,
                            make_float3(params.sharedF90), params.alpha);
}

__forceinline__ SampleResult<float3> sampleMicrofacetTransmission(
    const BSDFParams &params, float3 wo, float2 sample_uv)
{
    SampleResult<float3> sample = sampleMicrofacet<float3, true>(wo, sample_uv,
        params.transmissiveF0, make_float3(params.transmissiveF90),
        params.alpha);

    // Reflect back for transmission
    sample.dir.z *= -1.f;
    sample.weight *= params.rhoTransmissive;

    return sample;
}

__forceinline__ float3 microfacetMSBSDF(const BSDFParams &params,
                                        float wo_dot_n, float wi_dot_n)
{
    float3 ms_contrib =
        (1.f - fetchMicrofacetMSDirectionalAlbedo(params.roughness, wi_dot_n)) *
        params.microfacetMSWeight * M_1_PI * wi_dot_n /
        params.microfacetMSAvgAlbedoComplement;

    if (params.alpha == 0.f || min(wo_dot_n, wi_dot_n) < 1e-6f) {
        ms_contrib = make_float3(0.f);
    }

    return ms_contrib;
}

__forceinline__ SampleResult<float3> sampleMSMicrofacet(
    const BSDFParams &params, float3 wo, float2 sample_uv)
{
    float theta = sampleMSMicrofacetAngle(params.roughness, sample_uv.x);
    float phi = 2 * M_PI * sample_uv.y;
    float2 circle_dir = make_float2(cosf(phi), sinf(phi));
    float xy_mag = sqrtf(1.f - theta * theta);

    float3 wi = make_float3(circle_dir.x * xy_mag,
                            circle_dir.y * xy_mag, theta);

    // 1/(2pi) factor cancels out the 2 from the theta PDF
    float3 weight = params.microfacetMSWeight;

    SampleResult<float3> result {
        wi,
        weight,
        BSDFFlags::MicrofacetReflection,
    };

    if (min(wo.z, wi.z) < 1e-6f) {
        result.dir = make_float3(0.f);
        result.weight = make_float3(0.f);
        result.flags = BSDFFlags::Invalid;
    }

    return result;
}

#ifdef ADVANCED_MATERIAL
__forceinline__ pair<float, float> clearcoatBSDF(
    const BSDFParams &params, float wo_dot_n, float wi_dot_n,
    float n_dot_h, float dir_dot_h)
{
    float F = computeFresnel(0.04f, 1.f, dir_dot_h);

    float response = evalGGX(wo_dot_n, wi_dot_n, n_dot_h, F,
                             params.clearcoatAlpha);

    float max_fresnel_n = max(computeFresnel(0.04f, 1.f, wo_dot_n),
                              computeFresnel(0.04f, 1.f, wi_dot_n));

    return {
        response * params.clearcoatScale,
        1.f - params.clearcoatScale * max_fresnel_n,
    };
}

__forceinline__ SampleResult<float3> sampleClearcoat(
    const BSDFParams &params, float3 wo, float2 sample_uv)
{
    SampleResult<float> sample = sampleMicrofacet(wo, sample_uv,
        0.04f, 1.f, params.clearcoatAlpha);
    sample.weight *= params.clearcoatScale;

    return {
        sample.dir,
        make_float3(sample.weight),
        sample.flags,
    };
}
#endif

__forceinline__ float3 evalBSDF(const BSDFParams &params,
                                float3 wo, float3 wi)
{
    // Hammon 2017
    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = rsqrtf(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    float3 diffuse = diffuseBSDF(params, wo.z, wi.z);

    float3 microfacet =
        microfacetBSDF(params, wo.z, wi.z, n_dot_h, dir_dot_h);

    float3 microfacet_ms =
        microfacetMSBSDF(params, wo.z, wi.z);

    float3 transmissive =
        microfacetTransmissiveBSDF(params, wo, wi);

    float3 base = diffuse + microfacet + microfacet_ms + transmissive;

#ifdef ADVANCED_MATERIAL
    auto [clearcoat_response, base_scale] =
        clearcoatBSDF(params, wo.z, wi.z, n_dot_h, dir_dot_h);

    return base * base_scale + clearcoat_response;
#else
    return base;
#endif
}

__forceinline__ SampleResult<float3> sampleBSDF(Sampler &sampler,
                                                const BSDFParams &params,
                                                float3 wo)
{
    float selector = sampler.get1D();
    float2 uv = sampler.get2D();

    float cdf[] {
        params.diffuseProb,
        params.microfacetProb,
        params.microfacetMSProb,
        params.transmissionProb,
#ifdef ADVANCED_MATERIAL
        params.clearcoatProb,
#endif
    };

    cdf[1] += cdf[0];
    cdf[2] += cdf[1];
    cdf[3] += cdf[2];

#ifdef ADVANCED_MATERIAL
    cdf[4] += cdf[3];
#endif

    SampleResult<float3> sample {
        make_float3(0.f),
        make_float3(0.f),
        BSDFFlags::Invalid,
    };

    if (selector < cdf[0]) {
        sample = sampleDiffuse(params, wo, uv);
        sample.weight /= params.diffuseProb;
    } else if (selector < cdf[1]) {
        sample = sampleMicrofacetShared(params, wo, uv);
        sample.weight /= params.microfacetProb;
    } else if (selector < cdf[2]) {
        sample = sampleMSMicrofacet(params, wo, uv);
        sample.weight /= params.microfacetMSProb;
    } else if (selector < cdf[3]) {
        sample = sampleMicrofacetTransmission(params, wo, uv);
        sample.weight /= params.transmissionProb;
    } 
#ifdef ADVANCED_MATERIAL
    else if (selector < cdf[4]) {
        sample = sampleClearcoat(params, wo, uv);
        sample.weight /= params.clearcoatProb;
    }
#endif

    return sample;
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

float sign(float x)
{
    return x >= 0.f ? 1.f : -1.f;
}

// Ray Tracing Gems 16.5.4.2
float3 octSphereMap(float2 u)
{
    u = u * 2.f - 1.f;

    // Compute radius r (branchless)
    float d = 1.f - (fabsf(u.x) + fabsf(u.y));
    float r = 1.f - fabsf(d);

    // Compute phi in the first quadrant (branchless, except for the
    // division-by-zero test), using sign(u) to map the result to the
    // correct quadrant below
    float phi = (r == 0.f) ? 0.f :
        M_PI_4 * ((fabsf(u.y) - fabsf(u.x)) / r + 1.f);

    float f = r * sqrtf(2.f - r * r);
    float x = f * sign(u.x) * cosf(phi);
    float y = f * sign(u.y) * sinf(phi);
    float z = sign(d) * (1.f - r * r);

    return make_float3(x, y, z);
}

struct LightSample {
    float3 toLight;
    float3 weight; // Irradiance / PDF
};

struct LightInfo {
    LightSample sample;
    float3 shadowRayOrigin;
    float shadowRayLength;
};

__forceinline__ PointLight unpackPointLight(float4 data0, float4 data1)
{
    return PointLight {
        make_float3(data0.y, data0.z, data0.w),
        make_float3(data1.x, data1.y, data1.z),
    };
}

__forceinline__ PortalLight unpackPortalLight(float4 data1, float4 data2,
                                              float4 data3)
{
    return PortalLight {
        make_float3(data1.x, data1.y, data1.z),
        make_float3(data1.w, data2.x, data2.y),
        make_float3(data2.z, data2.w, data3.x),
        make_float3(data3.y, data3.z, data3.w),
    };
}

LightType unpackLight(const Environment &env,
                      uint32_t light_idx,
                      PointLight *point_light,
                      PortalLight *portal_light)
{
    const PackedLight &packed = env.lights[light_idx];
    float4 data0 = packed.data[0];
    LightType light_type = LightType(__float_as_uint(data0.x));

    float4 data1 = packed.data[1];
    float4 data2 = packed.data[2];
    float4 data3 = packed.data[3];

    if (light_type == LightType::Point) {
        *point_light = unpackPointLight(data0, data1);
    } else if (light_type == LightType::Portal) {
        *portal_light = unpackPortalLight(data1, data2, data3);
    }

    return light_type;
}

__forceinline__ float2 dirToLatLong(float3 dir)
{
    float3 n = normalize(dir);
    
    return make_float2(
        atan2f(n.x, -n.z) * (M_1_PI / 2.f) + 0.5f,
        acosf(n.y) * M_1_PI);
}

__forceinline__ float3 evalEnvMap(cudaTextureObject_t env_tex,
                                  float3 dir)
{
    float2 uv = dirToLatLong(dir);

    float4 v = textureFetch2D<float4>(env_tex, uv.x, uv.y, 0.f);

    v *= 90.f;

    return make_float3(v.x, v.y, v.z);
}

__forceinline__ LightSample sampleEnvMap(cudaTextureObject_t env_tex,
    float2 uv, float inv_selection_pdf)
{
    float3 dir = octSphereMap(uv);

    float3 irradiance = evalEnvMap(env_tex, dir);

    const float inv_pdf = 4.f * M_PI;

    return LightSample {
        dir,
        irradiance * inv_pdf * inv_selection_pdf,
    };
}

__forceinline__ float3 getPortalLightPoint(
    const PortalLight &light, float2 uv)
{
    float3 upper = lerp(light.corners[0], light.corners[3], uv.x);
    float3 lower = lerp(light.corners[1], light.corners[2], uv.x);

    return lerp(lower, upper, uv.y);
}

__forceinline__ LightSample samplePortal(
    const PortalLight &light, cudaTextureObject_t env_tex,
    float3 to_light, float inv_selection_pdf)
{
    float3 dir = normalize(to_light);

    float3 irradiance = evalEnvMap(env_tex, dir);

    const float inv_pdf = fabsf(
        length(light.corners[3] - light.corners[0]) *
        length(light.corners[1] - light.corners[0]));

    return LightSample {
        dir,
        irradiance * inv_pdf * inv_selection_pdf,
    };
}

__forceinline__ pair<LightSample, float> samplePointLight(
    const PointLight &light, float3 origin, float inv_selection_pdf)
{
    float3 to_light = light.position - origin;

    float dist_to_light2 = dot(to_light, to_light);

    float3 irradiance = light.rgb / dist_to_light2;

    float3 weight = irradiance * inv_selection_pdf;

    return {
        LightSample {
            normalize(to_light),
            weight,
        },
        dist_to_light2,
    };
}

__forceinline__ LightInfo sampleLights(Sampler &sampler,
    const Environment &env, cudaTextureObject_t env_tex, 
    float3 origin, float3 base_normal)
{
    uint32_t total_lights = env.numLights + 1;

    uint32_t light_idx = min(uint32_t(sampler.get1D() * total_lights),
                             total_lights - 1);

    float2 light_sample_uv = sampler.get2D();

    float inv_selection_pdf = float(total_lights);

    PointLight point_light {};
    PortalLight portal_light {};
    LightType light_type {};

    if (light_idx < env.numLights) {
        light_type =
            unpackLight(env, light_idx, &point_light, &portal_light);
    } else {
        light_type = LightType::Environment;
    }

    float3 light_position {};
    float3 dir_check {};
    LightSample light_sample {};
    if (light_type == LightType::Point) {
        light_position = point_light.position;
        dir_check = light_position - origin;
    } else if (light_type == LightType::Portal) {
        light_position = getPortalLightPoint(portal_light, light_sample_uv);
        dir_check = light_position - origin;
    } else {
        light_sample = sampleEnvMap(env_tex, light_sample_uv, inv_selection_pdf);
        dir_check = light_sample.toLight;
    }

    float3 shadow_offset_normal =
        dot(dir_check, base_normal) > 0 ? base_normal : -base_normal;

    float3 shadow_origin =
        offsetRayOrigin(origin, shadow_offset_normal);

    float shadow_len {};
    if (light_type == LightType::Point) {
        float shadow_len2;
        tie(light_sample, shadow_len2) =
            samplePointLight(point_light, shadow_origin, inv_selection_pdf);

        shadow_len = sqrtf(shadow_len2);
    } else if (light_type == LightType::Portal) {
        float3 to_light = light_position - shadow_origin;
        light_sample = samplePortal(portal_light, env_tex, to_light,
                                    inv_selection_pdf);

        shadow_len = 10000.f;
    } else {
        shadow_len = 10000.f;
    }

    return LightInfo {
        light_sample,
        shadow_origin,
        shadow_len,
    };
}

struct TangentFrame {
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

__forceinline__ float3 worldToLocal(float3 vec, const TangentFrame &frame) 
{
    return make_float3(dot(vec, frame.tangent), dot(vec, frame.bitangent),
        dot(vec, frame.normal));
}

__forceinline__ float3 localToWorld(float3 vec, const TangentFrame &frame)
{
    return vec.x * frame.tangent + vec.y * frame.bitangent +
        vec.z * frame.normal;
}

__forceinline__ ShadeResult shade(Sampler &sampler, const Material &material,
                                  const LightSample &light_sample,
                                  float3 outgoing,
                                  const TangentFrame &frame)
{
    // These normalizations shouldn't be necessary, but z component
    // needs to be accurate for cos angle
    float3 wo = normalize(worldToLocal(outgoing, frame));
    float3 wi = normalize(worldToLocal(light_sample.toLight, frame));

    BSDFParams bsdf = buildBSDF(material, wo);

    float3 bsdf_response = evalBSDF(bsdf, wo, wi);

    float3 nee = bsdf_response * light_sample.weight;

    // Compute bounce
    SampleResult<float3> bounce = sampleBSDF(sampler, bsdf, wo);

    float3 bounce_dir = localToWorld(bounce.dir, frame);

    return {
        nee,
        bounce_dir,
        bounce.weight,
        bounce.flags,
    };
}

__forceinline__ pair<float3, float3> computeCameraRay(
    const Camera &camera, uint3 idx, uint3 dim, Sampler &sampler)
{
    float2 jitter = sampler.get2D();

    float2 jittered_raster = make_float2(idx.x, idx.y) + jitter;

    float2 screen = make_float2((2.f * jittered_raster.x) / dim.x - 1,
                                (2.f * jittered_raster.y) / dim.y - 1);

    float3 direction = normalize(
        camera.right * screen.x + camera.up * screen.y + camera.view);

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

struct RayPayload {
    float3 barycentrics;
    uint32_t triIndex;
    uint32_t instanceIndex;
    uint32_t meshIndex;
};

__forceinline__ void packHitPayload(float2 barycentrics,
                                    uint32_t triangle_index,
                                    uint32_t instance_index,
                                    uint32_t mesh_index)
{
    triangle_index <<= 8;
    triangle_index |= ((mesh_index >> 8) & 0xFF);

    instance_index <<= 8;
    instance_index |= (mesh_index & 0xFF);

    optixSetPayload_0(unormFloat2To32(barycentrics));
    optixSetPayload_1(triangle_index);
    optixSetPayload_2(instance_index);
}

__forceinline__ RayPayload unpackHitPayload(uint32_t payload_0,
                                            uint32_t payload_1,
                                            uint32_t payload_2)
{
    float2 raw_barys = unormFloat2From32(payload_0);
    float3 barys = computeBarycentrics(raw_barys);
    uint32_t raw_tri = payload_1;
    uint32_t raw_inst = payload_2;
    uint32_t tri_idx = raw_tri >> 8;
    uint32_t instance_idx = raw_inst >> 8;
    uint32_t mesh_idx = raw_inst & 0xff | ((raw_tri & 0xFF) << 8);

    return {
        barys,
        tri_idx,
        instance_idx,
        mesh_idx,
    };
}

__forceinline__ pair<float3, float4> decodeNormalTangent(uint3 packed)
{
    auto octDecode = [](float2 f) {
        f = f * 2.f - 1.f;
        // https://twitter.com/Stubbesaurus/status/937994790553227264
        float3 n = make_float3(f.x, f.y, 1.f - fabsf(f.x) - fabsf(f.y));
        float t = __saturatef(-n.z);
        n.x += n.x >= 0.f ? -t : t;
        n.y += n.y >= 0.f ? -t : t;
        return normalize(n);
    };

    float3 normal {
        __half2float(__ushort_as_half(packed.x & 0xFFFF)),
        __half2float(__ushort_as_half(packed.x >> 16)),
        __half2float(__ushort_as_half(packed.y & 0xFFFF)),
    };

    float sign = __half2float(__ushort_as_half(packed.y >> 16));

    int2 snormtan {
        int16_t(packed.z & 0xFFFF),
        int16_t(packed.z >> 16),
    };

    float2 oct_tan = clamp(
        make_float2(__int2float_rn(snormtan.x), __int2float_rn(snormtan.y)) *
            3.0518509475997192297128208258309e-5f,
        -1.f, 1.f);

    float3 tangent = octDecode(oct_tan);

    return {
        normal,
        make_float4(tangent.x, tangent.y, tangent.z, sign),
    };
}

__forceinline__ DeviceVertex unpackVertex(
    const DevicePackedVertex &packed)
{
    float4 a = packed.data[0];
    float4 b = packed.data[1];

    uint3 packed_normal_tangent = make_uint3(
        __float_as_uint(a.w), __float_as_uint(b.x), __float_as_uint(b.y));
    auto [normal, tangent_and_sign] =
        decodeNormalTangent(packed_normal_tangent);

    return DeviceVertex {
        make_float3(a.x, a.y, a.z),
        normal,
        tangent_and_sign,
        make_float2(b.z, b.w),
    };
}

__forceinline__ Triangle fetchTriangle(
    const DevicePackedVertex *vertex_buffer,
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
        tri.a.tangentAndSign * barys.x +
            tri.b.tangentAndSign * barys.y +
            tri.c.tangentAndSign * barys.z,
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

inline TangentFrame computeTangentFrame(const DeviceVertex &v,
                                        const MaterialParams &mat_params,
                                        const cudaTextureObject_t *textures)
{
    float3 n = v.normal;
    float3 t = make_float3(v.tangentAndSign.x, v.tangentAndSign.y,
                           v.tangentAndSign.z);

    float bitangent_sign = v.tangentAndSign.w;
    float3 b = cross(n, t) * bitangent_sign;

    float3 perturb = make_float3(0.f, 0.f, 1.f);
    if (checkMaterialFlag(mat_params.flags, MaterialFlags::HasNormalMap)) {
        float2 xy = textureFetch2D<float2>(textures[TextureConstants::normalOffset],
                                    v.uv.x, v.uv.y, 0);

        float2 centered = xy * 2.f - 1.f;
        float length2 = clamp(dot(centered, centered), 0.f, 1.f);

        perturb = make_float3(
            centered.x,
            centered.y,
            sqrtf(1.f - length2));
    } 

    // Perturb normal
    n = normalize(t * perturb.x + b * perturb.y + n * perturb.z);
    // Ensure perpendicular (if new normal is parallel to old tangent... boom)
    t = normalize(t - n * dot(n, t));
    b = cross(n, t) * bitangent_sign;

    return {
        t,
        b,
        n,
    };
}

__forceinline__ float3 transformNormal(const TransformMat &w2o, float3 n)
{
    float4 r1 = w2o.rows[0];
    float4 r2 = w2o.rows[1];
    float4 r3 = w2o.rows[2];

    return make_float3(
        r1.x * n.x + r2.x * n.y + r3.x * n.z,
        r1.y * n.x + r2.y * n.y + r3.y * n.z,
        r1.z * n.x + r2.z * n.y + r3.z * n.z);

}

__forceinline__ float3 transformPosition(const TransformMat &o2w, float3 p)
{
    float4 r1 = o2w.rows[0];
    float4 r2 = o2w.rows[1];
    float4 r3 = o2w.rows[2];

    return make_float3(
        r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w,
        r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w,
        r3.x * p.x + r3.y * p.y + r3.z * p.z + r3.w);
}

__forceinline__ float3 transformVector(const TransformMat &o2w, float3 v)
{
    float4 r1 = o2w.rows[0];
    float4 r2 = o2w.rows[1];
    float4 r3 = o2w.rows[2];

    return make_float3(
        r1.x * v.x + r1.y * v.y + r1.z * v.z,
        r2.x * v.x + r2.y * v.y + r2.z * v.z,
        r3.x * v.x + r3.y * v.y + r3.z * v.z);
}

inline float3 faceforward(const float3 &n, const float3 &i, const float3 &nref)
{
  return n * copysignf(1.0f, dot(i, nref));
}

inline TangentFrame tangentFrameToWorld(const TransformMat &o2w,
                                        const TransformMat &w2o,
                                        TangentFrame frame,
                                        float3 ray_dir)
{
    frame.tangent = normalize(transformVector(o2w, frame.tangent));
    frame.bitangent = normalize(transformVector(o2w, frame.bitangent));
    frame.normal = normalize(transformNormal(w2o, frame.normal));
    frame.normal = faceforward(frame.normal, -ray_dir, frame.normal);
                                                               
    return frame;                                              
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    uint32_t batch_idx = launchInput.baseBatchOffset + idx.z;

    uint32_t base_out_offset = 
        4 * (batch_idx * dim.y * dim.x + idx.y * dim.x + idx.x);

    uint32_t base_aux_offset = 
        3 * (batch_idx * dim.y * dim.x + idx.y * dim.x + idx.x);

    const auto [cam, env] = unpackEnv(batch_idx);
    cudaTextureObject_t env_tex = env.textureHandles[0];
    
    float3 pixel_radiance = make_float3(0.f);
    uint16_t instance_id = 0xFFFF;

#ifdef AUXILIARY_OUTPUTS
    float3 aux_normal = make_float3(0.f);
    float3 aux_albedo = make_float3(0.f);
#endif

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

        BSDFFlags bounce_flags {};

#if MAX_DEPTH != (1u)
#pragma unroll 1
#endif
        for (int32_t path_depth = 0; path_depth < MAX_DEPTH;
             path_depth++) {

            if (path_prob.x == 0.f && path_prob.y == 0.f &&
                path_prob.z == 0.f) {
                break;
            }

            // Trace shade ray
            unsigned int payload_0;
            // Overwrite payload_1, miss sets it to ~0u
            unsigned int payload_1 = 0;
            unsigned int payload_2 = uint32_t(bounce_flags);

            // FIXME Min T for both shadow and this ray
            optixTrace(
                    env.tlas,
                    ray_origin,
                    ray_dir,
                    0.f, // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(3), // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload_0,
                    payload_1,
                    payload_2);

            // Miss, hit env map
            if (payload_1 == ~0u) {
                if (path_depth == 0 ||
                    BSDFFlags(payload_2) & BSDFFlags::Delta) {
                    sample_radiance += evalEnvMap(env_tex, ray_dir) * path_prob;
                }
                break;
            }

            auto [barys, tri_idx, instance_idx, mesh_idx] =
                unpackHitPayload(payload_0, payload_1, payload_2);

            if (path_depth == 0) {
                instance_id = instance_idx;
            }

            Instance inst = unpackInstance(env.instances[instance_idx]);

            MeshInfo mesh_info =
                unpackMeshInfo(env.meshInfos[inst.meshOffset + mesh_idx]);

            auto [o2w, w2o] = unpackTransforms(env.transforms[instance_idx]);

            uint32_t index_offset = mesh_info.indexOffset + tri_idx * 3;
            Triangle hit_tri = fetchTriangle(env.vertexBuffer,
                                             env.indexBuffer + index_offset);
            DeviceVertex interpolated = interpolateTriangle(hit_tri, barys);
            float3 obj_geo_normal = computeGeometricNormal(hit_tri);

            float3 world_position =
                transformPosition(o2w, interpolated.position);
            float3 world_geo_normal =
                transformNormal(w2o, obj_geo_normal);
            world_geo_normal = normalize(world_geo_normal);

            // Unpack materials
            uint32_t material_idx =
                env.instanceMaterials[inst.materialOffset + mesh_idx];
            const cudaTextureObject_t *textures = env.textureHandles + 1 +
                material_idx * TextureConstants::numTexturesPerMaterial;

            MaterialParams material_params =
                unpackMaterialParams(env.materialBuffer[material_idx]);

            TangentFrame obj_tangent_frame =
                computeTangentFrame(interpolated, material_params, textures);
            TangentFrame world_tangent_frame =
                tangentFrameToWorld(o2w, w2o, obj_tangent_frame, ray_dir);

            Material material = processMaterial(material_params, textures,
                interpolated.uv, 0);

#ifdef AUXILIARY_OUTPUTS
            if (path_depth == 0) {
                float3 view_normal = make_float3(
                    dot(normalize(cam.right), world_tangent_frame.normal),
                    dot(normalize(cam.up) * -1.f, world_tangent_frame.normal),
                    dot(normalize(cam.view) * -1.f, world_tangent_frame.normal));

                view_normal = normalize(view_normal);

                float3 albedo = material.rho;

                aux_normal += view_normal / SPP;
                aux_albedo += albedo / SPP;
            }
#endif

            LightInfo light_info = sampleLights(sampler, env, env_tex,
                world_position, world_geo_normal);

            auto [color, bounce_dir, bounce_prob, cur_bounce_flags] =
                shade(sampler, material, light_info.sample, -ray_dir,
                      world_tangent_frame);

            bounce_flags = cur_bounce_flags;

            float alpha_check = sampler.get1D();
            bool pass_through = material.transparencyMask == 0.f ||
                alpha_check > material.transparencyMask;

            float3 next_dir = pass_through ? ray_dir : bounce_dir;

            float3 bounce_offset_normal =
                dot(next_dir, world_geo_normal) > 0 ?
                    world_geo_normal : -world_geo_normal;
            ray_origin = offsetRayOrigin(world_position, bounce_offset_normal);

            // Start setup for next bounce
            if (!pass_through) {
                path_prob *= bounce_prob;
            }

            ray_dir = next_dir;

            payload_0 = pass_through;
            payload_1 = 0;
            payload_2 = uint32_t(bounce_flags);
            optixTrace(
                    env.tlas,
                    light_info.shadowRayOrigin,
                    light_info.sample.toLight,
                    0.f,                // Min intersection distance
                    light_info.shadowRayLength, // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(1), // Skip transparent
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload_0,
                    payload_1,
                    payload_2);

            bounce_flags = BSDFFlags(payload_2);
            bool unoccluded = payload_1 == ~0u;

            if (unoccluded && !payload_0) {
                float3 contrib = path_prob * color;
#ifdef INDIRECT_CLAMP
                if (path_depth > 0) {
                    contrib = fminf(contrib, make_float3(INDIRECT_CLAMP));
                }
#endif
                sample_radiance += contrib;
            }
        }

        pixel_radiance += sample_radiance / SPP;
    }

    setOutput(base_out_offset, pixel_radiance, instance_id);

#ifdef AUXILIARY_OUTPUTS
    setAuxiliaries(base_aux_offset, aux_normal, aux_albedo);
#endif
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_1(~0u);
}

extern "C" __global__ void __closesthit__ch()
{
    packHitPayload(optixGetTriangleBarycentrics(), optixGetPrimitiveIndex(),
                   optixGetInstanceIndex(), optixGetSbtGASIndex());
}
