#ifndef RLPBR_VK_MATERIALS_GLSL_INCLUDED
#define RLPBR_VK_MATERIALS_GLSL_INCLUDED

#include "inputs.glsl"

struct MaterialParams {
    vec3 baseColor;
    float baseTransmission;
    vec3 baseSpecular;
    float specularScale;
    float ior;
    float baseMetallic;
    float baseRoughness;
    uint32_t flags;

    float clearcoat;
    float clearcoatRoughness;
    vec3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    float anisoRotation;
    vec3 baseEmittance;
};

struct Material {
    vec3 rho;
    float transmission;
    vec3 rhoSpecular;
    float specularScale;
    float ior;
    float metallic;
    float roughness;
    float transparencyMask;

    float clearcoatScale;
    float clearcoatRoughness;
    vec3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    float anisoRotation;
    vec3 emittance;
};

MaterialParams unpackMaterialParams(MatRef mat_ref, uint32_t material_id)
{
    u32vec4 data0 = fetchSceneMaterialParams(mat_ref, material_id, 0);

    MaterialParams params;
    {
        vec4 unpack = unpackUnorm4x8(data0.x);
        params.baseColor = unpack.xyz * unpack.xyz;
        params.baseTransmission = unpack.w;
    }

    {
        vec4 unpack = unpackUnorm4x8(data0.y);
        params.specularScale = unpack.x;
        params.baseMetallic = unpack.z;
        params.baseRoughness = unpack.w;

        params.ior = float(uint8_t(data0.y >> 8)) / 170.0 + 1.0;
    }

    {
        vec2 xy = unpackHalf2x16(data0.z);
        vec2 zw = unpackHalf2x16(data0.w);

        params.baseSpecular = vec3(xy, zw.x);
        params.flags = uint16_t(data0.w >> 16);
    }

    if (bool(params.flags & MaterialFlagsComplex)) {
        u32vec4 data1 = fetchSceneMaterialParams(mat_ref, material_id, 1);

        {
            vec4 unpack = unpackUnorm4x8(data1.x);
            params.clearcoat = unpack.x;
            params.clearcoatRoughness = unpack.y;
            params.attenuationColor.xy = unpack.zw;

            vec4 unpack2 = unpackUnorm4x8(data1.y);
            params.attenuationColor.z = unpack2.x;

            params.attenuationColor *= params.attenuationColor;

            params.anisoScale = unpack2.y;
            params.anisoRotation = unpack2.z;
            // unpack2.w currently unused
        }

        {
            vec2 xy = unpackHalf2x16(data1.z);
            params.attenuationDistance = xy.x;
            vec2 zw = unpackHalf2x16(data1.w);
            params.baseEmittance = vec3(xy.y, zw.x, zw.y);
        }
    } else {
        params.clearcoat = 0.0;
        params.clearcoatRoughness = 0.0;
        params.attenuationColor = vec3(1.0);
        params.anisoScale = 0.0;
        params.anisoRotation = 0.0;
        params.attenuationDistance = uintBitsToFloat(0x7f800000);
        params.baseEmittance = vec3(0.0);
    }

    return params;
}

vec3 getMaterialEmittance(MatRef mat_addr, uint32_t mat_idx)
{
    u32vec4 data1 = fetchSceneMaterialParams(mat_addr, mat_idx, 1);

    vec2 xy = unpackHalf2x16(data1.z);
    vec2 zw = unpackHalf2x16(data1.w);
    return vec3(xy.y, zw.x, zw.y);
}

Material processMaterial(MaterialParams params,
                         uint32_t base_tex_idx,
                         vec2 uv, vec4 uv_derivs)
{
    Material mat;

    mat.rho = params.baseColor;
    mat.transparencyMask = 1.f;
    if (bool(params.flags & MaterialFlagsHasBaseTexture)) {
        vec4 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsBaseOffset, uv, uv_derivs);

        mat.rho.x *= tex_value.x;
        mat.rho.y *= tex_value.y;
        mat.rho.z *= tex_value.z;
        mat.transparencyMask *= tex_value.w;
    }

    mat.metallic = params.baseMetallic;
    mat.roughness = params.baseRoughness;
    if (bool(params.flags & MaterialFlagsHasMRTexture)) {
        vec2 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsMROffset, uv, uv_derivs).xy;

        mat.roughness *= tex_value.x;
        mat.metallic *= tex_value.y;
    }
    
    mat.rhoSpecular = params.baseSpecular;
    mat.specularScale = params.specularScale;
    if (bool(params.flags & MaterialFlagsHasSpecularTexture)) {
        vec4 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsSpecularOffset, uv, uv_derivs);

        mat.rhoSpecular.x *= tex_value.x;
        mat.rhoSpecular.y *= tex_value.y;
        mat.rhoSpecular.z *= tex_value.z;
        mat.specularScale *= tex_value.w;
    }

    mat.transmission = params.baseTransmission;
    if (bool(params.flags & MaterialFlagsHasTransmissionTexture)) {
        float tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsTransmissionOffset, uv, uv_derivs).x;

        mat.transmission *= tex_value;
    }
    mat.ior = params.ior;

    mat.emittance = params.baseEmittance;
    if (bool(params.flags & MaterialFlagsHasEmittanceTexture)) {
        vec3 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsEmittanceOffset, uv, uv_derivs).xyz;

        mat.emittance.x *= tex_value.x;
        mat.emittance.y *= tex_value.y;
        mat.emittance.z *= tex_value.z;
    }

    mat.clearcoatScale = params.clearcoat;
    mat.clearcoatRoughness = params.clearcoatRoughness;
    if (bool(params.flags & MaterialFlagsHasClearcoatTexture)) {
        vec2 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsClearcoatOffset, uv, uv_derivs).xy;

        mat.clearcoatScale *= tex_value.x;
        mat.clearcoatRoughness *= tex_value.y;
    }

    mat.anisoScale = params.anisoScale;
    mat.anisoRotation = params.anisoRotation;
    if (bool(params.flags & MaterialFlagsHasAnisotropicTexture)) {
        vec2 tex_value = fetchSceneTexture(
            base_tex_idx + TextureConstantsAnisoOffset, uv, uv_derivs).xy;

        vec2 aniso_v = tex_value * 2.f - 1.f;

        mat.anisoScale = length(aniso_v);
        mat.anisoRotation = atan(aniso_v.y, aniso_v.x);
    }

    mat.attenuationColor = params.attenuationColor;
    mat.attenuationDistance = params.attenuationDistance;

    return mat;
}

#endif
