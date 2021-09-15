#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#ifdef VALIDATE
#extension GL_EXT_debug_printf : enable
#endif

#define SHADER_CONST const
#include "rlpbr_core/device.h"
#undef SHADER_CONST

#include "shader_common.h"
#include "vulkan/shaders/utils.glsl"
#include "vulkan/shaders/inputs.glsl"

layout (push_constant, scalar) uniform PushConstant {
    DrawPushConst push_const;
};

layout (set = 0, binding = 2) uniform sampler repeatSampler;
layout (set = 0, binding = 3) uniform sampler clampSampler;

#if 0
layout (set = 0, binding = 3) uniform texture2D msDiffuseAverageTexture; 
layout (set = 0, binding = 4) uniform texture3D msDiffuseDirectionalTexture;
layout (set = 0, binding = 5) uniform texture1D msGGXAverageTexture;
layout (set = 0, binding = 6) uniform texture2D msGGXDirectionalTexture;
layout (set = 0, binding = 7) uniform texture2D msGGXInverseTexture;
#endif

layout (set = 1, binding = 1) uniform texture2D textures[];

layout (set = 1, binding = 2) readonly buffer MatParams {
    PackedMaterial matParams[];
};

layout (location = 0) in InInterface {
    vec3 cameraSpacePosition;
    vec3 normal;
    vec4 tangentAndSign;
    vec2 uv;
    flat uint materialIndex;
} iface;

layout (location = 0) out vec4 out_color;

vec4 fetchSceneTexture(uint32_t idx, vec2 uv, float)
{
    return texture(sampler2D(textures[idx], repeatSampler), uv);
}

#define MatRef uint32_t

u32vec4 fetchSceneMaterialParams(MatRef ref, uint32_t idx, uint32_t sub_idx)
{
    return matParams[idx].data[sub_idx];
}

#include "vulkan/shaders/materials.glsl"

void main()
{
    MaterialParams mat_params = unpackMaterialParams(0, iface.materialIndex);
    uint32_t base_texture_idx = 
        1 + iface.materialIndex * TextureConstantsTexturesPerMaterial;
    Material mat = processMaterial(mat_params, base_texture_idx,
        iface.uv, 0.f);

    out_color = vec4(mat.rho, 1.f);
}
