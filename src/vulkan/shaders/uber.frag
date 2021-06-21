#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout (location = 0) in InInterface {
#ifdef LIGHTING
    vec3 normal;
    vec3 cameraSpacePosition;
#endif

#ifdef MATERIALS
    vec2 uv;
    flat uint materialIndex;
#endif

#ifdef OUTPUT_DEPTH
    float linearDepth;
#endif
} iface;

#ifdef OUTPUT_COLOR
layout (location = COLOR_ATTACHMENT) out vec4 out_color;
#endif

#ifdef OUTPUT_DEPTH
layout (location = DEPTH_ATTACHMENT) out float out_depth;
#endif

#ifdef LIGHTING
#include "brdf.glsl"

layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (push_constant, scalar) uniform PushConstant {
    DrawPushConstant draw_const;
};

layout (set = 0, binding = 1, scalar) uniform LightingInfo {
    LightProperties lights[MAX_LIGHTS];
    uint numLights;
} lighting_info;

#endif

#ifdef MATERIALS

layout (set = 1, binding = 1) uniform sampler texture_sampler;
layout (set = 1, binding = 2) uniform texture2D textures[];

layout (set = 1, binding = 3, scalar) readonly buffer Params {
    MaterialParams material_params[];
};

#endif

#ifdef OUTPUT_COLOR

#ifdef LIGHTING 

vec4 compute_color()
{
    MaterialParams params = material_params[iface.materialIndex];

    vec3 diffuse = params.baseAlbedo;

    if (params.texIdxs.x != -1) {
        diffuse *= texture(sampler2D(textures[params.texIdxs.x],
                                     texture_sampler), iface.uv, 0.f).xyz;
    }

    diffuse *= params.baseAlbedo;

    vec3 specular = vec3(1.f, 1.f, 1.f);

    float shininess = 2.f / (pow(params.roughness, 4) + 1e-3f) - 2;

    vec3 Lo = vec3(0.0);
    for (int light_idx = 0; light_idx < lighting_info.numLights; light_idx++) {
        vec3 world_light_position =
            lighting_info.lights[light_idx].position.xyz;
        vec3 light_position =
                (view_info[draw_const.batchIdx].view *
                    vec4(world_light_position, 1.f)).xyz;
        vec3 light_color = lighting_info.lights[light_idx].color.xyz;
        BRDFParams brdf_params = makeBRDFParams(light_position,
                                                iface.cameraSpacePosition,
                                                iface.normal, light_color);

        Lo += blinnPhong(brdf_params, shininess, diffuse, specular);
    }

    return vec4(Lo, 1.f);
}

#else

vec4 compute_color()
{
    MaterialParams params = material_params[iface.materialIndex];

    return texture(sampler2D(textures[params.texIdxs.x],
                             texture_sampler), iface.uv, 0.f);
}

#endif

#endif

void main() 
{
#ifdef OUTPUT_COLOR
    out_color = compute_color();
#endif

#ifdef OUTPUT_DEPTH
    out_depth = iface.linearDepth;
#endif
}
