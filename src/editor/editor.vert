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

struct PackedVertex {
    vec4 data[2];
};

struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangentAndSign;
    vec2 uv;
};

struct InstanceTransform {
    mat4x3 o2w;
    mat4x3 w2o;
};

layout (set = 0, binding = 0, scalar) readonly buffer TransformInfos {
    InstanceTransform transforms[];
};

layout (set = 0, binding = 1, scalar) readonly buffer MaterialIndices {
    uint32_t matIndices[];
};

layout (set = 1, binding = 0) readonly buffer Vertices {
    PackedVertex vertices[];
};

layout (push_constant, scalar) uniform PushConstant {
    DrawPushConst draw_const;
};

layout (location = 0) out OutInterface {
    vec3 cameraSpacePosition;
    vec3 normal;
    vec4 tangentAndSign;
    vec2 uv;
    flat uint materialIndex;
} iface;

Vertex unpackVertex(uint32_t idx)
{
    PackedVertex packed = vertices[idx];

    vec4 a = packed.data[0];
    vec4 b = packed.data[1];

    u32vec3 packed_normal_tangent = u32vec3(
        floatBitsToUint(a.w), floatBitsToUint(b.x), floatBitsToUint(b.y));

    vec3 normal;
    vec4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = vec3(a.x, a.y, a.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = vec2(b.z, b.w);

    return vert;
}

void main()
{
    Vertex v = unpackVertex(gl_VertexIndex);
    vec4 object_space = vec4(v.position, 1.f);

    InstanceTransform txfm = transforms[gl_InstanceIndex];

    mat4 model_mat = mat4(txfm.o2w[0], 0.f,
                          txfm.o2w[1], 0.f,
                          txfm.o2w[2], 0.f,
                          txfm.o2w[3], 1.f);

    mat4 normal_mat = mat4(txfm.w2o[0], 0.f,
                           txfm.w2o[1], 0.f,
                           txfm.w2o[2], 0.f,
                           txfm.w2o[3], 1.f);

    mat4 mv = draw_const.view * model_mat;

    vec4 camera_space = mv * object_space;

    gl_Position = draw_const.proj * camera_space;

    iface.cameraSpacePosition = camera_space.xyz;
    iface.normal = v.normal;
    iface.tangentAndSign = v.tangentAndSign;
    iface.uv = v.uv;
    iface.materialIndex = matIndices[gl_InstanceIndex];
}
