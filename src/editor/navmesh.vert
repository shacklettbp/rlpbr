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

layout (set = 0, binding = 0, scalar) readonly buffer Vertices {
    vec4 vertices[];
};

layout (push_constant, scalar) uniform PushConstant {
    NavmeshPushConst draw_const;
};

layout (location = 0) out OutInterface {
    vec4 color;
} iface;

void main()
{
    vec4 vert = vertices[gl_VertexIndex];

    vec4 camera_space = draw_const.base.view * vec4(vert.xyz, 1.f);

    gl_Position = draw_const.base.proj * camera_space;

    iface.color = unpackUnorm4x8(floatBitsToUint(vert.w));
}
