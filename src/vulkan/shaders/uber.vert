#version 450
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout (location = 0) out OutInterface {
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

layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (set = 0, binding = 1, scalar) readonly buffer TransformInfos {
    mat4x3 transforms[];
};

#ifdef MATERIALS

layout (set = 0, binding = 2) readonly buffer MatIndices {
    uint materialIndices[];
};

#endif

layout (push_constant, scalar) uniform PushConstant {
    DrawPushConstant draw_const;
};

layout (set = 1, binding = 0, scalar) readonly buffer Vertices {
    Vertex vertices[];
};

void main() 
{
    Vertex v = vertices[gl_VertexIndex];
    vec4 object_space = vec4(v.px, v.py, v.pz, 1.f);

    mat4 view = view_info[draw_const.batchIdx].view;

    mat4x3 raw_txfm = transforms[gl_InstanceIndex];
    mat4 model = mat4(raw_txfm[0], 0.f,
                      raw_txfm[1], 0.f,
                      raw_txfm[2], 0.f,
                      raw_txfm[3], 1.f);

    mat4 mv = view * model;

    vec4 camera_space = mv * object_space;

    gl_Position = view_info[draw_const.batchIdx].projection * camera_space;

#ifdef LIGHTING
    mat3 normal_mat = mat3(mv);
    vec3 normal_scale = vec3(1.f / dot(normal_mat[0], normal_mat[0]),
                             1.f / dot(normal_mat[1], normal_mat[1]),
                             1.f / dot(normal_mat[2], normal_mat[2]));
    vec3 object_normal = vec3(v.nx, v.ny, v.nz);

    iface.normal = normal_mat * object_normal * normal_scale;
    iface.cameraSpacePosition = camera_space.xyz;
#endif

#ifdef MATERIALS
    iface.uv = vec2(v.ux, v.uy);
    iface.materialIndex = materialIndices[gl_InstanceIndex];
#endif

#ifdef OUTPUT_DEPTH
    iface.linearDepth = gl_Position.w;
#endif
}
