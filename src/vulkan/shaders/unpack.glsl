#ifndef RLPBR_VK_UNPACK_GLSL_INCLUDED
#define RLPBR_VK_UNPACK_GLSL_INCLUDED

#include "inputs.glsl"

// Unpack functions
Camera unpackCamera(PackedCamera packed)
{
    float aspect = float(RES_X) / float(RES_Y);

    vec4 rot = packed.rotation;
    vec3 view = quatRotate(rot, vec3(0.f, 0.f, 1.f));
    vec3 up = quatRotate(rot, vec3(0.f, 1.f, 0.f));
    vec3 right = quatRotate(rot, vec3(1.f, 0.f, 0.f));

    vec4 pos_fov = packed.posAndTanFOV;

    vec3 origin = pos_fov.xyz;

    float right_scale = aspect * pos_fov.w;
    float up_scale = pos_fov.w;

    return Camera(origin, view, up, right, right_scale, up_scale);
}

void unpackEnv(in uint32_t batch_idx,
               out Camera cam,
               out Camera prev_cam,
               out Environment env)
{
    PackedEnv packed = envs[nonuniformEXT(batch_idx)];
    cam = unpackCamera(packed.cam);
    prev_cam = unpackCamera(packed.prevCam);

    u32vec4 data = packed.data;

    // FIXME: data.x is currently instance offset, change to texture offset
    
    env.sceneID = data.x;
    env.baseMaterialOffset = data.y;
    env.baseLightOffset = data.z;
    env.numLights = data.w;
    env.tlasAddr = packed.tlasAddr;

    const uint32_t textures_per_scene = 1 + MAX_MATERIALS *
        TextureConstantsTexturesPerMaterial;
    env.baseTextureOffset = env.sceneID * textures_per_scene;
}

MeshInfo unpackMeshInfo(MeshRef mesh_ref, uint32_t mesh_idx)
{
    MeshInfo mesh_info;
    mesh_info.indexOffset = mesh_ref[nonuniformEXT(mesh_idx)].meshInfo.data.x;

    return mesh_info;
}

Vertex unpackVertex(VertRef vert_ref, uint32_t idx)
{
    PackedVertex packed = vert_ref[nonuniformEXT(idx)].vert;

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

vec3 unpackVertexPosition(VertRef vert_ref, uint32_t idx)
{
    vec4 data = vert_ref[nonuniformEXT(idx)].vert.data[0];

    return data.xyz;
}

uint32_t unpackMaterialID(uint32_t inst_material_idx)
{
    return instanceMaterials[nonuniformEXT(inst_material_idx)];
}

#endif
