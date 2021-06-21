#ifndef BPS3D_VK_MESH_COMMON_H_INCLUDED
#define BPS3D_VK_MESH_COMMON_H_INCLUDED

struct DrawInput {
    uint instanceID;
    uint chunkID;
};

struct FrustumBounds {
    vec4 sides;
    vec2 nearFar;
};

struct CullPushConstant {
    FrustumBounds frustumBounds;
    uint batchIdx;
    uint baseDrawID;
    uint numDrawCommands;
};

struct MeshCullInfo {
    uint chunkOffset;
    uint numChunks;
};

#endif
