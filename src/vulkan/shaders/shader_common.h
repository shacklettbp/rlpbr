#ifndef RLPBR_VK_SHADER_COMMON_H_INCLUDED
#define RLPBR_VK_SHADER_COMMON_H_INCLUDED

struct SceneAddresses {
    VertRef vertAddr;
    IdxRef idxAddr;
    MatRef matAddr;
    MeshRef meshAddr;
};

struct PackedCamera {
    vec4 rotation;
    vec4 posAndTanFOV;
};

struct PackedEnv {
    PackedCamera cam;
    PackedCamera prevCam;
    u32vec4 data;
    uint64_t tlasAddr;
    uint64_t reservoirGridAddr;
};

struct RTPushConstant {
    uint baseFrameCounter;
};

struct PackedLight {
    vec4 data;
};

struct PackedInstance {
    uint32_t materialOffset;
    uint32_t meshOffset;
};

struct Reservoir {
    vec3 y;
    float pHat;
    float wSum;
    float W;
    uint32_t M;
    uint32_t pad;
};

struct ReGIRCell {
    Reservoir reservoirs[64];
};

#define MAX_MATERIALS (2048)
#define MAX_LIGHTS (1000000)
#define MAX_SCENES (16)
#define WORKGROUP_SIZE (32)
#define LOCAL_WORKGROUP_X (8)
#define LOCAL_WORKGROUP_Y (4)
#define LOCAL_WORKGROUP_Z (1)

#endif
