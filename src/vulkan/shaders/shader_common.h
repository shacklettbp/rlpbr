#ifndef RLPBR_VK_SHADER_COMMON_H_INCLUDED
#define RLPBR_VK_SHADER_COMMON_H_INCLUDED

struct PackedCamera {
    vec4 data[3];
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
    vec4 data[4];
};

struct PackedInstance {
    uint32_t materialOffset;
    uint32_t meshOffset;
};

struct Reservoir {
    vec4 val;
};

struct ReGIRCell {
    Reservoir reservoirs[64];
};

#define MAX_MATERIALS (5000)
#define MAX_LIGHTS (100000)
#define WORKGROUP_SIZE (32)
#define LOCAL_WORKGROUP_X (8)
#define LOCAL_WORKGROUP_Y (4)
#define LOCAL_WORKGROUP_Z (1)

#endif
