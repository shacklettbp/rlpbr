#ifndef RLPBR_VK_SHADER_COMMON_H_INCLUDED
#define RLPBR_VK_SHADER_COMMON_H_INCLUDED

struct PackedCamera {
    vec4 data[3];
};

struct PackedEnv {
    PackedCamera cam;
    u32vec4 data;
    uint64_t tlasAddr;
    uint64_t pad;
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

#define MAX_MATERIALS (1000)
#define MAX_LIGHTS (100000)
#define WORKGROUP_SIZE (32)

#endif
