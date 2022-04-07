#ifndef RLPBR_VK_SHADER_COMMON_H_INCLUDED
#define RLPBR_VK_SHADER_COMMON_H_INCLUDED

#include "comp_definitions.h"

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
    vec4 envMapRotation;
    vec4 lightFilterAndEnvIdx;
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

struct InputTile {
    uint32_t xOffset;
    uint32_t yOffset;
    uint32_t batchIdx;
    uint32_t sampleOffset;
};

struct AdaptiveTile {
    float tileMean;
    float tileVarianceM2;
    uint32_t numSamples;
    uint32_t spinLock;
};

struct ReGIRCell {
    Reservoir reservoirs[64];
};

#define MAX_MATERIALS (2048)
#define MAX_LIGHTS (1000000)
#define MAX_SCENES (16)
#define MAX_ENV_MAPS (32)
#define MAX_TILES (524288)
#define ADAPTIVE_SAMPLES_PER_THREAD (4)

#endif
