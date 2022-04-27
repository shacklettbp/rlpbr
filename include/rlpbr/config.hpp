#pragma once

#include <glm/glm.hpp>

namespace RLpbr {

enum class BackendSelect : uint32_t {
    Optix,
    Vulkan,
};

enum class RenderMode : uint32_t {
    PathTracer,
    Biased,
};

enum class RenderFlags : uint32_t {
    AuxiliaryOutputs = 1 << 0,
    ForceUniform = 1 << 1,
    Tonemap = 1 << 2,
    EnablePhysics = 1 << 4,
    Randomize = 1 << 5,
    AdaptiveSample = 1 << 6,
    Denoise = 1 << 7,
    RandomizeMaterials = 1 << 8,
};

struct RenderConfig {
    int gpuID;
    uint32_t numLoaders;
    uint32_t batchSize;
    uint32_t imgWidth;
    uint32_t imgHeight;
    uint32_t spp;
    uint32_t maxDepth;
    uint32_t maxTextureResolution;
    RenderMode mode;
    RenderFlags flags;
    float clampThreshold;
    BackendSelect backend;
};

inline RenderFlags & operator|=(RenderFlags &a, RenderFlags b)
{
    a = RenderFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline bool operator&(RenderFlags a, RenderFlags b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}

inline RenderFlags operator|(RenderFlags a, RenderFlags b)
{
    a |= b;

    return a;
}

}
