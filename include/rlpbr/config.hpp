#pragma once

#include <glm/glm.hpp>

namespace RLpbr {

enum class BackendSelect {
    Optix,
    Vulkan,
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
    bool enablePhysics;
    bool pathTracer;
    bool auxiliaryOutputs;
    float clampThreshold;
    BackendSelect backend;
};

}
