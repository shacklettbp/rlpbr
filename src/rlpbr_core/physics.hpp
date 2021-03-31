#pragma once

#include "utils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vector>

namespace RLpbr {

struct SDFConfig {
    static constexpr glm::vec3 sdfSampleResolution {0.01f, 0.01f, 0.01f};
};

struct AABB {
    glm::vec3 pMin;
    glm::vec3 pMax;
};

struct SDFBoundingBox {
    AABB aabb;
    // Offset from the edge of the bounding box to the start of
    // of the volume texture
    glm::vec3 edgeOffset; 
    // Minimum (over all directions) width of 
    // texel in normalized texture coords,
    float derivativeOffset; 
};

struct alignas(16) PhysicsObject {
    SDFBoundingBox bounds;
    uint32_t sdfID;
    glm::vec3 interia;
    glm::vec3 com;
    float mass;
    uint32_t indexOffset;
    uint32_t numTriangles;
};

struct PhysicsInstance {
    uint32_t instanceID;
    uint32_t objectID;
};

struct alignas(16) PhysicsTransform {
    glm::vec3 position;
    glm::quat rotation;
};

struct PhysicsMetadata {
    std::vector<std::string> sdfPaths;
    DynArray<PhysicsInstance> staticInstances;
    DynArray<PhysicsInstance> dynamicInstances;
    DynArray<PhysicsTransform> dynamicTransforms;
};

}
