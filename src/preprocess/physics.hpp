#pragma once

#include <rlpbr_core/physics.hpp>

#include <optional>
#include <string_view>
#include <vector>

#include "preprocess.hpp"

namespace RLpbr {

struct SDF {
    glm::u32vec3 numCells;
    std::vector<float> grid;
    glm::vec3 edgeOffset;
    float derivativeOffset;

    void dump(const std::string_view dump_path) const;
};

struct PhysicsMeshProperties {
    glm::vec3 interia;
    glm::vec3 com;
    float mass;
};

struct PhysicsMeshInfo {
    template <typename VertexType>
    static PhysicsMeshInfo make(const VertexType *vertices,
                                const uint32_t *indices,
                                uint32_t num_indices,
                                bool skip_sdf);

    PhysicsMeshProperties meshProps;
    AABB bbox;
    SDF sdf;
};

struct ProcessedPhysicsState {
    std::vector<SDF> sdfs;
    std::vector<PhysicsObject> objects;

    template <typename VertexType>
    static ProcessedPhysicsState make(
        const ProcessedGeometry<VertexType> &geometry, bool skip_sdfs);
};


}
