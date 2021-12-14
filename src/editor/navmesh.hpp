#pragma once

#include <rlpbr_core/scene.hpp>
#include "renderer.hpp"
#include "utils.hpp"

#include <optional>

namespace RLpbr {
namespace editor {

struct NavmeshConfig {
    AABB bbox;
    float cellSize = 0.05f;
    float cellHeight = 0.05f;
    float agentHeight = 1.f;
    float agentRadius = 0.1f;
    float maxSlope = 20.f;
    float agentMaxClimb = 0.1f;
    float maxEdgeLen = 0.f;
    float maxError = 2.5f;
    float regionMinSize = 20.f;
    float regionMergeSize = 20.f;
    float detailSampleDist = 5.f;
    float detailSampleMaxError = 2.f;
};

struct NavmeshRenderData {
    std::vector<OverlayVertex> vertices;
    std::vector<uint32_t> triIndices;
    std::vector<uint32_t> boundaryLines;
    std::vector<uint32_t> internalLines;
};

struct NavmeshInternal;

struct NavmeshDeleter {
    void operator()(NavmeshInternal *ptr) const;
};

struct Navmesh {
    std::unique_ptr<NavmeshInternal, NavmeshDeleter> internal;
    AABB bbox;
    NavmeshRenderData renderData;

    uint32_t findPath(const glm::vec3 &start, const glm::vec3 &end,
                      uint32_t max_verts, glm::vec3 *verts,
                      void *scratch) const;

    static uint32_t scratchBytesPerTri();

    glm::vec3 getRandomPoint();
};

std::optional<Navmesh> buildNavmesh(const NavmeshConfig &cfg,
                     uint32_t total_triangles,
                     const PackedVertex *vertices,
                     const uint32_t *indices,
                     const std::vector<ObjectInfo> &objects,
                     const std::vector<MeshInfo> &meshes,
                     const std::vector<ObjectInstance> &instances,
                     const std::vector<InstanceTransform> &transforms,
                     const char **err_msg);

std::optional<Navmesh> loadNavmesh(const char *file_path,
                                   const char **err_msg);
void saveNavmesh(const char *file_path, const Navmesh &navmesh);

}
}
