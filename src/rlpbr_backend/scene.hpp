#pragma once

#include "common.hpp"

#include <glm/glm.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string_view>
#include <variant>
#include <vector>

namespace RLpbr {

struct LoaderBackend;

struct InstanceProperties {
    uint32_t meshIndex;
    uint32_t materialIndex;
    glm::mat4x3 txfm;
};

struct LightProperties {
    glm::vec3 position;
    glm::vec3 color;
};

struct EnvironmentInit {
    EnvironmentInit(const std::vector<InstanceProperties> &instances,
                    const std::vector<LightProperties> &lights,
                    uint32_t num_meshes);

    std::vector<std::vector<glm::mat4x3>> transforms;
    std::vector<std::vector<uint32_t>> materials;
    std::vector<std::pair<uint32_t, uint32_t>> indexMap;
    std::vector<std::vector<uint32_t>> reverseIDMap;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct MeshInfo {
    uint32_t indexOffset;
    uint32_t numTriangles;
    uint32_t numVertices;
};

struct MaterialMetadata {
    std::vector<std::string> albedoTextures;
    std::vector<uint32_t> albedoIndices;
};

struct StagingHeader {
    uint32_t numVertices;
    uint32_t numIndices;
    uint64_t indexOffset;
    uint64_t materialOffset;
    uint64_t materialBytes;
    
    uint64_t totalBytes;
    uint32_t numMeshes;
};

struct SceneLoadData {
    StagingHeader hdr;
    std::vector<MeshInfo> meshInfo;
    MaterialMetadata materialInfo;
    EnvironmentInit envInit;

    std::variant<std::ifstream, std::vector<char>> data;

    static SceneLoadData loadFromDisk(std::string_view scene_path);
};

struct Scene {
    std::vector<MeshInfo> meshInfo;
    EnvironmentInit envInit;
};

}
