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

struct alignas(16) Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

enum class MaterialModelType : uint32_t {
    MetallicRoughness,
    SpecularGlossiness,
};

struct alignas(16) MetallicRoughnessParams {
    glm::vec3 baseColor;
    float baseMetallic;
    float baseRoughness;
    uint32_t colorIdx;
    uint32_t roughnessMetallicIdx;
};

struct alignas(16) SpecularGlossinessParams {
    glm::vec4 baseDiffuseAndIndices;
    glm::vec4 baseSpecularGlossiness;
};

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

struct TextureInfo {
    std::string textureDir;
    std::vector<std::string> diffuse;
    std::vector<std::string> specular;
};

struct MaterialMetadata {
    TextureInfo textureInfo;
    std::vector<MetallicRoughnessParams> metallicRoughness;
    std::vector<SpecularGlossinessParams> specularGlossiness;
};

struct StagingHeader {
    uint32_t numMeshes;
    uint32_t numVertices;
    uint32_t numIndices;
    uint32_t numMaterials;

    uint64_t indexOffset;
    uint64_t materialOffset;
    
    uint64_t totalBytes;

    uint32_t materialModel; 
};

struct SceneLoadData {
    StagingHeader hdr;
    std::vector<MeshInfo> meshInfo;
    TextureInfo textureInfo;
    EnvironmentInit envInit;

    std::variant<std::ifstream, std::vector<char>> data;

    static SceneLoadData loadFromDisk(std::string_view scene_path);
};

struct Scene {
    std::vector<MeshInfo> meshInfo;
    EnvironmentInit envInit;
};

}
