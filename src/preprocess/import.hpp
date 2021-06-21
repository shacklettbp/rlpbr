#pragma once

#include <rlpbr_core/scene.hpp>

#include <optional>
#include <vector>

namespace RLpbr {
namespace SceneImport {

template <typename VertexType>
struct Mesh {
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;
};

template <typename VertexType>
struct Object {
    std::string name;
    std::vector<Mesh<VertexType>> meshes;
};

struct Material {
    std::string name;
    std::string baseColorTexture;
    std::string metallicRoughnessTexture;
    std::string specularTexture;
    std::string normalMapTexture;
    std::string emittanceTexture;
    std::string transmissionTexture;
    std::string clearcoatTexture;
    std::string anisoTexture;
    glm::vec3 baseColor;
    float baseTransmission;
    glm::vec3 baseSpecular;
    float specularScale;
    float baseMetallic;
    float baseRoughness;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    glm::vec3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    float anisoRotation;
    glm::vec3 baseEmittance;
    bool thinwalled;
};

struct InstanceProperties {
    std::string name;
    uint32_t objectIndex;
    std::vector<uint32_t> materials;
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;
    bool dynamic;
    bool transparent;
};

template <typename VertexType, typename MaterialType>
struct SceneDescription {
    std::vector<Object<VertexType>> objects;
    std::vector<MaterialType> materials;

    std::vector<InstanceProperties> defaultInstances;
    std::vector<LightProperties> defaultLights;

    std::string envMap;

    static SceneDescription parseScene(std::string_view scene_path,
        const glm::mat4 &base_txfm,
        std::optional<std::string_view> texture_dir);

    static std::pair<Object<VertexType>, std::vector<uint32_t>> mergeScene(
        SceneDescription desc, uint32_t mat_offset=0);
};

}
}
