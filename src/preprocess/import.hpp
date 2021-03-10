#pragma once

#include <rlpbr_core/scene.hpp>

#include <vector>

namespace RLpbr {
namespace SceneImport {

template <typename VertexType>
struct Mesh {
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;
};

struct Material {
    MaterialModelType materialModel;

    std::string baseColorTexture;
    std::string metallicRoughnessTexture;
    glm::vec3 baseColor;
    float baseMetallic;
    float baseRoughness;

    std::string diffuseTexture;
    std::string specularTexture;
    glm::vec3 baseDiffuse;
    glm::vec3 baseSpecular;
    float baseShininess;

    static Material makeMetallicRoughness(
        const std::string_view base_color_texture,
        const std::string_view metallic_roughness_texture,
        const glm::vec3 &base_color,
        float base_metallic, float base_roughness);

    static Material makeSpecularGlossiness(
        const std::string_view diffuse_texture,
        const std::string_view specular_texture,
        const glm::vec3 &base_diffuse,
        const glm::vec3 &base_specular,
        float base_shininess);
};

template <typename VertexType, typename MaterialType>
struct SceneDescription {
    std::vector<Mesh<VertexType>> meshes;
    std::vector<MaterialType> materials;

    std::vector<InstanceProperties> defaultInstances;
    std::vector<LightProperties> defaultLights;

    static SceneDescription parseScene(std::string_view scene_path,
        const glm::mat4 &base_txfm,
        std::optional<std::string_view> texture_dir);
};

}
}
