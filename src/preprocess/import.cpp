#include "import.hpp"
#include "gltf.hpp"
#include "habitat_json.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace SceneImport {

Material Material::makeMetallicRoughness(
    const string_view base_color_texture,
    const string_view metallic_roughness_texture,
    const glm::vec3 &base_color,
    float base_metallic, float base_roughness)
{
    Material mat {};

    mat.materialModel = MaterialModelType::MetallicRoughness;
    mat.baseColorTexture = base_color_texture;
    mat.metallicRoughnessTexture = metallic_roughness_texture;
    mat.baseColor = base_color;
    mat.baseMetallic = base_metallic;
    mat.baseRoughness = base_roughness;

    return mat;
}

Material Material::makeSpecularGlossiness(
    const string_view diffuse_texture,
    const string_view specular_texture,
    const glm::vec3 &base_diffuse,
    const glm::vec3 &base_specular,
    float base_shininess)
{
    Material mat {};

    mat.materialModel = MaterialModelType::SpecularGlossiness;
    mat.diffuseTexture = diffuse_texture;
    mat.specularTexture = specular_texture;
    mat.baseDiffuse = base_diffuse;
    mat.baseSpecular = base_specular;
    mat.baseShininess = base_shininess;

    return mat;
}


static bool isGLTF(string_view gltf_path)
{
    auto suffix = gltf_path.substr(gltf_path.rfind('.') + 1);
    return suffix == "glb" || suffix == "gltf";
}

static bool isHabitatJSON(string_view habitat_path)
{
    auto suffix = habitat_path.substr(habitat_path.rfind('.') + 1);

    return suffix == "json";
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType>
SceneDescription<VertexType, MaterialType>::parseScene(
    string_view scene_path, const glm::mat4 &base_txfm,
    optional<string_view> texture_dir)
{
    if (isGLTF(scene_path)) {
        return parseGLTF<VertexType, MaterialType>(scene_path,
            base_txfm, texture_dir);
    }

    if (isHabitatJSON(scene_path)) {
        return parseHabitatJSON<VertexType, MaterialType>(scene_path,
            base_txfm, texture_dir);
    }

    cerr << "Unsupported input format" << endl;
    abort();
}

template struct SceneDescription<Vertex, Material>;

}
}

#include "gltf.inl"
#include "habitat_json.inl"
