#include "import.hpp"
#include "gltf.hpp"
#include "habitat_json.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace SceneImport {

Material Material::make(const string_view albedo_name,
    const glm::vec3 &base_albedo, float roughness)
{
    return Material {
        string(albedo_name),
        base_albedo,
        roughness,
    };
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
    string_view scene_path, const glm::mat4 &base_txfm)
{
    if (isGLTF(scene_path)) {
        return parseGLTF<VertexType, MaterialType>(scene_path, base_txfm);
    }

    if (isHabitatJSON(scene_path)) {
        return parseHabitatJSON<VertexType, MaterialType>(scene_path, base_txfm);
    }

    cerr << "Unsupported input format" << endl;
    abort();
}

template struct SceneDescription<Vertex, Material>;

}
}

#include "gltf.inl"
#include "habitat_json.inl"
