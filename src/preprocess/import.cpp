#include "import.hpp"
#include "gltf.hpp"
#include "habitat_json.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace SceneImport {

Material Material::make(const std::string_view albedo_name)
{
    return Material {
        string(albedo_name),
        {},
    };
}

static bool isGLTF(string_view gltf_path)
{
    auto suffix = gltf_path.substr(gltf_path.rfind('.') + 1);
    return suffix == "glb" || suffix == "gltf";
}

template <typename VertexType, typename MaterialType>
static SceneDescription<VertexType, MaterialType> parseGLTF(
    string_view scene_path, const glm::mat4 &base_txfm)
{
    auto raw_scene = gltfLoad(scene_path);

    vector<MaterialType> materials =
        gltfParseMaterials<MaterialType>(raw_scene);

    vector<Mesh<VertexType>> geometry;

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene.meshes.size();
         mesh_idx++) {
        auto [vertices, indices] =
            gltfParseMesh<VertexType>(raw_scene, mesh_idx);

        geometry.push_back({
            move(vertices),
            move(indices),
        });
    }

    vector<InstanceProperties> instances =
        gltfParseInstances(raw_scene, base_txfm);

    return SceneDescription<VertexType, MaterialType> {
        move(geometry),
        move(materials),
        move(instances),
        {},
    };
}

static bool isHabitatJSON(string_view habitat_path)
{
    auto suffix = habitat_path.substr(habitat_path.rfind('.') + 1);

    return suffix == "json";
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseHabitatJSON(
    string_view scene_path, const glm::mat4 &)
{
    auto raw_scene = habitatJSONLoad(scene_path);
    (void)raw_scene;

    return SceneDescription<VertexType, MaterialType> {
        {},
        {},
        {},
        {},
    };
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
