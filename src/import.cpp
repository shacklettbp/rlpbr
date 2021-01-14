#include "import.hpp"
#include "scene.hpp"
#include "gltf.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace SceneImport {

static bool isGLTF(string_view gltf_path)
{
    auto suffix = gltf_path.substr(gltf_path.rfind('.') + 1);
    return suffix == "glb" || suffix == "gltf";
}

Material Material::make(const std::string_view albedo_name)
{
    return Material {
        string(albedo_name),
        {},
    };
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType>
SceneDescription<VertexType, MaterialType>::parseScene(
    string_view scene_path, const glm::mat4 &base_txfm)
{
    if (isGLTF(scene_path)) {
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

        return SceneDescription {
            move(geometry),
            move(materials),
            move(instances),
            {},
        };
    }

    cerr << "Currently only GLTF import is supported" << endl;
    abort();
}

template struct SceneDescription<Vertex, Material>;

}
}
