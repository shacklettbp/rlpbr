#include "import.hpp"
#include "gltf.hpp"
#include "habitat_json.hpp"

#include <iostream>

#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace RLpbr {
namespace SceneImport {

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

template <typename VertexType, typename MaterialType>
pair<Object<VertexType>, vector<uint32_t>> 
SceneDescription<VertexType, MaterialType>::mergeScene(
    SceneDescription desc, uint32_t mat_offset)
{
    Object<VertexType> merged_obj;
    vector<uint32_t> merged_mats;
    for (const auto &inst : desc.defaultInstances) {
        auto &obj = desc.objects[inst.objectIndex];

        glm::mat4 rot_mat = glm::mat4_cast(inst.rotation);

        glm::mat4 txfm = glm::translate(inst.position) *
            rot_mat * glm::scale(inst.scale);

        glm::mat4 inv_txfm = glm::scale(1.f / inst.scale) *
            glm::transpose(rot_mat) *
            glm::translate(-inst.position);

        for (auto &mesh : obj.meshes) {
            for (auto &vert : mesh.vertices) {
                vert.position = txfm * glm::vec4(vert.position, 1.f);
                vert.normal =
                    glm::transpose(glm::mat3(inv_txfm)) * vert.normal;
                vert.normal = normalize(vert.normal);
            }
            merged_obj.meshes.emplace_back(move(mesh));
        }

        for (uint32_t mat_idx : inst.materials) {
            merged_mats.push_back(mat_idx + mat_offset);
        }
    }

    return {
        move(merged_obj),
        move(merged_mats),
    };
}

template struct SceneDescription<Vertex, Material>;

}
}

#include "gltf.inl"
#include "habitat_json.inl"
