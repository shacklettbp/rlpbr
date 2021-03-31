#include "habitat_json.hpp"
#include "gltf.hpp"

#include <filesystem>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace RLpbr {
namespace SceneImport {

HabitatJSON::Scene habitatJSONLoad(string_view scene_path_name)
{
    using namespace filesystem;
    using namespace simdjson;
    using namespace HabitatJSON;

    path scene_path(absolute(scene_path_name));
    path root_path = scene_path.parent_path().parent_path();

    path stage_dir = root_path / "stages";

    Scene scene;

    try {
        simdjson::dom::parser scene_parser;
        simdjson::dom::element root = scene_parser.load(scene_path);

        string_view stage_name = root["stage_instance"]["template_name"];
        auto stage_path = stage_dir / stage_name;
        stage_path.concat(".stage_config.json");

        simdjson::dom::parser stage_parser;
        auto stage_root = stage_parser.load(stage_path);
        scene.stagePath = 
            stage_dir / string_view(stage_root["render_asset"]);

        path lighting_path = string_view(root["default_lighting"]);
        lighting_path.concat(".lighting_config.json");

        simdjson::dom::parser light_parser;
        auto light_root = light_parser.load(root_path / lighting_path);

        vector<Light> lights;
        for (auto [idx, light] : dom::object(light_root["lights"])) {
            string_view type_str = light["type"];
            LightType type;
            if (type_str == "point") {
                type = LightType::Point;
            } else if (type_str == "environment") {
                type = LightType::Environment;
            } else {
                cerr << scene_path_name << ": Unknown light type" << endl;
                abort();
            }

            if (type == LightType::Point) {
                uint32_t vec_idx = 0;
                glm::vec3 position;
                for (auto c : light["position"]) {
                    position[vec_idx++] = float(double(c));
                }

                float intensity = double(light["intensity"]);
                if (intensity <= 0.f) {
                    cerr << "Warning: Skipping negative intensity light" << endl;
                    continue;
                }

                vec_idx = 0;
                glm::vec3 color;
                for (auto c : light["color"]) {
                    color[vec_idx++] = float(double(c));
                }

                scene.lights.push_back({
                    type,
                    position,
                    intensity,
                    color,
                });
            } else if (type == LightType::Environment) {
                if (!scene.envMap.empty()) {
                    cerr << "Can only specify one environment map per scene" <<
                        endl;
                    abort();
                }

                scene.envMap = light["path"];
            }
        }

        simdjson::dom::parser inst_parser;
        auto insts = root["object_instances"];
        for (const auto &inst : insts) {
            glm::vec3 translation;
            uint32_t idx = 0;
            for (auto c : inst["translation"]) {
                translation[idx++] = float(double(c));
            }

            glm::quat rotation;
            idx = 0;
            for (auto c : inst["rotation"]) {
                rotation[3 - idx++] = float(double(c));
            }


            auto template_path =
                root_path / string_view(inst["template_name"]);
            template_path.concat(".object_config.json");

            auto inst_root = inst_parser.load(template_path);
            string_view inst_asset = inst_root["render_asset"];

            auto inst_path = template_path.parent_path() / inst_asset;

            scene.additionalInstances.push_back({
                inst_path,
                translation,
                rotation,
                string_view(inst["motion_type"]) == "DYNAMIC",
            });
        }
    } catch (const simdjson_error &e) {
        cerr << "Habitat JSON loading '" << scene_path_name
             << "' failed: " << e.what() << endl;
        abort();
    };

    return scene;
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseHabitatJSON(
    string_view scene_path, const glm::mat4 &base_txfm,
    optional<string_view> texture_dir)
{
    using namespace HabitatJSON;
    using SceneDesc = SceneDescription<VertexType, MaterialType>;

    auto raw_scene = habitatJSONLoad(scene_path);

    SceneDesc desc = parseGLTF<VertexType, MaterialType>(
            raw_scene.stagePath, base_txfm, texture_dir);
    desc.envMap = raw_scene.envMap;

    unordered_map<string, uint32_t> loaded_gltfs;

    for (const Instance &inst : raw_scene.additionalInstances) {
        uint32_t mat_offset = desc.materials.size();
        
        glm::vec3 inst_pos = base_txfm * glm::vec4(inst.pos, 1.f);
        glm::quat inst_rot = glm::quat_cast(base_txfm) * inst.rotation;

        auto [iter, inserted] =
            loaded_gltfs.emplace(inst.gltfPath, desc.defaultInstances.size());

        if (!inserted) {
            auto new_inst = desc.defaultInstances[iter->second];
            new_inst.position = inst_pos;
            new_inst.rotation = inst_rot;
            new_inst.dynamic = inst.dynamic;
            desc.defaultInstances.emplace_back(move(new_inst));
        } else {
            auto inst_desc = parseGLTF<VertexType, MaterialType>(
                inst.gltfPath, glm::mat4(1.f),
                texture_dir);

            bool is_transparent = false;
            for (const auto &child_inst : inst_desc.defaultInstances) {
                if (child_inst.transparent) {
                    is_transparent = true;
                }
            }

            for (const auto &mat : inst_desc.materials) {
                desc.materials.push_back(mat);
            }

            // Merge sub GLTF into single object
            auto [merged_obj, merged_mats] =
                SceneDesc::mergeScene(move(inst_desc), mat_offset);

            desc.objects.emplace_back(move(merged_obj));

            desc.defaultInstances.push_back({
                uint32_t(desc.objects.size() - 1),
                move(merged_mats),
                inst_pos,
                inst_rot,
                glm::vec3(1.f),
                inst.dynamic,
                is_transparent,
            });
        }
    }

    for (const Light &light : raw_scene.lights) {
        desc.defaultLights.push_back({
            light.position,
            light.color * light.intensity,
        });
    }

    return desc;
}

}
}
