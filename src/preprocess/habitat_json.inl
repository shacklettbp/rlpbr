#include "habitat_json.hpp"
#include "gltf.hpp"

#include <filesystem>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>

using namespace std;

namespace RLpbr {
namespace SceneImport {

HabitatJSON::Scene habitatJSONLoad(string_view scene_path_name)
{
    using namespace filesystem;
    using namespace simdjson;
    using namespace HabitatJSON;

    path scene_path(scene_path_name);
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
            } else {
                cerr << scene_path_name << ": Unknown light type" << endl;
                abort();
            }

            uint32_t vec_idx = 0;
            glm::vec3 position;
            for (auto c : light["position"]) {
                position[vec_idx++] = float(double(c));
            }

            float intensity = double(light["intensity"]);

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
                rotation[idx++] = float(double(c));
            }

            glm::mat4 txfm =
                glm::translate(translation) * glm::mat4_cast(rotation);

            auto template_path =
                root_path / string_view(inst["template_name"]);
            template_path.concat(".object_config.json");

            auto inst_root = inst_parser.load(template_path);
            string_view inst_asset = inst_root["render_asset"];

            auto inst_path = template_path.parent_path() / inst_asset;

            scene.additionalInstances.push_back({
                inst_path,
                string_view(inst["motion_type"]) == "DYNAMIC",
                glm::mat4x3(txfm),
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

}
}
