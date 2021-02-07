#include "habitat_json.hpp"

#include <filesystem>

using namespace std;

namespace RLpbr {
namespace SceneImport {

HabitatJSONScene habitatJSONLoad(string_view scene_path)
{
    (void)scene_path;

    return HabitatJSONScene {
    };
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
