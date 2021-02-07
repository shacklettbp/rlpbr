#pragma once

#include "import.hpp"

#include <rlpbr_backend/scene.hpp>

#include <glm/glm.hpp>
#include <simdjson.h>

namespace RLpbr {
namespace SceneImport {

struct HabitatJSONScene {

};

HabitatJSONScene habitatJSONLoad(std::string_view scene_path);

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseHabitatJSON(
    std::string_view scene_path, const glm::mat4 &);

}
}
