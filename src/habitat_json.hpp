#pragma once

#include "scene.hpp"

#include <glm/glm.hpp>
#include <simdjson.h>

namespace RLpbr {
namespace SceneImport {

struct HabitatJSONScene {

};

HabitatJSONScene habitatJSONLoad(std::string_view scene_path);

}
}