#pragma once

#include "import.hpp"

#include <rlpbr_core/scene.hpp>

#include <glm/glm.hpp>
#include <simdjson.h>

namespace RLpbr {
namespace SceneImport {

namespace HabitatJSON {

enum class LightType {
    Point,
};

struct Light {
    LightType type;
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
};

struct Instance {
    std::filesystem::path gltfPath;
    bool dynamic;
    glm::mat4x3 transform;
};

struct Scene {
    std::filesystem::path stagePath;
    std::vector<Instance> additionalInstances;
    std::vector<Light> lights;
};

}

HabitatJSON::Scene habitatJSONLoad(std::string_view scene_path);

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseHabitatJSON(
    std::string_view scene_path, const glm::mat4 &base_txfm,
    std::optional<std::string_view> texture_dir);

}
}
