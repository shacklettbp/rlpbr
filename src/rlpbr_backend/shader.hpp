#pragma once

#include <glm/glm.hpp>

namespace RLpbr {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct MaterialParams {
    glm::vec3 baseAlbedo;
    float roughness;
    glm::uvec4 texIdxs;
};

}
