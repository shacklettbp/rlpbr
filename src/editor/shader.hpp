#pragma once

#include <glm/glm.hpp>

namespace RLpbr {
namespace editor {

namespace Shader {
using namespace glm;
using uint = uint32_t;

#include "shader_common.h"
};

using Shader::DrawPushConst;
using Shader::NavmeshPushConst;

}
}
