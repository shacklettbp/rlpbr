#pragma once

#include <cstdint>

namespace RLpbr {
namespace optix {

struct Config {
    static constexpr uint32_t maxInstances = 4'000'000;
    static constexpr uint32_t maxInstanceMaterials = 8'000'000;
    static constexpr uint32_t maxLights = 1'000;
};

}
}
