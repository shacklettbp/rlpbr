#pragma once

#include <cstdint>

namespace RLpbr {
namespace optix {

struct Config {
    static constexpr uint32_t maxInstances = 1'000'000;
    static constexpr uint32_t maxInstanceMaterials = 2'000'000;
    static constexpr uint32_t maxLights = 100'000;
};

}
}
