#pragma once

#include <cstdint>

namespace RLpbr {
namespace optix {

struct Config {
    static constexpr uint32_t maxInstances = 10'000;
    static constexpr uint32_t maxLights = 1'000;
};

}
}
