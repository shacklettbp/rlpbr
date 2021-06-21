#pragma once

#include <vulkan/vulkan.h>

namespace RLpbr {
namespace vk {

struct InstanceDispatch {
#include "dispatch_instance_impl.hpp"

    InstanceDispatch(VkInstance inst, bool need_present);
};

struct DeviceDispatch {
#include "dispatch_device_impl.hpp"

    DeviceDispatch(VkDevice dev, bool need_present, bool need_rt);
};

}
}
