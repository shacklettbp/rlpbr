#include "dispatch.hpp"
#include <iostream>
#include <cstdlib>

extern "C" {
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance,
                                                               const char *);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice,
                                                             const char *);
}

namespace RLpbr {
namespace vk {

static inline PFN_vkVoidFunction checkPtr(PFN_vkVoidFunction ptr,
                                          const std::string &name)
{
    if (!ptr) {
        std::cerr << name << " failed to load" << std::endl;
        exit(EXIT_FAILURE);
    }

    return ptr;
}

InstanceDispatch::InstanceDispatch(VkInstance ctx, bool need_present)
#include "dispatch_instance_impl.cpp"
{}

DeviceDispatch::DeviceDispatch(VkDevice ctx, bool need_present, bool need_rt)
#include "dispatch_device_impl.cpp"
{}

}
}
