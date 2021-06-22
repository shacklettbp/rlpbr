#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "core.hpp"
#include "utils.hpp"

#include <vector>

namespace RLpbr {
namespace vk {

class PresentationState {
public:
    static void init();
    static std::vector<const char *> getInstanceExtensions();
    static VkSurfaceFormatKHR selectSwapchainFormat(const InstanceState &inst,
                                                    VkPhysicalDevice phy,
                                                    VkSurfaceKHR surface);

    static VkBool32 deviceSupportCallback(VkInstance inst,
                                          VkPhysicalDevice phy,
                                          uint32_t idx);

    PresentationState(const InstanceState &inst,
                      const DeviceState &dev,
                      const QueueState &present_queue,
                      uint32_t qf_idx,
                      uint32_t num_frames_inflight);

    uint32_t acquireNext(const DeviceState &dev);

    VkSemaphore getRenderSemaphore();

    void present(const DeviceState &dev, uint32_t swapchain_idx,
                 const QueueState &present_queue);

private:
    GLFWwindow *window_;
    VkSurfaceKHR surface_;
    VkSwapchainKHR swapchain_;
    DynArray<VkImage> swapchain_imgs_;
    DynArray<std::pair<VkSemaphore, VkSemaphore>> present_semas_;

    uint32_t cur_sema_;
};

}
}
