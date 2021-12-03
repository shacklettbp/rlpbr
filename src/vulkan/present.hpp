#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "core.hpp"
#include "utils.hpp"

#include <vector>

namespace RLpbr {
namespace vk {

struct Swapchain {
    VkSwapchainKHR hdl;
    glm::u32vec2 dims;
};

class PresentationState {
public:
    static PFN_vkGetInstanceProcAddr init();
    static std::vector<const char *> getInstanceExtensions();
    static VkSurfaceFormatKHR selectSwapchainFormat(const InstanceState &inst,
                                                    VkPhysicalDevice phy,
                                                    VkSurfaceKHR surface);

    static VkBool32 deviceSupportCallback(VkInstance inst,
                                          VkPhysicalDevice phy,
                                          uint32_t idx);

    PresentationState(const InstanceState &inst,
                      const DeviceState &dev,
                      uint32_t qf_idx,
                      uint32_t num_frames_inflight,
                      glm::u32vec2 window_dims,
                      bool need_immediate);

    inline GLFWwindow *getWindow() { return window_; }

    void forceTransition(const DeviceState &dev,
                         const QueueState &present_queue,
                         uint32_t qf_idx);

    uint32_t acquireNext(const DeviceState &dev,
                         VkSemaphore signal_sema);

    VkImage getImage(uint32_t idx) const;
    uint32_t numSwapchainImages() const;

    void present(const DeviceState &dev, uint32_t swapchain_idx,
                 const QueueState &present_queue,
                 uint32_t num_wait_semas,
                 const VkSemaphore *wait_semas);

private:
    GLFWwindow *window_;
    VkSurfaceKHR surface_;
    Swapchain swapchain_;
    DynArray<VkImage> swapchain_imgs_;
};

}
}
