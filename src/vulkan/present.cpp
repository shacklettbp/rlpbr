#include "present.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

namespace RLpbr {
namespace vk {

void PresentationState::init()
{
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        fatalExit();
    }
}

vector<const char *> PresentationState::getInstanceExtensions()
{
    uint32_t count;
    const char **names = glfwGetRequiredInstanceExtensions(&count);

    vector<const char *> exts(count);
    memcpy(exts.data(), names, count * sizeof(const char *));

    return exts;
}

VkBool32 PresentationState::deviceSupportCallback(VkInstance inst,
                                                  VkPhysicalDevice phy,
                                                  uint32_t idx)
{
    auto glfw_ret = glfwGetPhysicalDevicePresentationSupport(inst, phy, idx);
    return glfw_ret == GLFW_TRUE ? VK_TRUE : VK_FALSE;
}

static GLFWwindow *makeWindow(glm::u32vec2 dims)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

#if 0
    auto monitor = glfwGetPrimaryMonitor();
    auto mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    return glfwCreateWindow(mode->width, mode->height, "RLPBR", monitor, nullptr);
#endif

    return glfwCreateWindow(dims.x, dims.y, "RLPBR", nullptr, nullptr);
}

VkSurfaceKHR getWindowSurface(const InstanceState &inst, GLFWwindow *window)
{
    VkSurfaceKHR surface;
    REQ_VK(glfwCreateWindowSurface(inst.hdl, window, nullptr, &surface));

    return surface;
}

static VkSurfaceFormatKHR selectSwapchainFormat(const InstanceState &inst,
                                                VkPhysicalDevice phy,
                                                VkSurfaceKHR surface)
{
    uint32_t num_formats;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, nullptr));

    DynArray<VkSurfaceFormatKHR> formats(num_formats);
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, formats.data()));

    if (num_formats == 0) {
        cerr  << "Zero swapchain formats" << endl;
        fatalExit();
    }

    // FIXME
    for (VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats[0];
}

static VkPresentModeKHR selectSwapchainMode(const InstanceState &inst,
                                            VkPhysicalDevice phy,
                                            VkSurfaceKHR surface,
                                            bool need_immediate)
{
    uint32_t num_modes;
    REQ_VK(inst.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, nullptr));

    DynArray<VkPresentModeKHR> modes(num_modes);
    REQ_VK(inst.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, modes.data()));

    for (VkPresentModeKHR mode : modes) {
        if (need_immediate && mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            return mode;
        } else if (!need_immediate && mode == VK_PRESENT_MODE_FIFO_KHR) {
            return mode;
        }
    }

    if (!need_immediate) {
        return modes[0];
    } else {
        cerr << "Could not find immediate swapchain" << endl;
        fatalExit();
    }
}

static Swapchain makeSwapchain(const InstanceState &inst,
                               const DeviceState &dev,
                               GLFWwindow *window,
                               VkSurfaceKHR surface,
                               uint32_t qf_idx,
                               uint32_t num_frames_inflight,
                               bool need_immediate)
{
    // Need to include this call despite the platform specific check
    // earlier (pre surface creation), or validation layers complain
    VkBool32 surface_supported;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceSupportKHR(
            dev.phy, qf_idx, surface, &surface_supported));

    if (surface_supported == VK_FALSE) {
        cerr << "GLFW surface doesn't support presentation" << endl;
        fatalExit();
    }

    VkSurfaceFormatKHR format = selectSwapchainFormat(inst, dev.phy, surface);
    VkPresentModeKHR mode = selectSwapchainMode(inst, dev.phy, surface,
                                                need_immediate);

    VkSurfaceCapabilitiesKHR caps;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceCapabilitiesKHR(
            dev.phy, surface, &caps));

    VkExtent2D swapchain_size = caps.currentExtent;
    if (swapchain_size.width == UINT32_MAX &&
        swapchain_size.height == UINT32_MAX) {
        glfwGetWindowSize(window, (int *)&swapchain_size.width,
                          (int *)&swapchain_size.height);

        swapchain_size.width = max(caps.minImageExtent.width,
                                   min(caps.maxImageExtent.width,
                                       swapchain_size.width));

        swapchain_size.height = max(caps.minImageExtent.height,
                                    min(caps.maxImageExtent.height,
                                        swapchain_size.height));
    }

    uint32_t num_requested_images =
        max(caps.minImageCount + 1, num_frames_inflight);
    if (caps.maxImageCount != 0 && num_requested_images > caps.maxImageCount) {
        num_requested_images = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchain_info;
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.pNext = nullptr;
    swapchain_info.flags = 0;
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = num_requested_images;
    swapchain_info.imageFormat = format.format;
    swapchain_info.imageColorSpace = format.colorSpace;
    swapchain_info.imageExtent = swapchain_size;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 0;
    swapchain_info.pQueueFamilyIndices = nullptr;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain;
    REQ_VK(dev.dt.createSwapchainKHR(dev.hdl, &swapchain_info, nullptr,
                                     &swapchain));

    return Swapchain {
        swapchain,
        glm::u32vec2(swapchain_size.width, swapchain_size.height),
    };
}

static DynArray<VkImage> getSwapchainImages(const DeviceState &dev,
                                            VkSwapchainKHR swapchain)
{
    uint32_t num_images;
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        nullptr));

    DynArray<VkImage> swapchain_images(num_images);
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        swapchain_images.data()));

    return swapchain_images;
}

PresentationState::PresentationState(const InstanceState &inst,
                                     const DeviceState &dev,
                                     uint32_t qf_idx,
                                     uint32_t num_frames_inflight,
                                     glm::u32vec2 window_dims,
                                     bool need_immediate)
    : window_(makeWindow(window_dims)),
      surface_(getWindowSurface(inst, window_)),
      swapchain_(makeSwapchain(inst, dev, window_, surface_,
                               qf_idx, num_frames_inflight,
                               need_immediate)),
      swapchain_imgs_(getSwapchainImages(dev, swapchain_.hdl))
{
}

void PresentationState::forceTransition(const DeviceState &dev,
    const QueueState &present_queue, uint32_t qf_idx)
{
    VkCommandPool tmp_pool = makeCmdPool(dev, qf_idx);
    VkCommandBuffer cmd = makeCmdBuffer(dev, tmp_pool);

    vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(swapchain_imgs_.size());

    for (int i = 0; i < (int)swapchain_imgs_.size(); i++) {
        barriers.push_back({
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            0,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_imgs_[i],
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        });
    }

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 0;
    render_submit.pWaitSemaphores = nullptr;
    render_submit.pWaitDstStageMask = nullptr;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &cmd;

    VkFence fence = makeFence(dev);

    present_queue.submit(dev, 1, &render_submit, fence);

    waitForFenceInfinitely(dev, fence);

    // FIXME, get an initialization pool / fence for stuff like this
    dev.dt.destroyFence(dev.hdl, fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);
}

uint32_t PresentationState::acquireNext(const DeviceState &dev,
                                        VkSemaphore signal_sema)
{
    uint32_t swapchain_idx;
    REQ_VK(dev.dt.acquireNextImageKHR(dev.hdl, swapchain_.hdl,
                                      0, signal_sema,
                                      VK_NULL_HANDLE,
                                      &swapchain_idx));

    return swapchain_idx;
}

VkImage PresentationState::getImage(uint32_t idx) const
{
    return swapchain_imgs_[idx];
}

uint32_t PresentationState::numSwapchainImages() const
{
    return swapchain_imgs_.size();
}

void PresentationState::present(const DeviceState &dev, uint32_t swapchain_idx,
                                const QueueState &present_queue,
                                uint32_t num_wait_semas,
                                const VkSemaphore *wait_semas)
{
    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.waitSemaphoreCount = num_wait_semas;
    present_info.pWaitSemaphores = wait_semas;

    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_.hdl;
    present_info.pImageIndices = &swapchain_idx;
    present_info.pResults = nullptr;

    present_queue.presentSubmit(dev, &present_info);
}

}
}
