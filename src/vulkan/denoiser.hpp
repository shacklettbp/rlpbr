#pragma once

#include "memory.hpp"
#include <rlpbr/config.hpp>
#include <rlpbr/utils.hpp>

namespace RLpbr {
namespace vk {

struct FramebufferConfig;

class Denoiser {
public:
    Denoiser(MemoryAllocator &alloc, const RenderConfig &cfg);
    ~Denoiser();

    void denoise(const DeviceState &dev,
                 const FramebufferConfig &fb_cfg,
                 VkCommandPool cmd_pool,
                 VkCommandBuffer cmd,
                 VkFence fence,
                 QueueState &gfx_queue,
                 LocalBuffer &output,
                 const LocalBuffer &albedo,
                 const LocalBuffer &normal,
                 uint32_t batch_size);

private:
    struct Impl;

    HostBuffer color_readback_;
    HostBuffer albedo_readback_;
    HostBuffer normal_readback_;
    std::unique_ptr<float[]> color_input_;
    std::unique_ptr<float[]> albedo_input_;
    std::unique_ptr<float[]> normal_input_;
    std::unique_ptr<float[]> output_;

    std::unique_ptr<Impl> impl_;
};

}
}
