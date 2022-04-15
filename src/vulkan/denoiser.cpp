#include "denoiser.hpp"
#include <OpenImageDenoise/oidn.hpp>
#include "render.hpp"

#include <iostream>

namespace RLpbr {
namespace vk {

struct Denoiser::Impl {
    oidn::DeviceRef dev;
    oidn::FilterRef filter;
};

Denoiser::Denoiser(MemoryAllocator &alloc, const RenderConfig &cfg)
    : color_readback_(alloc.makeHostBuffer(
        cfg.batchSize * cfg.imgHeight * cfg.imgWidth * sizeof(uint16_t) * 4,
        true)),
      albedo_readback_(alloc.makeHostBuffer(
        cfg.batchSize * cfg.imgHeight * cfg.imgWidth * sizeof(uint16_t) * 3,
        true)),
      normal_readback_(alloc.makeHostBuffer(
        cfg.batchSize * cfg.imgHeight * cfg.imgWidth * sizeof(uint16_t) * 3,
        true)),
      color_input_(
          new uint16_t[cfg.imgHeight * cfg.imgWidth * 3]),
      albedo_input_(
          new uint16_t[cfg.imgHeight * cfg.imgWidth * 3]),
      normal_input_(
          new uint16_t[cfg.imgHeight * cfg.imgWidth * 3]),
      output_(
          new uint16_t[cfg.imgHeight * cfg.imgWidth * 3]),
      impl_([&]() {
          using namespace oidn;

          auto dev = newDevice();
          dev.commit();

          auto filter = dev.newFilter("RT");
          filter.setImage("color", color_input_.get(),
                          Format::Half3, cfg.imgWidth, cfg.imgHeight);
          filter.setImage("albedo", albedo_input_.get(), Format::Half3,
                          cfg.imgWidth, cfg.imgHeight);
          filter.setImage("normal", normal_input_.get(), Format::Half3,
                          cfg.imgWidth, cfg.imgHeight);
          filter.setImage("output", output_.get(), Format::Half3,
                          cfg.imgWidth, cfg.imgHeight);
          filter.set("hdr", true);
          filter.commit();

          return new Denoiser::Impl {
              std::move(dev),
              std::move(filter),
          };
      }())
{}

Denoiser::~Denoiser()
{}

void Denoiser::denoise(const DeviceState &dev,
                       const FramebufferConfig &fb_cfg,
                       VkCommandPool cmd_pool,
                       VkCommandBuffer cmd,
                       VkFence fence,
                       QueueState &gfx_queue,
                       LocalBuffer &output,
                       const LocalBuffer &albedo,
                       const LocalBuffer &normal,
                       uint32_t batch_size)
{
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, cmd_pool, 0));

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    VkMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dev.dt.cmdPipelineBarrier(cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0,
                              1, &barrier, 0, nullptr, 0, nullptr);

    VkBufferCopy color_copy;
    color_copy.srcOffset = 0;
    color_copy.dstOffset = 0;
    color_copy.size = fb_cfg.outputBytes;

    dev.dt.cmdCopyBuffer(cmd, output.buffer,
                         color_readback_.buffer, 1, &color_copy);

    VkBufferCopy albedo_copy;
    albedo_copy.srcOffset = 0;
    albedo_copy.dstOffset = 0;
    albedo_copy.size = fb_cfg.albedoBytes;

    dev.dt.cmdCopyBuffer(cmd, albedo.buffer,
                         albedo_readback_.buffer, 1, &albedo_copy);

    VkBufferCopy normal_copy;
    normal_copy.srcOffset = 0;
    normal_copy.dstOffset = 0;
    normal_copy.size = fb_cfg.normalBytes;

    dev.dt.cmdCopyBuffer(cmd, normal.buffer,
                         normal_readback_.buffer, 1, &normal_copy);

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &cmd,
        0,
        nullptr,
    };

    gfx_queue.submit(
        dev, 1, &submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);

    for (int i = 0; i < (int)batch_size; i++) {
        for (int y = 0; y < (int)fb_cfg.imgHeight; y++) {
            for (int x = 0; x < (int)fb_cfg.imgWidth; x++) {
                for (int c = 0; c < 3; c++) {
                    int idx_3c = y * fb_cfg.imgWidth * 3 + x * 3 + c;

                    int batch_offset_3c =
                        i * fb_cfg.imgHeight * fb_cfg.imgWidth * 3;

                    color_input_.get()[idx_3c] = 
                        ((uint16_t *)color_readback_.ptr)[
                           i * fb_cfg.imgHeight * fb_cfg.imgWidth * 4 +
                               y * fb_cfg.imgWidth * 4 + x * 4 + c];

                    albedo_input_.get()[idx_3c] =
                        ((uint16_t *)albedo_readback_.ptr)[
                            batch_offset_3c + idx_3c];

                    normal_input_.get()[idx_3c] =
                        ((uint16_t *)normal_readback_.ptr)[
                            batch_offset_3c + idx_3c];

                    output_.get()[idx_3c] = 0;
                }
            }
        }

        impl_->filter.execute();

        const char* oidn_error_message;
        if (impl_->dev.getError(oidn_error_message) != oidn::Error::None) {
            std::cerr << "OIDN Error: " << oidn_error_message << std::endl;
            abort();
        }

        for (int y = 0; y < (int)fb_cfg.imgHeight; y++) {
            for (int x = 0; x < (int)fb_cfg.imgWidth; x++) {
                for (int c = 0; c < 3; c++) {
                    ((uint16_t *)color_readback_.ptr)[
                        i * fb_cfg.imgHeight * fb_cfg.imgWidth * 4 +
                            y * fb_cfg.imgWidth * 4 + x * 4 + c] =
                        output_.get()[y * fb_cfg.imgWidth * 3 + x * 3 + c];
                }
            }
        }
    }

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, cmd_pool, 0));

    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    dev.dt.cmdCopyBuffer(cmd, color_readback_.buffer,
                         output.buffer, 1, &color_copy);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dev.dt.cmdPipelineBarrier(cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1, &barrier, 0, nullptr, 0, nullptr);

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    gfx_queue.submit(
        dev, 1, &submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);
}

}
}
