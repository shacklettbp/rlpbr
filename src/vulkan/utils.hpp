#pragma once

#include <deque>
#include <mutex>
#include <string>

#include <rlpbr_core/utils.hpp>

#include "core.hpp"
#include "shader.hpp"

namespace RLpbr {
namespace vk {

class QueueState {
public:
    inline QueueState(VkQueue queue_hdl, bool shared);

    inline void submit(const DeviceState &dev,
                       uint32_t submit_count,
                       const VkSubmitInfo *pSubmits,
                       VkFence fence) const;

    inline void bindSubmit(const DeviceState &dev,
                           uint32_t submit_count,
                           const VkBindSparseInfo *pSubmits,
                           VkFence fence) const;

    inline bool presentSubmit(const DeviceState &dev,
                              const VkPresentInfoKHR *present_info) const;

private:
    VkQueue queue_hdl_;
    bool shared_;
    mutable std::mutex mutex_;
};

inline VkCommandPool makeCmdPool(const DeviceState &dev, uint32_t qf_idx);

inline VkCommandBuffer makeCmdBuffer(
    const DeviceState &dev,
    VkCommandPool pool,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

inline VkQueue makeQueue(const DeviceState &dev,
                         uint32_t qf_idx,
                         uint32_t queue_idx);
inline VkSemaphore makeBinarySemaphore(const DeviceState &dev);

inline VkSemaphore makeBinaryExternalSemaphore(const DeviceState &dev);
int exportBinarySemaphore(const DeviceState &dev, VkSemaphore semaphore);

inline VkFence makeFence(const DeviceState &dev, bool pre_signal = false);

VkSampler makeImmutableSampler(const DeviceState &dev,
                               VkSamplerAddressMode mode);

inline void waitForFenceInfinitely(const DeviceState &dev, VkFence fence);

inline void resetFence(const DeviceState &dev, VkFence fence);

inline VkDescriptorSet makeDescriptorSet(const DeviceState &dev,
                                         VkDescriptorPool pool,
                                         VkDescriptorSetLayout layout);

inline VkDeviceSize alignOffset(VkDeviceSize offset, VkDeviceSize alignment);

template <typename T>
inline T divideRoundUp(T a, T b);

inline uint32_t getWorkgroupSize(uint32_t num_items);

void printVkError(VkResult res, const char *msg);

static inline VkResult checkVk(VkResult res,
                               const char *msg,
                               bool fatal = true) noexcept
{
    if (res != VK_SUCCESS) {
        printVkError(res, msg);
        if (fatal) {
            fatalExit();
        }
    }

    return res;
}

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)
#define REQ_VK(expr) checkVk((expr), LOC_APPEND(#expr))
#define CHK_VK(expr) checkVk((expr), LOC_APPEND(#expr), false)

}
}

#include "utils.inl"
