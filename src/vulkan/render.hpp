#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rlpbr/config.hpp>
#include <rlpbr_core/common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

#include "core.hpp"
#include "cuda_interop.hpp"
#include "descriptors.hpp"
#include "memory.hpp"
#include "shader.hpp"
#include "present.hpp"

namespace RLpbr {
namespace vk {

struct BackendConfig {
    uint32_t numBatches;
    bool useZSobol;
    bool useAdvancedMaterials;
    bool validate;
    bool needPresent;
};

struct FramebufferConfig {
    uint32_t imgWidth;
    uint32_t imgHeight;

    uint32_t miniBatchSize;
    uint32_t numImagesWidePerMiniBatch;
    uint32_t numImagesTallPerMiniBatch;

    uint32_t numImagesWidePerBatch;
    uint32_t numImagesTallPerBatch;

    uint32_t frameWidth;
    uint32_t frameHeight;
    uint32_t totalWidth;
    uint32_t totalHeight;

    uint64_t linearOutputBytesPerBatch;
    uint64_t linearNormalBytesPerBatch;
    uint64_t linearAlbedoBytesPerBatch;

    uint64_t totalLinearOutputBytes;
    uint64_t totalLinearNormalBytes;
    uint64_t totalLinearAlbedoBytes;
};

struct ParamBufferConfig {
    VkDeviceSize totalTransformBytes;

    VkDeviceSize materialIndicesOffset;
    VkDeviceSize totalMaterialIndexBytes;

    VkDeviceSize lightsOffset;
    VkDeviceSize totalLightParamBytes;

    VkDeviceSize envOffset;
    VkDeviceSize totalEnvParamBytes;

    VkDeviceSize totalParamBytes;
};

struct FramebufferState {
    std::vector<LocalBuffer> outputs;
    std::vector<VkDeviceMemory> backings;

    std::vector<CudaImportedBuffer> exported;
};

struct RenderState {
    VkSampler repeatSampler;
    VkSampler clampSampler;

    ShaderPipeline rt;
    FixedDescriptorPool rtPool;
};

struct RTPipelineState {
    VkPipelineLayout layout;
    VkPipeline hdl;
};

struct PipelineState {
    // Not saved (no caching)
    VkPipelineCache pipelineCache;

    RTPipelineState rtState;
};

struct PerBatchState {
    VkFence fence;
    VkCommandPool cmdPool;
    VkCommandBuffer renderCmd;

    half *outputBuffer;
    half *normalBuffer;
    half *albedoBuffer;

    VkDescriptorSet rtSet;

    InstanceTransform *transformPtr;
    uint32_t *materialPtr;
    PackedLight *lightPtr;
    PackedEnv *envPtr;
};

struct BSDFPrecomputed {
    using ImgAndView = std::pair<LocalTexture, VkImageView>;

    VkDeviceMemory backing;

    ImgAndView msDiffuseAverage;
    ImgAndView msDiffuseDirectional;
    ImgAndView msGGXAverage;
    ImgAndView msGGXDirectional;
    ImgAndView msGGXInverse;
};

class VulkanBackend : public RenderBackend {
public:
    VulkanBackend(const RenderConfig &cfg, bool validate);
    LoaderImpl makeLoader();

    EnvironmentImpl makeEnvironment(const std::shared_ptr<Scene> &scene);

    uint32_t render(const Environment *envs);

    void waitForFrame(uint32_t batch_idx);

    half *getOutputPointer(uint32_t batch_idx);
    AuxiliaryOutputs getAuxiliaryOutputs(uint32_t batch_idx);

private:
    VulkanBackend(const RenderConfig &cfg,
                  const BackendConfig &backend_cfg);

    const uint32_t batch_size_;

    const InstanceState inst;
    const DeviceState dev;

    MemoryAllocator alloc;

    const FramebufferConfig fb_cfg_;
    const ParamBufferConfig param_cfg_;
    RenderState render_state_;
    PipelineState pipeline_;
    FramebufferState fb_;

    DynArray<QueueState> transfer_queues_;
    DynArray<QueueState> graphics_queues_;
    DynArray<QueueState> compute_queues_;

    HostBuffer render_input_buffer_;

    BSDFPrecomputed bsdf_precomp_;
    glm::u32vec3 launch_size_;

    std::atomic_int num_loaders_;
    int max_loaders_;
    uint32_t max_texture_resolution_;

    std::vector<PerBatchState> batch_states_;

    uint32_t cur_batch_;
    const uint32_t batch_mask_;
    uint32_t frame_counter_;

    std::optional<PresentationState> present_;
};

}
}
