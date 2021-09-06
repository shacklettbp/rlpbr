#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rlpbr/config.hpp>
#include <rlpbr/render.hpp>
#include <rlpbr_core/common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

#include "core.hpp"
#include "cuda_interop.hpp"
#include "descriptors.hpp"
#include "memory.hpp"
#include "shader.hpp"
#include "present.hpp"
#include "scene.hpp"

namespace RLpbr {
namespace vk {

struct InitConfig {
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

    uint32_t outputBytes;
    uint32_t normalBytes;
    uint32_t albedoBytes;
    uint32_t reservoirBytes;
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

    std::vector<LocalBuffer> reservoirs;
    std::vector<VkDeviceMemory> reservoirMemory;
};

struct RenderState {
    VkSampler repeatSampler;
    VkSampler clampSampler;

    ShaderPipeline rt;
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
    VkSemaphore renderSignal;
    VkSemaphore swapchainReady;
    VkCommandPool cmdPool;
    VkCommandBuffer renderCmd;

    half *outputBuffer;
    half *normalBuffer;
    half *albedoBuffer;

    FixedDescriptorPool rtPool;
    std::array<VkDescriptorSet, 2> rtSets;

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

struct VulkanBatch : public BatchBackend {
    FramebufferState fb;

    HostBuffer renderInputBuffer;
    PerBatchState state;

    uint32_t curBuffer;
};

class VulkanBackend : public RenderBackend {
public:
    struct Config {
        int gpuID;
        uint32_t batchSize;
        uint32_t maxLoaders;
        uint32_t maxTextureResolution;
        bool auxiliaryOutputs;
    };

    VulkanBackend(const RenderConfig &cfg, bool validate);
    LoaderImpl makeLoader();

    EnvironmentImpl makeEnvironment(const std::shared_ptr<Scene> &scene,
                                    const Camera &cam);

    RenderBatch::Handle makeRenderBatch();

    void render(RenderBatch &batch);

    void waitForBatch(RenderBatch &batch);

    half *getOutputPointer(RenderBatch &batch);
    AuxiliaryOutputs getAuxiliaryOutputs(RenderBatch &batch);

private:
    VulkanBackend(const RenderConfig &cfg,
                  const InitConfig &backend_cfg);

    const Config cfg_;

    const InstanceState inst;
    const DeviceState dev;

    MemoryAllocator alloc;

    const FramebufferConfig fb_cfg_;
    const ParamBufferConfig param_cfg_;
    RenderState render_state_;
    PipelineState pipeline_;

    DynArray<QueueState> transfer_queues_;
    DynArray<QueueState> graphics_queues_;
    DynArray<QueueState> compute_queues_;

    BSDFPrecomputed bsdf_precomp_;
    glm::u32vec3 launch_size_;

    std::atomic_int num_loaders_;
    VkDescriptorPool scene_pool_;
    SharedSceneState shared_scene_state_;

    uint32_t cur_queue_;
    uint32_t frame_counter_;

    std::optional<PresentationState> present_;
};

}
}
