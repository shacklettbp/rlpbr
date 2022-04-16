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
#include "denoiser.hpp"

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

    uint32_t numTilesWide;
    uint32_t numTilesTall;

    uint32_t outputBytes;
    uint32_t hdrBytes;
    uint32_t normalBytes;
    uint32_t albedoBytes;
    uint32_t reservoirBytes;
    uint32_t illuminanceBytes;
    uint32_t adaptiveBytes;
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

    std::optional<HostBuffer> adaptiveReadback;
    std::optional<HostBuffer> exposureReadback;

    int outputIdx;
    int hdrIdx;
    int normalIdx;
    int albedoIdx;
    int illuminanceIdx;
    int adaptiveIdx;
};

struct RenderState {
    VkSampler repeatSampler;
    VkSampler clampSampler;

    ShaderPipeline rt;
    ShaderPipeline exposure;
    ShaderPipeline tonemap;
};

struct PipelineState {
    VkPipelineLayout layout;
    VkPipeline hdl;
};

struct RenderPipelines {
    // Not saved (no caching)
    VkPipelineCache pipelineCache;

    PipelineState rt;
    PipelineState exposure;
    PipelineState tonemap;
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
    FixedDescriptorPool exposurePool;
    VkDescriptorSet exposureSet;
    FixedDescriptorPool tonemapPool;
    VkDescriptorSet tonemapSet;

    InstanceTransform *transformPtr;
    uint32_t *materialPtr;
    PackedLight *lightPtr;
    PackedEnv *envPtr;
    InputTile *tileInputPtr;
    AdaptiveTile *adaptiveReadbackPtr;
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

    std::optional<HostBuffer> adaptiveInput;
    HostBuffer renderInputStaging;
    LocalBuffer renderInputDev;
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
        uint32_t spp;
        bool auxiliaryOutputs;
        bool tonemap;
        bool enableRandomization;
        bool adaptiveSampling;
        bool denoise;
    };

    VulkanBackend(const RenderConfig &cfg, bool validate);
    LoaderImpl makeLoader();

    EnvironmentImpl makeEnvironment(const std::shared_ptr<Scene> &scene,
                                    const Camera &cam);

    void setActiveEnvironmentMaps(
        std::shared_ptr<EnvironmentMapGroup> env_maps);

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
    RenderPipelines pipelines_;

    DynArray<QueueState> transfer_queues_;
    DynArray<QueueState> compute_queues_;

    BSDFPrecomputed bsdf_precomp_;
    glm::u32vec3 launch_size_;

    std::atomic_int num_loaders_;
    VkDescriptorPool scene_pool_;
    SharedSceneState shared_scene_state_;
    std::unique_ptr<SharedEnvMapState> env_map_state_;
    std::shared_ptr<VulkanEnvMapGroup> cur_env_maps_;

    uint32_t cur_queue_;
    uint32_t frame_counter_;

    std::optional<PresentationState> present_;
    std::optional<Denoiser> denoiser_;
};

}
}
