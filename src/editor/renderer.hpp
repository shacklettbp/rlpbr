#pragma once

#include "vulkan/render.hpp"
#include "vulkan/scene.hpp"

namespace RLpbr {
namespace editor {

template <size_t N>
struct Pipeline {
    vk::ShaderPipeline shader;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    vk::FixedDescriptorPool descPool;
};

struct Framebuffer {
    vk::LocalImage colorAttachment;
    vk::LocalImage depthAttachment;
    VkImageView colorView;
    VkImageView depthView;

    VkFramebuffer hdl;
};

struct OverlayVertex {
    glm::vec3 position;
    glm::u8vec4 color;
};

struct HostRenderInput {
    vk::HostBuffer buffer;
    InstanceTransform *transformPtr;
    uint32_t *matIndices;
    OverlayVertex *overlayVertices;
    uint32_t *overlayIndices;
    uint32_t overlayIndexOffset;
    uint32_t maxTransforms;
    uint32_t maxMatIndices;
    uint32_t maxOverlayVertices;
    uint32_t maxOverlayIndices;
};

struct Frame {
    Framebuffer fb;
    VkCommandPool cmdPool;
    VkCommandBuffer drawCmd;
    VkFence cpuFinished;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;
    VkDescriptorSet defaultShaderSet;
    VkDescriptorSet overlayShaderSet;
    HostRenderInput renderInput;
};

struct EditorCam {
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;

    bool perspective = true;
    float fov = 90.f;
    float orthoHeight = 5.f;
};

class Renderer {
public:
    struct FrameConfig {
        float lineWidth = 2.f;
        bool linesNoDepthTest = false;

        // FIXME These should all be moved somewhere else
        bool showNavmesh = true;
        bool navmeshNoEdges = false;
        bool wireframeOnly = false;
    };

    Renderer(uint32_t gpu_id, uint32_t img_width,
             uint32_t img_height);
    Renderer(const Renderer &) = delete;
    GLFWwindow *getWindow();

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_data);

    void waitUntilFrameReady();
    void startFrame();
    void render(Scene *scene, const EditorCam &cam,
                const FrameConfig &cfg,
                uint32_t num_overlay_vertices,
                const OverlayVertex *overlay_vertices,
                uint32_t num_overlay_tri_indices,
                uint32_t num_overlay_line_indices,
                const uint32_t *overlay_indices);

    void waitForIdle();

    vk::InstanceState inst;
    vk::DeviceState dev;
    vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;
    VkQueue transfer_queue_;
    VkQueue compute_queue_;
    VkQueue render_transfer_queue_;
    VkQueue compute_transfer_queue_;

    // Fixme remove
    vk::QueueState transfer_wrapper_;
    vk::QueueState render_transfer_wrapper_;
    vk::QueueState present_wrapper_;

    glm::u32vec2 fb_dims_;
    std::array<VkClearValue, 2> fb_clear_;
    vk::PresentationState present_;
    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass gui_render_pass_;
    Pipeline<1> default_pipeline_;
    Pipeline<3> overlay_pipeline_;

    uint32_t cur_frame_;
    DynArray<Frame> frames_;

    VkDescriptorPool scene_desc_pool_;
    vk::SharedSceneState shared_scene_state_;

    vk::VulkanLoader loader_;
};

}
}
