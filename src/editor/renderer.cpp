#include "renderer.hpp"
#include "shader.hpp"

#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace editor {

namespace InternalConfig {

struct NavmeshPipelineIndices {
    int wireNoDepth;
    int wireWithDepth;
    int filled;
};

inline constexpr uint32_t numFrames = 2;
inline constexpr NavmeshPipelineIndices navPipelineIdxs { 0, 1, 2 };
inline constexpr uint32_t initMaxTransforms = 100000;
inline constexpr uint32_t initMaxMatIndices = 100000;
inline constexpr uint32_t initMaxOverlayVertices = 50000000;
inline constexpr uint32_t initMaxOverlayIndices = 10000000;
}

using namespace vk;

static VkQueue makeGFXQueue(const DeviceState &dev, uint32_t idx)
{
    if (idx >= dev.numGraphicsQueues) {
        cerr << "Not enough graphics queues" << endl;
        fatalExit();
    }

    return makeQueue(dev, dev.gfxQF, idx);
}

static VkQueue makeComputeQueue(const DeviceState &dev, uint32_t idx)
{
    if (idx >= dev.numComputeQueues) {
        cerr << "Not enough compute queues" << endl;
        fatalExit();
    }

    return makeQueue(dev, dev.computeQF, idx);
}

static VkQueue makeTransferQueue(const DeviceState &dev, uint32_t idx)
{
    if (idx >= dev.numTransferQueues) {
        cerr << "Not enough transfer queues" << endl;
        fatalExit();
    }

    return makeQueue(dev, dev.transferQF, idx);
}

static VkRenderPass makeRenderPass(const DeviceState &dev,
                                   VkFormat color_fmt,
                                   VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, color_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    
    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    array<VkSubpassDependency, 3> pre_deps {{
        {
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
        },
        {
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            0,
        },
        {
            0,
            VK_SUBPASS_EXTERNAL,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
        },
    }};

    render_pass_info.dependencyCount = pre_deps.size();
    render_pass_info.pDependencies = pre_deps.data();

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static ShaderPipeline makeDefaultShader(const DeviceState &dev,
                                        VkSampler repeat_sampler,
                                        VkSampler clamp_sampler)
{
    vector<string> shader_defines;

    return ShaderPipeline(dev, {
            "editor.vert",
            "editor.frag",
        }, {
            {0, 2, repeat_sampler, 1, 0},
            {0, 3, clamp_sampler, 1, 0},
            {1, 1, VK_NULL_HANDLE,
                VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
        }, 
        shader_defines,
        STRINGIFY(EDITOR_SHADER_DIR));
}

static VkPipelineCache getPipelineCache(const DeviceState &dev)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    return pipeline_cache;
}

static Pipeline<1> makePipeline(const DeviceState &dev,
                                VkPipelineCache pipeline_cache,
                                VkRenderPass render_pass,
                                VkSampler repeat_sampler,
                                VkSampler clamp_sampler,
                                uint32_t num_frames)
{
    ShaderPipeline shader =
        makeDefaultShader(dev, repeat_sampler, clamp_sampler);

    // Disable auto vertex assembly
    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly (standard tri indices)
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport (fully dynamic)
    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo raster_info {};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach,
    }};

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
        shader.getLayout(0),
        shader.getLayout(1),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shader.getShader(0),
            "main",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shader.getShader(1),
            "main",
            nullptr,
        },
    }};

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &input_assembly_info;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &raster_info;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &depth_info;
    gfx_info.pColorBlendState = &blend_info;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_pass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline draw_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr, &draw_pipeline));

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return {
        move(shader),
        draw_layout,
        { draw_pipeline },
        move(desc_pool),
    };
}

static ShaderPipeline makeNavmeshShader(const DeviceState &dev)
{
    vector<string> shader_defines;

    return ShaderPipeline(dev, {
            "navmesh.vert",
            "navmesh.frag",
        }, {}, {},
        STRINGIFY(EDITOR_SHADER_DIR));
}

static Pipeline<3> makeNavmeshPipeline(const DeviceState &dev,
                                       VkPipelineCache pipeline_cache,
                                       VkRenderPass render_pass,
                                       uint32_t num_frames)
{
    ShaderPipeline shader = makeNavmeshShader(dev);

    // Disable auto vertex assembly
    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Viewport (fully dynamic)
    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Dynamic
    array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(NavmeshPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 1> draw_desc_layouts {{
        shader.getLayout(0),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shader.getShader(0),
            "main",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shader.getShader(1),
            "main",
            nullptr,
        },
    }};

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_TRUE;
    blend_attach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend_attach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attach.colorBlendOp = VK_BLEND_OP_ADD;
    blend_attach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_attach.alphaBlendOp = VK_BLEND_OP_ADD;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach,
    }};

    VkPipelineRasterizationLineStateCreateInfoEXT line_info {};
    line_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT;
    line_info.pNext = nullptr;
    line_info.lineRasterizationMode =
        VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;

    auto makeRasterInfo = [&](bool lines, bool depth_test) {
        // Assembly
        VkPipelineInputAssemblyStateCreateInfo input_info {};
        input_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        if (lines) {
            input_info.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        } else {
            input_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }
        input_info.primitiveRestartEnable = VK_FALSE;

        // Rasterization
        VkPipelineRasterizationStateCreateInfo raster_info {};
        raster_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster_info.pNext = &line_info;
        raster_info.depthClampEnable = VK_FALSE;
        raster_info.rasterizerDiscardEnable = VK_FALSE;
        raster_info.polygonMode =
            lines ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        raster_info.cullMode = VK_CULL_MODE_NONE;
        raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        if (depth_test) {
            raster_info.depthBiasEnable = VK_TRUE;
            if (lines) {
                raster_info.depthBiasConstantFactor = -2.1f;
            } else {
                raster_info.depthBiasConstantFactor = -2.f;
            }
            raster_info.depthBiasSlopeFactor = 0.f;
        } else {
            raster_info.depthBiasEnable = VK_FALSE;
        }
        raster_info.lineWidth = 0.f;

        // Depth/Stencil
        VkPipelineDepthStencilStateCreateInfo depth_info {};
        depth_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_info.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
        depth_info.depthWriteEnable = VK_FALSE;
        depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depth_info.depthBoundsTestEnable = VK_FALSE;
        depth_info.stencilTestEnable = VK_FALSE;
        depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

        VkPipelineColorBlendStateCreateInfo blend_info {};
        blend_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend_info.logicOpEnable = VK_FALSE;

        blend_info.attachmentCount =
            static_cast<uint32_t>(blend_attachments.size());
        blend_info.pAttachments = blend_attachments.data();

        return make_tuple(input_info, raster_info, depth_info, blend_info);
    };

    auto [wire_nodepth_input, wire_nodepth_raster,
          wire_nodepth_depth, wire_nodepth_blend] =
              makeRasterInfo(true, false);

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &wire_nodepth_input;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &wire_nodepth_raster;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &wire_nodepth_depth;
    gfx_info.pColorBlendState = &wire_nodepth_blend;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_pass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline wire_nodepth_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr,
                                          &wire_nodepth_pipeline));

    auto [wire_depth_input, wire_depth_raster,
          wire_depth_depth, wire_depth_blend] = 
            makeRasterInfo(true, true);

    gfx_info.pInputAssemblyState = &wire_depth_input;
    gfx_info.pRasterizationState = &wire_depth_raster;
    gfx_info.pDepthStencilState = &wire_depth_depth;
    gfx_info.pColorBlendState = &wire_depth_blend;

    VkPipeline wire_depth_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr,
                                          &wire_depth_pipeline));

    auto [fill_input, fill_raster, fill_depth, fill_blend] = 
        makeRasterInfo(false, true);

    gfx_info.pInputAssemblyState = &fill_input;
    gfx_info.pRasterizationState = &fill_raster;
    gfx_info.pDepthStencilState = &fill_depth;
    gfx_info.pColorBlendState = &fill_blend;

    VkPipeline fill_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr,
                                          &fill_pipeline));

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    std::array<VkPipeline, 3> pipelines;
    pipelines[InternalConfig::navPipelineIdxs.wireNoDepth] =
        wire_nodepth_pipeline;
    pipelines[InternalConfig::navPipelineIdxs.wireWithDepth] =
        wire_depth_pipeline;
    pipelines[InternalConfig::navPipelineIdxs.filled] =
        fill_pipeline;

    return {
        move(shader),
        draw_layout,
        move(pipelines),
        move(desc_pool),
    };
}

static InstanceState initializeInstance()
{
    auto get_inst_addr = vk::PresentationState::init();
    vk::ShaderPipeline::initCompiler();

    bool enable_validation;
    char *validate_env = getenv("RLPBR_VALIDATE");
    if (!validate_env || validate_env[0] == '0') {
        enable_validation = false;
    } else {
        enable_validation = true;
    }

    return InstanceState(get_inst_addr, enable_validation, true,
                         PresentationState::getInstanceExtensions());
}

static Framebuffer makeFramebuffer(const DeviceState &dev,
                                   MemoryAllocator &alloc,
                                   glm::u32vec2 fb_dims,
                                   VkRenderPass render_pass)
{
    auto color = alloc.makeColorAttachment(fb_dims.x, fb_dims.y);
    auto depth = alloc.makeDepthAttachment(fb_dims.x, fb_dims.y);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = color.image;
    view_info.format = alloc.getColorAttachmentFormat();

    VkImageView color_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &color_view));

    view_info.image = depth.image;
    view_info.format = alloc.getDepthAttachmentFormat();
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    array attachment_views {
        color_view,
        depth_view,
    };

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_dims.x;
    fb_info.height = fb_dims.y;
    fb_info.layers = 1;

    VkFramebuffer hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &hdl));

    return Framebuffer {
        move(color),
        move(depth),
        color_view,
        depth_view,
        hdl,
    };
}

static HostRenderInput makeHostRenderInput(const DeviceState &dev,
                                           MemoryAllocator &alloc,
                                           VkDescriptorSet default_shader_set,
                                           VkDescriptorSet overlay_shader_set,
                                           uint32_t num_transforms,
                                           uint32_t num_mat_indices,
                                           uint32_t num_overlay_verts,
                                           uint32_t num_overlay_indices)
{
    size_t txfm_bytes = sizeof(InstanceTransform) * num_transforms;
    size_t mat_idx_bytes = sizeof(uint32_t) * num_mat_indices;
    size_t overlay_vert_bytes = sizeof(OverlayVertex) * num_overlay_verts;
    size_t overlay_idx_bytes = sizeof(uint32_t) * num_overlay_indices;

    size_t input_buf_size = txfm_bytes;

    size_t mat_idx_offset = alloc.alignStorageBufferOffset(input_buf_size);
    input_buf_size =
        mat_idx_offset + mat_idx_bytes;

    size_t overlay_vert_offset =
        alloc.alignStorageBufferOffset(input_buf_size);
    input_buf_size = overlay_vert_offset + overlay_vert_bytes;

    size_t overlay_idx_offset = alloc.alignStorageBufferOffset(input_buf_size);
    input_buf_size = overlay_idx_offset + overlay_idx_bytes;

    // FIXME, rework MemoryAllocator to get rid of dev_addr = true
    auto render_input = alloc.makeHostBuffer(input_buf_size, true);

    InstanceTransform *transform_ptr =
        (InstanceTransform *)render_input.ptr;

    uint32_t *mat_idx_ptr =
        (uint32_t *)((char *)render_input.ptr + mat_idx_offset);

    OverlayVertex *overlay_vert_ptr =
        (OverlayVertex *)((char *)render_input.ptr + overlay_vert_offset);

    uint32_t *overlay_idx_ptr =
        (uint32_t *)((char *)render_input.ptr + overlay_idx_offset);

    DescriptorUpdates desc_updates(3);

    VkDescriptorBufferInfo txfm_info;
    txfm_info.buffer = render_input.buffer;
    txfm_info.offset = 0;
    txfm_info.range = txfm_bytes;

    desc_updates.buffer(default_shader_set, &txfm_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo mat_idx_info;
    mat_idx_info.buffer = render_input.buffer;
    mat_idx_info.offset = mat_idx_offset;
    mat_idx_info.range = mat_idx_bytes;

    desc_updates.buffer(default_shader_set, &mat_idx_info, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo overlay_vert_info;
    overlay_vert_info.buffer = render_input.buffer;
    overlay_vert_info.offset = overlay_vert_offset;
    overlay_vert_info.range = overlay_vert_bytes;

    desc_updates.buffer(overlay_shader_set, &overlay_vert_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    desc_updates.update(dev);

    return {
        move(render_input),
        transform_ptr,
        mat_idx_ptr,
        overlay_vert_ptr,
        overlay_idx_ptr,
        (uint32_t)overlay_idx_offset,
        num_transforms,
        num_mat_indices,
        num_overlay_verts,
        num_overlay_indices,
    };
}

static void makeFrame(const DeviceState &dev, MemoryAllocator &alloc,
                      glm::u32vec2 fb_dims, VkRenderPass render_pass,
                      VkDescriptorSet default_shader_set,
                      VkDescriptorSet overlay_shader_set,
                      Frame *dst)
{
    auto fb = makeFramebuffer(dev, alloc, fb_dims, render_pass);

    VkCommandPool cmd_pool = makeCmdPool(dev, dev.gfxQF);

    new (dst) Frame {
        move(fb),
        cmd_pool,
        makeCmdBuffer(dev, cmd_pool),
        makeFence(dev, true),
        makeBinarySemaphore(dev),
        makeBinarySemaphore(dev),
        default_shader_set,
        overlay_shader_set,
        makeHostRenderInput(dev, alloc, default_shader_set,
                            overlay_shader_set,
                            InternalConfig::initMaxTransforms,
                            InternalConfig::initMaxMatIndices,
                            InternalConfig::initMaxOverlayVertices,
                            InternalConfig::initMaxOverlayIndices),
    };
}

static array<VkClearValue, 2> makeClearValues()
{
    VkClearValue color_clear;
    color_clear.color = {{0.f, 0.f, 0.f, 1.f}};

    VkClearValue depth_clear;
    depth_clear.depthStencil = {1.f, 0};

    return {
        color_clear,
        depth_clear,
    };
}

static void imguiVkCheck(VkResult res)
{
    checkVk(res, "ImGui vulkan error");
}

static VkRenderPass makeImGuiRenderPass(const DeviceState &dev,
                                        VkFormat color_fmt,
                                        VkFormat depth_fmt)
{
    array<VkAttachmentDescription, 2> attachment_descs {{
        {
            0,
            color_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        },
        {
            0,
            depth_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};

    array<VkAttachmentReference, 2> attachment_refs {{
        {
            0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        },
        {
            1,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};
    
    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = attachment_refs.size() - 1;
    subpass_desc.pColorAttachments = attachment_refs.data();
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount = attachment_descs.size();
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    array<VkSubpassDependency, 2> pre_deps {{
        {
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
        },
        {
            0,
            VK_SUBPASS_EXTERNAL,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            0,
        },
    }};

    render_pass_info.dependencyCount = pre_deps.size();
    render_pass_info.pDependencies = pre_deps.data();

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

struct ImGUIVkLookupData {
    PFN_vkGetDeviceProcAddr getDevAddr;
    VkDevice dev;
    PFN_vkGetInstanceProcAddr getInstAddr;
    VkInstance inst;
};

static PFN_vkVoidFunction imguiVKLookup(const char *fname,
                                        void *user_data)
{
    auto data = (ImGUIVkLookupData *)user_data;

    auto addr = data->getDevAddr(data->dev, fname);

    if (!addr) {
        addr = data->getInstAddr(data->inst, fname);
    }

    if (!addr) {
        cerr << "Failed to load ImGUI vulkan function: " << fname << endl;
        abort();
    }

    return addr;
}

static VkRenderPass imguiInit(GLFWwindow *window, const DeviceState &dev,
                              const InstanceState &inst, VkQueue ui_queue,
                              VkPipelineCache pipeline_cache,
                              VkFormat color_fmt,
                              VkFormat depth_fmt)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    auto font_path = string(STRINGIFY(RLPBR_DATA_DIR)) + "/font.ttf";

    io.Fonts->AddFontFromFileTTF(font_path.c_str(), 26);

    auto &style = ImGui::GetStyle();
    style.ScaleAllSizes(2.f);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Taken from imgui/examples/example_glfw_vulkan/main.cpp
    VkDescriptorPool desc_pool;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl,
            &pool_info, nullptr, &desc_pool));
    }

    ImGui_ImplVulkan_InitInfo vk_init = {};
    vk_init.Instance = inst.hdl;
    vk_init.PhysicalDevice = dev.phy;
    vk_init.Device = dev.hdl;
    vk_init.QueueFamily = dev.gfxQF;
    vk_init.Queue = ui_queue;
    vk_init.PipelineCache = pipeline_cache;
    vk_init.DescriptorPool = desc_pool;
    vk_init.MinImageCount = InternalConfig::numFrames;
    vk_init.ImageCount = InternalConfig::numFrames;
    vk_init.CheckVkResultFn = imguiVkCheck;

    VkRenderPass imgui_renderpass = makeImGuiRenderPass(dev, color_fmt,
                                                        depth_fmt);

    ImGUIVkLookupData lookup_data {
        dev.dt.getDeviceProcAddr,
        dev.hdl,
        inst.dt.getInstanceProcAddr,
        inst.hdl,
    };
    ImGui_ImplVulkan_LoadFunctions(imguiVKLookup, &lookup_data);
    ImGui_ImplVulkan_Init(&vk_init, imgui_renderpass);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer tmp_cmd = makeCmdBuffer(dev, tmp_pool);
    VkCommandBufferBeginInfo tmp_begin {};
    tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd, &tmp_begin));
    ImGui_ImplVulkan_CreateFontsTexture(tmp_cmd);
    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd));

    VkFence tmp_fence = makeFence(dev);

    VkSubmitInfo tmp_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &tmp_cmd,
        0,
        nullptr,
    };

    REQ_VK(dev.dt.queueSubmit(ui_queue, 1, &tmp_submit, tmp_fence));
    waitForFenceInfinitely(dev, tmp_fence);

    dev.dt.destroyFence(dev.hdl, tmp_fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return imgui_renderpass;
}

Renderer::Renderer(uint32_t gpu_id, uint32_t img_width, uint32_t img_height)
    : inst(initializeInstance()),
      dev(inst.makeDevice(getUUIDFromCudaID(gpu_id), 2, 2, 1,
                          PresentationState::deviceSupportCallback)),
      alloc(dev, inst),
      render_queue_(makeGFXQueue(dev, 0)),
      transfer_queue_(makeTransferQueue(dev, 0)),
      compute_queue_(makeComputeQueue(dev, 0)),
      render_transfer_queue_(makeGFXQueue(dev, 1)),
      compute_transfer_queue_(makeComputeQueue(dev, 1)),
      transfer_wrapper_(transfer_queue_, false),
      render_transfer_wrapper_(render_transfer_queue_, false),
      present_wrapper_(render_queue_, false),
      fb_dims_(img_width, img_height),
      fb_clear_(makeClearValues()),
      present_(inst, dev, dev.gfxQF, InternalConfig::numFrames, fb_dims_,
               true),
      pipeline_cache_(getPipelineCache(dev)),
      repeat_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_REPEAT)),
      clamp_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)),
      render_pass_(makeRenderPass(dev, alloc.getColorAttachmentFormat(),
                                  alloc.getDepthAttachmentFormat())),
      gui_render_pass_(imguiInit(present_.getWindow(), dev, inst,
                                 render_queue_, pipeline_cache_,
                                 alloc.getColorAttachmentFormat(),
                                 alloc.getDepthAttachmentFormat())),
      default_pipeline_(makePipeline(dev, pipeline_cache_, render_pass_,
                                     repeat_sampler_, clamp_sampler_,
                                     InternalConfig::numFrames)),
      overlay_pipeline_(makeNavmeshPipeline(dev, pipeline_cache_,
                                            render_pass_,
                                            InternalConfig::numFrames)),
      cur_frame_(0),
      frames_(InternalConfig::numFrames),
      scene_desc_pool_(default_pipeline_.shader.makePool(1, 1)),
      scene_set_(makeDescriptorSet(dev, scene_desc_pool_,
                                   default_pipeline_.shader.getLayout(1))),
      loader_(dev, alloc, transfer_wrapper_,
              render_transfer_wrapper_,
              scene_set_,
              dev.gfxQF, 128)
{
    for (int i = 0; i < (int)frames_.size(); i++) {
        makeFrame(dev, alloc, fb_dims_, render_pass_,
                  default_pipeline_.descPool.makeSet(),
                  overlay_pipeline_.descPool.makeSet(), &frames_[i]);
    }

}

GLFWwindow *Renderer::getWindow()
{
    return present_.getWindow();
}

shared_ptr<Scene> Renderer::loadScene(SceneLoadData &&load_data)
{
    return loader_.loadScene(move(load_data));
}

void Renderer::waitUntilFrameReady()
{
    Frame &frame = frames_[cur_frame_];
    // Wait until frame using this slot has finished
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &frame.cpuFinished, VK_TRUE,
                                UINT64_MAX));
}

void Renderer::startFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

static glm::mat4 makePerspectiveMatrix(float tan_hfov, float aspect,
                                       float near, float far)
{
    return glm::mat4(1.f / tan_hfov, 0.f, 0.f, 0.f,
                     0.f, -aspect / tan_hfov, 0.f, 0.f,
                     0.f, 0.f, far / (near - far), -1.f,
                     0.f, 0.f, far * near / (near - far), 0.f);
}

static glm::mat4 makeOrthographicMatrix(float height, float aspect,
                                        float near, float far)
{
    float width = height * aspect;
    float rminusl = 2.f * width;
    float bminust = -2.f * height;

    return glm::mat4(2.f / rminusl, 0.f, 0.f, 0.f,
                     0.f, 2.f / bminust, 0.f, 0.f,
                     0.f, 0.f, 1.f / (near - far), 0.f,
                     0.f, 0.f, near / (near - far), 1.f);
}


static pair<glm::mat4, glm::mat4> computeCameraMatrices(
    const EditorCam &cam, float aspect)
{
    glm::mat4 proj;

    if (cam.perspective) {
        float tan_hfov = tanf(glm::radians(cam.fov) / 2.f);
        proj = makePerspectiveMatrix(tan_hfov, aspect, 0.001f,
                                     10000.f);
    } else {
        proj = makeOrthographicMatrix(cam.orthoHeight, aspect, 0.001f,
                                     10000.f);
    }

    glm::mat4 view(1.f);
    view[0][0] = cam.right.x;
    view[1][0] = cam.right.y;
    view[2][0] = cam.right.z;
    view[0][1] = cam.up.x;
    view[1][1] = cam.up.y;
    view[2][1] = cam.up.z;
    view[0][2] = -cam.view.x;
    view[1][2] = -cam.view.y;
    view[2][2] = -cam.view.z;
    view[3][0] = -glm::dot(cam.right, cam.position);
    view[3][1] = -glm::dot(cam.up, cam.position);
    view[3][2] = glm::dot(cam.view, cam.position);

    return {
        proj,
        view,
    };
}

void Renderer::render(Scene *raw_scene, const EditorCam &cam,
                      const FrameConfig &cfg,
                      const OverlayVertex *extra_vertices,
                      const uint32_t *extra_indices,
                      uint32_t num_extra_vertices,
                      uint32_t num_overlay_tri_indices,
                      uint32_t num_overlay_line_indices,
                      uint32_t num_light_indices)
{
    vk::VulkanScene *scene = static_cast<vk::VulkanScene *>(raw_scene);

    Frame &frame = frames_[cur_frame_];
    uint32_t swapchain_idx = present_.acquireNext(dev, frame.swapchainReady);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, frame.cmdPool, 0));
    VkCommandBuffer draw_cmd = frame.drawCmd;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(draw_cmd, &begin_info));

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           default_pipeline_.hdls[0]);

    VkViewport viewport {
        0,
        0,
        (float)fb_dims_.x,
        (float)fb_dims_.y,
        0.f,
        1.f,
    };

    dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

    VkRect2D scissor {
        { 0, 0 },
        { fb_dims_.x, fb_dims_.y },
    };

    dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 default_pipeline_.layout, 0, 1,
                                 &frame.defaultShaderSet, 0, nullptr);

    auto [proj, view] =
        computeCameraMatrices(cam, (float)fb_dims_.x / (float)fb_dims_.y);

    DrawPushConst draw_const {
        proj,
        view,
    };

    dev.dt.cmdPushConstants(draw_cmd, default_pipeline_.layout,
                            VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(DrawPushConst), &draw_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 default_pipeline_.layout, 1, 1,
                                 &scene_set_, 0, nullptr);
    dev.dt.cmdBindIndexBuffer(draw_cmd, scene->data.buffer,
                              scene->indexOffset, VK_INDEX_TYPE_UINT32);

    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = frame.fb.hdl;
    render_pass_info.clearValueCount = fb_clear_.size();
    render_pass_info.pClearValues = fb_clear_.data();
    render_pass_info.renderArea.offset = {
        0, 0,
    };
    render_pass_info.renderArea.extent = {
        fb_dims_.x, fb_dims_.y,
    };

    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    uint32_t fake_inst_idx = 0;
    for (int inst_idx = 0; inst_idx < (int)scene->envInit.defaultInstances.size();
         inst_idx++) {
        const InstanceTransform &txfm =
            scene->envInit.defaultTransforms[inst_idx];
        const ObjectInstance &instance =
            scene->envInit.defaultInstances[inst_idx];
        const ObjectInfo &obj = scene->objectInfo[instance.objectIndex];

        for (int mesh_idx = 0; mesh_idx < (int)obj.numMeshes; mesh_idx++) {
            const uint32_t mat_idx =
                scene->envInit.defaultInstanceMaterials[
                instance.materialOffset + mesh_idx];

            const MeshInfo &mesh = scene->meshInfo[mesh_idx + obj.meshIndex];

            dev.dt.cmdDrawIndexed(draw_cmd, mesh.numTriangles * 3,
                                  1, mesh.indexOffset, 0, fake_inst_idx);
            frame.renderInput.transformPtr[fake_inst_idx] = txfm;
            frame.renderInput.matIndices[fake_inst_idx] = mat_idx;
            fake_inst_idx++;
        }
    }

    for (int i = 0; i < (int)num_extra_vertices; i++) {
        frame.renderInput.overlayVertices[i] = extra_vertices[i];
    }

    uint32_t *extra_idx_ptr = frame.renderInput.overlayIndices;
    int total_overlay_indices =
        num_overlay_line_indices + num_overlay_tri_indices +
        num_light_indices;
    assert(total_overlay_indices <
           (int)InternalConfig::initMaxOverlayIndices);
    assert(num_extra_vertices < InternalConfig::initMaxOverlayVertices);

    for (int i = 0; i < (int)total_overlay_indices; i++) {
        extra_idx_ptr[i] = extra_indices[i];
    }

    if (total_overlay_indices > 0) {
        glm::vec4 fake;
        dev.dt.cmdBindPipeline(draw_cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            overlay_pipeline_.hdls[
                InternalConfig::navPipelineIdxs.filled]);
        dev.dt.cmdBindDescriptorSets(draw_cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            overlay_pipeline_.layout, 0, 1,
            &frame.overlayShaderSet, 0, nullptr);

        dev.dt.cmdPushConstants(draw_cmd, overlay_pipeline_.layout,
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT,
            offsetof(NavmeshPushConst, color),
            sizeof(glm::vec4), &fake);

        dev.dt.cmdBindIndexBuffer(draw_cmd,
            frame.renderInput.buffer.buffer,
            frame.renderInput.overlayIndexOffset,
            VK_INDEX_TYPE_UINT32);

        if (!cfg.wireframeOnly) {
            dev.dt.cmdDrawIndexed(draw_cmd, num_overlay_tri_indices,
                                 1, 0, 0, 0);

        }

        dev.dt.cmdDrawIndexed(draw_cmd, num_light_indices,
            1, num_overlay_tri_indices + num_overlay_line_indices,
            0, 0);

        VkPipeline wire_pipeline = overlay_pipeline_.hdls[
            cfg.linesNoDepthTest ?
                InternalConfig::navPipelineIdxs.wireNoDepth :
                InternalConfig::navPipelineIdxs.wireWithDepth];

        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               wire_pipeline);
        dev.dt.cmdSetLineWidth(draw_cmd, cfg.lineWidth);
        dev.dt.cmdBindDescriptorSets(draw_cmd,
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     overlay_pipeline_.layout, 0, 1,
                                     &frame.overlayShaderSet, 0, nullptr);

        dev.dt.cmdPushConstants(draw_cmd, overlay_pipeline_.layout,
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT,
            offsetof(NavmeshPushConst, color),
            sizeof(glm::vec4), &fake);

        dev.dt.cmdDrawIndexed(draw_cmd, num_overlay_line_indices,
                             1, num_overlay_tri_indices, 0, 0);
    }
    
    dev.dt.cmdEndRenderPass(draw_cmd);

    render_pass_info.renderPass = gui_render_pass_;
    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), draw_cmd);

    dev.dt.cmdEndRenderPass(draw_cmd);

    VkImage swapchain_img = present_.getImage(swapchain_idx);

    array<VkImageMemoryBarrier, 1> blit_prepare {{
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_img,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        }
    }};

    dev.dt.cmdPipelineBarrier(draw_cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr, 0, nullptr,
        blit_prepare.size(), blit_prepare.data());

    VkImageBlit blit_region {
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fb_dims_.x, (int32_t)fb_dims_.y, 1 },
        },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fb_dims_.x, (int32_t)fb_dims_.y, 1 },
        },
    };

    dev.dt.cmdBlitImage(draw_cmd,
                        frame.fb.colorAttachment.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain_img,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit_region,
                        VK_FILTER_NEAREST);

    VkImageMemoryBarrier swapchain_prepare {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        swapchain_img,
        {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, 0, 1
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0,
                              0, nullptr, 0, nullptr,
                              1, &swapchain_prepare);

    REQ_VK(dev.dt.endCommandBuffer(draw_cmd));

    frame.renderInput.buffer.flush(dev);

    VkPipelineStageFlags swapchain_wait_flag =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        1,
        &frame.swapchainReady,
        &swapchain_wait_flag,
        1,
        &draw_cmd,
        1,
        &frame.renderFinished,
    };

    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &frame.cpuFinished));
    REQ_VK(dev.dt.queueSubmit(render_queue_, 1, &gfx_submit,
                              frame.cpuFinished));

    present_.present(dev, swapchain_idx, present_wrapper_,
                     1, &frame.renderFinished);

    cur_frame_ = (cur_frame_ + 1) % frames_.size();
}

void Renderer::waitForIdle()
{
    dev.dt.deviceWaitIdle(dev.hdl);
}

}
}
