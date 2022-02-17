#include "render.hpp"
#include <vulkan/vulkan_core.h>

#include "scene.hpp"

#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;

namespace RLpbr {
namespace vk {

static InitConfig getInitConfig(const RenderConfig &cfg, bool validate)
{
    bool use_zsobol =  uint64_t(cfg.spp) * uint64_t(cfg.imgWidth) *
        uint64_t(cfg.imgHeight) < uint64_t(~0u);

    bool need_present = false;
    char *fake_present = getenv("VK_FAKE_PRESENT");
    if (fake_present && fake_present[0] == '1') {
        need_present = true;
    }

    return InitConfig {
        use_zsobol,
        false,
        validate,
        need_present,
    };
}

static ParamBufferConfig getParamBufferConfig(uint32_t batch_size,
                                              const MemoryAllocator &alloc)
{
    ParamBufferConfig cfg {};

    cfg.totalTransformBytes =
        sizeof(InstanceTransform) * VulkanConfig::max_instances;

    VkDeviceSize cur_offset = cfg.totalTransformBytes;

    cfg.materialIndicesOffset = alloc.alignStorageBufferOffset(cur_offset);

    cfg.totalMaterialIndexBytes =
        sizeof(uint32_t) * VulkanConfig::max_instances;

    cur_offset = cfg.materialIndicesOffset + cfg.totalMaterialIndexBytes;

    cfg.lightsOffset = alloc.alignStorageBufferOffset(cur_offset);
    cfg.totalLightParamBytes = sizeof(PackedLight) * VulkanConfig::max_lights;

    cur_offset = cfg.lightsOffset + cfg.totalLightParamBytes;

    cfg.envOffset = alloc.alignStorageBufferOffset(cur_offset);
    cfg.totalEnvParamBytes = sizeof(PackedEnv) * batch_size;

    cur_offset = cfg.envOffset + cfg.totalEnvParamBytes;

    // Ensure that full block is aligned to maximum requirement
    cfg.totalParamBytes =  alloc.alignStorageBufferOffset(cur_offset);

    return cfg;
}

static FramebufferConfig getFramebufferConfig(const RenderConfig &cfg)
{
    uint32_t batch_size = cfg.batchSize;

    uint32_t minibatch_size =
        max(batch_size / VulkanConfig::minibatch_divisor, batch_size);
    assert(batch_size % minibatch_size == 0);

    uint32_t batch_fb_images_wide = ceil(sqrt(batch_size));
    while (batch_size % batch_fb_images_wide != 0) {
        batch_fb_images_wide++;
    }

    uint32_t minibatch_fb_images_wide;
    uint32_t minibatch_fb_images_tall;
    if (batch_fb_images_wide >= minibatch_size) {
        assert(batch_fb_images_wide % minibatch_size == 0);
        minibatch_fb_images_wide = minibatch_size;
        minibatch_fb_images_tall = 1;
    } else {
        minibatch_fb_images_wide = batch_fb_images_wide;
        minibatch_fb_images_tall = minibatch_size / batch_fb_images_wide;
    }

    assert(minibatch_fb_images_wide * minibatch_fb_images_tall ==
           minibatch_size);

    uint32_t batch_fb_images_tall = (batch_size / batch_fb_images_wide);
    assert(batch_fb_images_wide * batch_fb_images_tall == batch_size);

    uint32_t batch_fb_width = cfg.imgWidth * batch_fb_images_wide;
    uint32_t batch_fb_height = cfg.imgHeight * batch_fb_images_tall;

    uint32_t pixels_per_batch = batch_fb_width * batch_fb_height;

    uint32_t output_bytes = 4 * sizeof(uint16_t) * pixels_per_batch;
    uint32_t normal_bytes = 3 * sizeof(uint16_t) * pixels_per_batch;
    uint32_t albedo_bytes = 4 * sizeof(uint16_t) * pixels_per_batch;
    uint32_t reservoir_bytes = sizeof(Reservoir) * pixels_per_batch;

    // FIXME: dedup this with the EXPOSURE_RES_X code
    uint32_t illuminance_width =
        divideRoundUp(cfg.imgWidth, VulkanConfig::localWorkgroupX);
    uint32_t illuminance_height =
        divideRoundUp(cfg.imgWidth, VulkanConfig::localWorkgroupY);

    uint32_t illuminance_bytes =
        sizeof(float) * illuminance_width * illuminance_height * batch_size;

    return FramebufferConfig {
        cfg.imgWidth,
        cfg.imgHeight,
        minibatch_size,
        minibatch_fb_images_wide,
        minibatch_fb_images_tall,
        batch_fb_images_wide,
        batch_fb_images_tall,
        batch_fb_width,
        batch_fb_height,
        output_bytes,
        normal_bytes,
        albedo_bytes,
        reservoir_bytes,
        illuminance_bytes,
    };
}

static RenderState makeRenderState(const DeviceState &dev,
                                   const RenderConfig &cfg,
                                   const InitConfig &init_cfg)
{
    VkSampler repeat_sampler =
        makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    VkSampler clamp_sampler =
        makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    auto log2Int = [](uint32_t v) {
        return 32 - __builtin_clz(v) - 1;
    };

    // In optix these are just C++ constexprs
    // Compute a bunch of sampler variables ahead of time
    uint32_t log2_spp = log2Int(cfg.spp);
    bool is_odd_power2 = log2_spp & 1;

    uint32_t index_shift = is_odd_power2 ? (log2_spp + 1) : log2_spp;

    uint32_t num_index_digits_base4 =
        log2Int(max(cfg.imgWidth, cfg.imgHeight) - 1) + 1 + (log2_spp + 1) / 2;

    string sampling_define;
    if (num_index_digits_base4 * 2 <= 32) {
        sampling_define = "ZSOBOL_SAMPLING";
    } else {
        cerr << "Warning: Not enough bits for ZSobol morton code.\n"
             << "Falling back to uniform sampling." << endl;

        sampling_define = "UNIFORM_SAMPLING";
    }

    if (cfg.flags & RenderFlags::ForceUniform) {
        sampling_define = "UNIFORM_SAMPLING";
    }

    float inv_sqrt_spp = 1.f / sqrtf(float(cfg.spp));
    ostringstream float_conv;
    float_conv.precision(9);
    float_conv << fixed << inv_sqrt_spp;

    uint32_t num_workgroups_x = divideRoundUp(cfg.imgWidth,
                                              VulkanConfig::localWorkgroupX);

    uint32_t num_workgroups_y = divideRoundUp(cfg.imgHeight,
                                              VulkanConfig::localWorkgroupY);

    uint32_t exposure_res_x = num_workgroups_x;
    uint32_t exposure_res_y = num_workgroups_y;
    
    vector<string> rt_defines {
        string("SPP (") + to_string(cfg.spp) + "u)",
        string("INV_SQRT_SPP (") + float_conv.str() + "f)",
        string("MAX_DEPTH (") + to_string(cfg.maxDepth) + "u)",
        string("RES_X (") + to_string(cfg.imgWidth) + "u)",
        string("RES_Y (") + to_string(cfg.imgHeight) + "u)",
        string("EXPOSURE_RES_X (") + to_string(exposure_res_x) + "u)",
        string("EXPOSURE_RES_Y (") + to_string(exposure_res_y) + "u)",
        string("BATCH_SIZE (") + to_string(cfg.batchSize) + "u)",
        sampling_define,
        string("ZSOBOL_NUM_BASE4 (") + to_string(num_index_digits_base4) + "u)",
        string("ZSOBOL_INDEX_SHIFT (") + to_string(index_shift) + "u)",
    };

    if (is_odd_power2) {
        rt_defines.push_back("ZSOBOL_ODD_POWER");
    }

    if (cfg.spp == 1) {
        rt_defines.push_back("ONE_SAMPLE");
    }

    if (cfg.maxDepth == 1) {
        rt_defines.push_back("PRIMARY_ONLY");
    }

    if (cfg.flags & RenderFlags::AuxiliaryOutputs) {
         rt_defines.emplace_back("AUXILIARY_OUTPUTS");
    }

    if (cfg.flags & RenderFlags::Tonemap) {
         rt_defines.emplace_back("TONEMAP");
    }

    if (cfg.clampThreshold > 0.f) {
        rt_defines.emplace_back(string("-DINDIRECT_CLAMP (") +
            to_string(cfg.clampThreshold) + "f)");
    }

    const char *rt_name;
    switch (cfg.mode) {
        case RenderMode::PathTracer:
            rt_name = "pathtracer.comp";
            break;
        case RenderMode::Biased:
            rt_name = "basic.comp";
            break;
    }

    // Give 9 digits of precision for lossless conversion
    auto floatToString = [](float v) {
        ostringstream strm;
        strm.precision(9);
        strm << fixed << v;

        return strm.str();
    };
    
    // 22 is probably overkill but makes for 32 stops
    float min_log_luminance = -10.f;
    float max_log_luminance = 22.f;
    float min_luminance = exp2(min_log_luminance);

    float log_luminance_range = max_log_luminance - min_log_luminance;
    float inv_log_luminance_range = 1.f / log_luminance_range;

    int num_exposure_bins = 128;
    float exposure_bias = 0.f;

    uint32_t thread_elems_x = divideRoundUp(exposure_res_x,
                                            VulkanConfig::localWorkgroupX);
    uint32_t thread_elems_y = divideRoundUp(exposure_res_y,
                                            VulkanConfig::localWorkgroupY);

    vector<string> exposure_defines {
        string("NUM_BINS (") + to_string(num_exposure_bins) + ")",
        string("LOG_LUMINANCE_RANGE (") +
            floatToString(log_luminance_range) + "f)",
        string("INV_LOG_LUMINANCE_RANGE (") +
            floatToString(inv_log_luminance_range) + "f)",
        string("MIN_LOG_LUMINANCE (") +
            floatToString(min_log_luminance) + "f)",
        string("MAX_LOG_LUMINANCE (") +
            floatToString(max_log_luminance) + "f)",
        string("EXPOSURE_BIAS (") +
            floatToString(exposure_bias) + "f)",
        string("MIN_LUMINANCE (") +
            floatToString(min_luminance) + "f)",
        string("EXPOSURE_RES_X (") + to_string(exposure_res_x) + "u)",
        string("EXPOSURE_RES_Y (") + to_string(exposure_res_y) + "u)",
        string("THREAD_ELEMS_X (") + to_string(thread_elems_x) + "u)",
        string("THREAD_ELEMS_Y (") + to_string(thread_elems_y) + "u)",
    };

    vector<string> tonemap_defines {
        string("RES_X (") + to_string(cfg.imgWidth) + "u)",
        string("RES_Y (") + to_string(cfg.imgHeight) + "u)",
    };

    if (init_cfg.validate) {
        rt_defines.push_back("VALIDATE");
        exposure_defines.push_back("VALIDATE");
        tonemap_defines.push_back("VALIDATE");
    }

    ShaderPipeline::initCompiler();

    ShaderPipeline rt(dev,
        { rt_name },
        {
            {0, 4, repeat_sampler, 1, 0},
            {0, 5, clamp_sampler, 1, 0},
            {
                1, 1, VK_NULL_HANDLE,
                VulkanConfig::max_scenes * (1 + VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material),
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
            },
        },
        rt_defines,
        STRINGIFY(SHADER_DIR));

    ShaderPipeline exposure(dev,
        { "exposure_histogram.comp" },
        {},
        exposure_defines,
        STRINGIFY(SHADER_DIR));

    ShaderPipeline tonemap(dev,
        { "tonemap.comp" },
        {},
        tonemap_defines,
        STRINGIFY(SHADER_DIR));

    return RenderState {
        repeat_sampler,
        clamp_sampler,
        move(rt),
        move(exposure),
        move(tonemap),
    };
}

static RenderPipelines makePipelines(const DeviceState &dev,
                                     const RenderState &render_state)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(RTPushConstant),
    };
    // Layout configuration

    array<VkDescriptorSetLayout, 2> desc_layouts {{
        render_state.rt.getLayout(0),
        render_state.rt.getLayout(1),
    }};

    VkPipelineLayoutCreateInfo pt_layout_info;
    pt_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pt_layout_info.pNext = nullptr;
    pt_layout_info.flags = 0;
    pt_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    pt_layout_info.pSetLayouts = desc_layouts.data();
    pt_layout_info.pushConstantRangeCount = 1;
    pt_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout pt_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &pt_layout_info, nullptr,
                                       &pt_layout));

    VkDescriptorSetLayout exposure_set_layout =
        render_state.exposure.getLayout(0);

    VkPipelineLayoutCreateInfo exposure_layout_info;
    exposure_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    exposure_layout_info.pNext = nullptr;
    exposure_layout_info.flags = 0;
    exposure_layout_info.setLayoutCount = 1;
    exposure_layout_info.pSetLayouts = &exposure_set_layout;
    exposure_layout_info.pushConstantRangeCount = 0;
    exposure_layout_info.pPushConstantRanges = nullptr;

    VkPipelineLayout exposure_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &exposure_layout_info, nullptr,
                                       &exposure_layout));

    VkDescriptorSetLayout tonemap_set_layout =
        render_state.tonemap.getLayout(0);

    VkPipelineLayoutCreateInfo tonemap_layout_info;
    tonemap_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tonemap_layout_info.pNext = nullptr;
    tonemap_layout_info.flags = 0;
    tonemap_layout_info.setLayoutCount = 1;
    tonemap_layout_info.pSetLayouts = &tonemap_set_layout;
    tonemap_layout_info.pushConstantRangeCount = 0;
    tonemap_layout_info.pPushConstantRanges = nullptr;

    VkPipelineLayout tonemap_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &tonemap_layout_info, nullptr,
                                       &tonemap_layout));

    array<VkComputePipelineCreateInfo, 3> compute_infos;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size;
    subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    subgroup_size.pNext = nullptr;
    subgroup_size.requiredSubgroupSize = VulkanConfig::subgroupSize;

    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        &subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        render_state.rt.getShader(0),
        "main",
        nullptr,
    };
    compute_infos[0].layout = pt_layout;
    compute_infos[0].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[0].basePipelineIndex = -1;

    compute_infos[1].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[1].pNext = nullptr;
    compute_infos[1].flags = 0;
    compute_infos[1].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        &subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        render_state.exposure.getShader(0),
        "main",
        nullptr,
    };
    compute_infos[1].layout = exposure_layout;
    compute_infos[1].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[1].basePipelineIndex = -1;

    compute_infos[2].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[2].pNext = nullptr;
    compute_infos[2].flags = 0;
    compute_infos[2].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        &subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        render_state.tonemap.getShader(0),
        "main",
        nullptr,
    };
    compute_infos[2].layout = tonemap_layout;
    compute_infos[2].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[2].basePipelineIndex = -1;

    array<VkPipeline, compute_infos.size()> pipelines;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache,
                                         compute_infos.size(),
                                         compute_infos.data(), nullptr,
                                         pipelines.data()));

    return RenderPipelines {
        pipeline_cache,
        PipelineState {
            pt_layout,
            pipelines[0],
        },
        PipelineState {
            exposure_layout,
            pipelines[1],
        },
        PipelineState {
            tonemap_layout,
            pipelines[2],
        },
    };
}

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        const VulkanBackend::Config &cfg,
                                        const FramebufferConfig &fb_cfg,
                                        MemoryAllocator &alloc)
{
    vector<LocalBuffer> outputs;
    vector<VkDeviceMemory> backings;
    vector<CudaImportedBuffer> exported;

    uint32_t num_buffers = 1;
    uint32_t num_exported_buffers = 1;

    if (cfg.auxiliaryOutputs) {
        num_buffers += 2;
        num_exported_buffers += 2;
    }

    if (cfg.tonemap) {
        num_buffers += 1;
    }

    outputs.reserve(num_buffers);
    backings.reserve(num_buffers);
    exported.reserve(num_exported_buffers);

    auto [main_buffer, main_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.outputBytes);

    outputs.emplace_back(move(main_buffer));
    backings.emplace_back(move(main_mem));
    exported.emplace_back(dev, cfg.gpuID, backings.back(),
                          fb_cfg.outputBytes);

    if (cfg.auxiliaryOutputs) {
        auto [normal_buffer, normal_mem] =
            alloc.makeDedicatedBuffer(fb_cfg.normalBytes);

        outputs.emplace_back(move(normal_buffer));
        backings.emplace_back(move(normal_mem));
        exported.emplace_back(dev, cfg.gpuID, backings.back(),
                              fb_cfg.normalBytes);

        auto [albedo_buffer, albedo_mem] =
            alloc.makeDedicatedBuffer(fb_cfg.albedoBytes);

        outputs.emplace_back(move(albedo_buffer));
        backings.emplace_back(move(albedo_mem));
        exported.emplace_back(dev, cfg.gpuID, backings.back(),
                              fb_cfg.albedoBytes);
    }

    if (cfg.tonemap) {
        auto [illum_buffer, illum_mem] =
            alloc.makeDedicatedBuffer(fb_cfg.illuminanceBytes);

        outputs.emplace_back(move(illum_buffer));
        backings.emplace_back(move(illum_mem));
    }

    vector<LocalBuffer> reservoirs;
    vector<VkDeviceMemory> reservoir_mem;

    auto [prev_reservoir_buffer, prev_reservoir_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.reservoirBytes);

    reservoirs.emplace_back(move(prev_reservoir_buffer));
    reservoir_mem.emplace_back(move(prev_reservoir_mem));

    auto [cur_reservoir_buffer, cur_reservoir_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.reservoirBytes);

    reservoirs.emplace_back(move(cur_reservoir_buffer));
    reservoir_mem.emplace_back(move(cur_reservoir_mem));

    return FramebufferState {
        move(outputs),
        move(backings),
        move(exported),
        move(reservoirs),
        move(reservoir_mem),
    };
}

static PerBatchState makePerBatchState(const DeviceState &dev,
                                       const FramebufferConfig &fb_cfg,
                                       const FramebufferState &fb,
                                       const ParamBufferConfig &param_cfg,
                                       HostBuffer &param_buffer,
                                       FixedDescriptorPool &&rt_pool,
                                       FixedDescriptorPool &&exposure_pool,
                                       FixedDescriptorPool &&tonemap_pool,
                                       const BSDFPrecomputed &precomp_tex,
                                       bool auxiliary_outputs,
                                       bool tonemap,
                                       bool need_present)
{
    array rt_sets {
        rt_pool.makeSet(),
        rt_pool.makeSet(),
    };

    VkDescriptorSet exposure_set = exposure_pool.makeSet();
    VkDescriptorSet tonemap_set = tonemap_pool.makeSet();

    VkCommandPool cmd_pool = makeCmdPool(dev, dev.computeQF);
    VkCommandBuffer render_cmd = makeCmdBuffer(dev, cmd_pool);

    half *output_buffer = (half *)((char *)fb.exported[0].getDevicePointer());

    half *normal_buffer = nullptr, *albedo_buffer = nullptr;
    if (auxiliary_outputs) {
        normal_buffer = (half *)((char *)fb.exported[1].getDevicePointer());

        albedo_buffer = (half *)((char *)fb.exported[2].getDevicePointer());
    }

    vector<VkWriteDescriptorSet> desc_set_updates;

    uint8_t *base_ptr =
        reinterpret_cast<uint8_t *>(param_buffer.ptr);

    InstanceTransform *transform_ptr =
        reinterpret_cast<InstanceTransform *>(base_ptr);

    uint32_t *material_ptr = reinterpret_cast<uint32_t *>(
        base_ptr + param_cfg.materialIndicesOffset);

    PackedLight *light_ptr =
        reinterpret_cast<PackedLight *>(base_ptr + param_cfg.lightsOffset);

    PackedEnv *env_ptr =
        reinterpret_cast<PackedEnv *>(base_ptr + param_cfg.envOffset);

    DescriptorUpdates desc_updates(14);

    VkDescriptorBufferInfo transform_info {
        param_buffer.buffer,
        0,
        param_cfg.totalTransformBytes,
    };

    VkDescriptorBufferInfo mat_info {
        param_buffer.buffer,
        param_cfg.materialIndicesOffset,
        param_cfg.totalMaterialIndexBytes,
    };

    VkDescriptorBufferInfo light_info {
        param_buffer.buffer,
        param_cfg.lightsOffset,
        param_cfg.totalLightParamBytes,
    };

    VkDescriptorBufferInfo env_info {
        param_buffer.buffer,
        param_cfg.envOffset,
        param_cfg.totalEnvParamBytes,
    };

    VkDescriptorImageInfo diffuse_avg_info {
        VK_NULL_HANDLE,
        precomp_tex.msDiffuseAverage.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo diffuse_dir_info {
        VK_NULL_HANDLE,
        precomp_tex.msDiffuseDirectional.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo ggx_avg_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXAverage.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo ggx_dir_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXDirectional.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo ggx_inv_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXInverse.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorBufferInfo out_info {
        fb.outputs[0].buffer,
        0,
        fb_cfg.outputBytes,
    };

    for (int i = 0; i < (int)rt_sets.size(); i++) {
        desc_updates.buffer(rt_sets[i], &transform_info, 0,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        desc_updates.buffer(rt_sets[i], &mat_info, 1,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        desc_updates.buffer(rt_sets[i], &light_info, 2,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        desc_updates.buffer(rt_sets[i], &env_info, 3,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        desc_updates.textures(rt_sets[i], &diffuse_avg_info, 1, 6);
        desc_updates.textures(rt_sets[i], &diffuse_dir_info, 1, 7);
        desc_updates.textures(rt_sets[i], &ggx_avg_info, 1, 8);
        desc_updates.textures(rt_sets[i], &ggx_dir_info, 1, 9);
        desc_updates.textures(rt_sets[i], &ggx_inv_info, 1, 10);
        desc_updates.buffer(rt_sets[i], &out_info, 13,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    VkDescriptorBufferInfo cur_reservoir_info {
        fb.reservoirs[0].buffer,
        0,
        fb_cfg.reservoirBytes,
    };

    desc_updates.buffer(rt_sets[0], &cur_reservoir_info, 11,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    desc_updates.buffer(rt_sets[1], &cur_reservoir_info, 12,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo prev_reservoir_info {
        fb.reservoirs[1].buffer,
        0,
        fb_cfg.reservoirBytes,
    };

    desc_updates.buffer(rt_sets[0], &prev_reservoir_info, 12,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    desc_updates.buffer(rt_sets[1], &prev_reservoir_info, 11,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo normal_info;
    VkDescriptorBufferInfo albedo_info;

    if (auxiliary_outputs) {
        normal_info = {
            fb.outputs[1].buffer,
            0,
            fb_cfg.normalBytes,
        };

        albedo_info = {
            fb.outputs[2].buffer,
            0,
            fb_cfg.albedoBytes,
        };

        for (int i = 0; i < (int)rt_sets.size(); i++) {
            desc_updates.buffer(rt_sets[i], &normal_info, 14,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            desc_updates.buffer(rt_sets[i], &albedo_info, 15,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    }

    VkDescriptorBufferInfo illuminance_info {
        fb.outputs.back().buffer,
        0,
        fb_cfg.illuminanceBytes,
    };

    if (tonemap) {
        for (int i = 0; i < (int)rt_sets.size(); i++) {
            desc_updates.buffer(rt_sets[i], &illuminance_info, 16,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    }

    desc_updates.buffer(exposure_set, &illuminance_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    desc_updates.buffer(tonemap_set, &illuminance_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    desc_updates.buffer(tonemap_set, &out_info, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    desc_updates.update(dev);

    return PerBatchState {
        makeFence(dev),
        need_present ? makeBinarySemaphore(dev) : VK_NULL_HANDLE,
        need_present ? makeBinarySemaphore(dev) : VK_NULL_HANDLE,
        cmd_pool,
        render_cmd,
        output_buffer,
        normal_buffer,
        albedo_buffer,
        move(rt_pool),
        move(rt_sets),
        move(exposure_pool),
        exposure_set,
        move(tonemap_pool),
        tonemap_set,
        transform_ptr,
        material_ptr,
        light_ptr,
        env_ptr,
    };
}

static BSDFPrecomputed loadPrecomputedTextures(const DeviceState &dev,
    MemoryAllocator &alloc,
    QueueState &render_queue,
    uint32_t qf_idx)
{
    const string dir = STRINGIFY(RLPBR_DATA_DIR);

    vector<pair<string, vector<uint32_t>>> names_and_dims {
        { "diffuse_avg_albedo.bin", { 16, 16 } },
        { "diffuse_dir_albedo.bin", { 16, 16, 16 } },
        { "ggx_avg_albedo.bin", { 32 } },
        { "ggx_dir_albedo.bin", { 32, 32 } },
        { "ggx_dir_inv.bin", { 128, 32 } },
    };

    vector<DynArray<float>> raw_data;
    raw_data.reserve(names_and_dims.size());

    // This code assumes all the LUTs are stored as R32 (4 byte alignment)
    vector<uint32_t> num_elems;
    num_elems.reserve(names_and_dims.size());
    vector<size_t> offsets;
    offsets.reserve(names_and_dims.size());
    vector<LocalTexture> imgs;
    imgs.reserve(names_and_dims.size());

    VkFormat fmt = alloc.getTextureFormat(TextureFormat::R32_SFLOAT);

    vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(imgs.size());

    size_t cur_tex_offset = 0;
    for (const auto &[name, dims] : names_and_dims) {
        uint32_t total_elems = 1;
        for (uint32_t d : dims) {
            total_elems *= d;
        }
        num_elems.push_back(total_elems);

        LocalTexture tex;
        TextureRequirements tex_reqs;
        if (dims.size() == 1) {
            tie(tex, tex_reqs) = alloc.makeTexture1D(dims[0], 1, fmt);
        } else if (dims.size() == 2) {
            tie(tex, tex_reqs) = alloc.makeTexture2D(dims[0], dims[1], 1, fmt);
        } else if (dims.size() == 3) {
            tie(tex, tex_reqs) =
                alloc.makeTexture3D(dims[0], dims[1], dims[2], 1, fmt);
        } else {
            assert(false);
        }
        imgs.emplace_back(move(tex));

        cur_tex_offset = alignOffset(cur_tex_offset, tex_reqs.alignment);
        offsets.push_back(cur_tex_offset);

        cur_tex_offset += tex_reqs.size;

        barriers.push_back({
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            imgs.back().image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        });
    }

    size_t total_bytes = cur_tex_offset;

    HostBuffer staging = alloc.makeStagingBuffer(total_bytes);
    optional<VkDeviceMemory> backing_opt = alloc.alloc(total_bytes);

    if (!backing_opt.has_value()) {
        cerr << "Not enough memory to allocate BSDF LUTs" << endl;
        fatalExit();
    }

    VkDeviceMemory backing = backing_opt.value();

    vector<VkImageView> views;
    views.reserve(imgs.size());

    VkCommandPool cmd_pool = makeCmdPool(dev, qf_idx);

    VkCommandBuffer cmd = makeCmdBuffer(dev, cmd_pool);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    for (int i = 0; i < (int)imgs.size(); i++) {
        char *cur = (char *)staging.ptr + offsets[i];

        ifstream file(dir + "/" + names_and_dims[i].first);
        file.read(cur, num_elems[i] * sizeof(float));

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, imgs[i].image, backing,
                                      offsets[i]));

        const auto &dims = names_and_dims[i].second;

        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = imgs[i].image;

        if (dims.size() == 1) {
            view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
        } else if (dims.size() == 2) {
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        } else if (dims.size() == 3) {
            view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
        }

        view_info.format = fmt;
        view_info.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A,
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        };

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));
        views.push_back(view);
    }

    dev.dt.cmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
        0, nullptr, barriers.size(), barriers.data());

    for (int i = 0; i < (int)imgs.size(); i++) {
        const auto &dims = names_and_dims[i].second;

        VkBufferImageCopy copy_info {};
        copy_info.bufferOffset = offsets[i];
        copy_info.imageSubresource.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        copy_info.imageSubresource.mipLevel = 0;
        copy_info.imageSubresource.baseArrayLayer = 0;
        copy_info.imageSubresource.layerCount = 1;
        copy_info.imageExtent.width = dims[0];

        if (dims.size() > 1) {
            copy_info.imageExtent.height = dims[1];
        } else {
            copy_info.imageExtent.height = 1;
        }

        if (dims.size() > 2) {
            copy_info.imageExtent.depth = dims[2];
        } else {
            copy_info.imageExtent.depth = 1;
        }

        dev.dt.cmdCopyBufferToImage(
            cmd, staging.buffer, imgs[i].image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copy_info);

        VkImageMemoryBarrier &barrier = barriers[i];
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    staging.flush(dev);

    dev.dt.cmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
        0, nullptr, barriers.size(), barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 0;
    render_submit.pWaitSemaphores = nullptr;
    render_submit.pWaitDstStageMask = nullptr;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &cmd;

    VkFence fence = makeFence(dev);

    render_queue.submit(dev, 1, &render_submit, fence);

    waitForFenceInfinitely(dev, fence);

    dev.dt.destroyFence(dev.hdl, fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, cmd_pool, nullptr);

    return BSDFPrecomputed {
        backing,
        { imgs[0], views[0] },
        { imgs[1], views[1] },
        { imgs[2], views[2] },
        { imgs[3], views[3] },
        { imgs[4], views[4] },
    };
}

static DynArray<QueueState> initTransferQueues(const RenderConfig &cfg,
                                               const DeviceState &dev)
{
    bool transfer_shared = cfg.numLoaders > dev.numTransferQueues;

    DynArray<QueueState> queues(dev.numTransferQueues);

    for (int i = 0; i < (int)queues.size(); i++) {
        new (&queues[i])
            QueueState(makeQueue(dev, dev.transferQF, i), transfer_shared);
    }

    return queues;
}

static DynArray<QueueState> initGraphicsQueues(const DeviceState &dev)
{
    DynArray<QueueState> queues(dev.numGraphicsQueues);

    for (int i = 0; i < (int)queues.size(); i++) {
        new (&queues[i])
            QueueState(makeQueue(dev, dev.gfxQF, i),
                       i == (int)queues.size() - 1 ? true : false);
    }

    return queues;
}

static DynArray<QueueState> initComputeQueues(const DeviceState &dev)
{
    DynArray<QueueState> queues(dev.numComputeQueues);

    for (int i = 0; i < (int)queues.size(); i++) {
        new (&queues[i]) QueueState(makeQueue(dev, dev.computeQF, i), false);
    }

    return queues;
}

static glm::u32vec3 getLaunchSize(const RenderConfig &cfg)
{
    return {
        divideRoundUp(cfg.imgWidth, VulkanConfig::localWorkgroupX),
        divideRoundUp(cfg.imgHeight, VulkanConfig::localWorkgroupY),
        divideRoundUp(cfg.batchSize, VulkanConfig::localWorkgroupZ),
    };
}

static InstanceState makeInstance(const InitConfig &init_cfg)
{
    if (init_cfg.needPresent) {
        auto get_inst_addr = PresentationState::init();

        return InstanceState(get_inst_addr, init_cfg.validate, true,
                             PresentationState::getInstanceExtensions());

    } else {
        return InstanceState(nullptr, init_cfg.validate, false, {});
    }
}

static DeviceState makeDevice(const InstanceState &inst,
                              const RenderConfig &cfg,
                              const InitConfig &init_cfg)
{
    auto present_callback = init_cfg.needPresent ?
        PresentationState::deviceSupportCallback :
        nullptr;

    return inst.makeDevice(getUUIDFromCudaID(cfg.gpuID), 1, 3, cfg.numLoaders,
                           present_callback);
}

VulkanBackend::VulkanBackend(const RenderConfig &cfg, bool validate)
    : VulkanBackend(cfg, getInitConfig(cfg, validate))
{}

VulkanBackend::VulkanBackend(const RenderConfig &cfg,
                             const InitConfig &init_cfg)
    : cfg_({
          cfg.gpuID,
          cfg.batchSize,
          cfg.numLoaders,
          cfg.maxTextureResolution == 0 ? ~0u :
              cfg.maxTextureResolution,
          cfg.flags & RenderFlags::AuxiliaryOutputs,
          cfg.flags & RenderFlags::Tonemap,
      }),
      inst(makeInstance(init_cfg)),
      dev(makeDevice(inst, cfg, init_cfg)),
      alloc(dev, inst),
      fb_cfg_(getFramebufferConfig(cfg)),
      param_cfg_(getParamBufferConfig(cfg.batchSize, alloc)),
      render_state_(makeRenderState(dev, cfg, init_cfg)),
      pipelines_(makePipelines(dev, render_state_)),
      transfer_queues_(initTransferQueues(cfg, dev)),
      graphics_queues_(initGraphicsQueues(dev)),
      compute_queues_(initComputeQueues(dev)),
      bsdf_precomp_(loadPrecomputedTextures(dev, alloc, compute_queues_[0],
                    dev.computeQF)),
      launch_size_(getLaunchSize(cfg)),
      num_loaders_(0),
      scene_pool_(render_state_.rt.makePool(1, 1)),
      shared_scene_state_(dev, scene_pool_, render_state_.rt.getLayout(1),
                          alloc),
      cur_queue_(0),
      frame_counter_(0),
      present_(init_cfg.needPresent ?
          make_optional<PresentationState>(inst, dev, dev.computeQF,
                                           1, glm::u32vec2(1, 1), true) :
          optional<PresentationState>())
{
    if (init_cfg.needPresent) {
        present_->forceTransition(dev, compute_queues_[0], dev.computeQF);
    }
}

LoaderImpl VulkanBackend::makeLoader()
{
    int loader_idx = num_loaders_.fetch_add(1, memory_order_acq_rel);
    assert(loader_idx < (int)cfg_.maxLoaders);

    auto loader = new VulkanLoader(
        dev, alloc, transfer_queues_[loader_idx % transfer_queues_.size()],
        compute_queues_.back(), shared_scene_state_,
        dev.computeQF, cfg_.maxTextureResolution);

    return makeLoaderImpl<VulkanLoader>(loader);
}

EnvironmentImpl VulkanBackend::makeEnvironment(const shared_ptr<Scene> &scene,
                                               const Camera &cam)
{
    const VulkanScene &vk_scene = *static_cast<VulkanScene *>(scene.get());
    VulkanEnvironment *environment =
        new VulkanEnvironment(dev, alloc, vk_scene, cam);
    return makeEnvironmentImpl<VulkanEnvironment>(environment);
}

RenderBatch::Handle VulkanBackend::makeRenderBatch()
{
    auto deleter = [](void *, BatchBackend *base_ptr) {
        auto ptr = static_cast<VulkanBatch *>(base_ptr);
        delete ptr;
    };

    auto fb = makeFramebuffer(dev, cfg_, fb_cfg_, alloc);

    auto render_input_buffer =
        alloc.makeParamBuffer(param_cfg_.totalParamBytes);

    PerBatchState batch_state = makePerBatchState(
            dev, fb_cfg_, fb, param_cfg_,
            render_input_buffer,
            FixedDescriptorPool(dev, render_state_.rt, 0, 2),
            FixedDescriptorPool(dev, render_state_.exposure, 0, 1),
            FixedDescriptorPool(dev, render_state_.tonemap, 0, 1),
            bsdf_precomp_, cfg_.auxiliaryOutputs, cfg_.tonemap,
            present_.has_value());

    auto backend = new VulkanBatch {
        {},
        move(fb),
        move(render_input_buffer),
        move(batch_state),
        0,
    };

    return RenderBatch::Handle(backend, {nullptr, deleter});
}

static PackedCamera packCamera(const Camera &cam)
{
    PackedCamera packed;

    // FIXME, consider elevating the quaternion representation to
    // external camera interface
    // FIXME: cam.aspectRatio is no longer respected
    glm::quat rotation =
        glm::quat_cast(glm::mat3(cam.right, -cam.up, cam.view));

    packed.rotation =
        glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);

    packed.posAndTanFOV = glm::vec4(cam.position, cam.tanFOV);

    return packed;
}

static VulkanBatch *getVkBatch(RenderBatch &batch)
{
    return static_cast<VulkanBatch *>(batch.getBackend());
}

void VulkanBackend::render(RenderBatch &batch)
{
    VulkanBatch &batch_backend = *getVkBatch(batch);
    Environment *envs = batch.getEnvironments();
    PerBatchState &batch_state = batch_backend.state;

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, batch_state.cmdPool, 0));

    VkCommandBuffer render_cmd = batch_state.renderCmd;
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd, &begin_info));

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelines_.rt.hdl);

    RTPushConstant push_const {
        frame_counter_,
    };

    dev.dt.cmdPushConstants(render_cmd, pipelines_.rt.layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(RTPushConstant),
                            &push_const);

    dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelines_.rt.layout, 0, 1,
                                 &batch_state.rtSets[batch_backend.curBuffer],
                                 0, nullptr);

    // TLAS build
    for (int batch_idx = 0; batch_idx < (int)cfg_.batchSize; batch_idx++) {
        const Environment &env = envs[batch_idx];

        if (env.isDirty()) {
            VulkanEnvironment &env_backend =
                *(VulkanEnvironment *)(env.getBackend());
            const VulkanScene &scene =
                *static_cast<const VulkanScene *>(env.getScene().get());

            env_backend.tlas.build(dev, alloc, env.getInstances(),
                                   env.getTransforms(), env.getInstanceFlags(),
                                   scene.objectInfo, scene.blases, render_cmd);

            env.clearDirty();
        }
    }

    VkMemoryBarrier tlas_barrier;
    tlas_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    tlas_barrier.pNext = nullptr;
    tlas_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    tlas_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    dev.dt.cmdPipelineBarrier(render_cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
        &tlas_barrier, 0, nullptr, 0, nullptr);

    uint32_t inst_offset = 0;
    uint32_t material_offset = 0;
    uint32_t light_offset = 0;

    // Write environment data into linear buffers
    for (int batch_idx = 0; batch_idx < (int)cfg_.batchSize; batch_idx++) {
        Environment &env = envs[batch_idx];
        VulkanEnvironment &env_backend =
            *static_cast<VulkanEnvironment *>(env.getBackend());
        const VulkanScene &scene_backend =
            *static_cast<const VulkanScene *>(env.getScene().get());

        PackedEnv &packed_env = batch_state.envPtr[batch_idx];

        packed_env.cam = packCamera(env.getCamera());
        packed_env.prevCam = packCamera(env_backend.prevCam);
        packed_env.data.x = scene_backend.sceneID->getID();

        // Set prevCam for next iteration
        env_backend.prevCam = env.getCamera();

        const auto &env_transforms = env.getTransforms();
        uint32_t num_instances = env.getNumInstances();
        memcpy(&batch_state.transformPtr[inst_offset], env_transforms.data(),
               sizeof(InstanceTransform) * num_instances);
        inst_offset += num_instances;

        const auto &env_mats = env.getInstanceMaterials();

        memcpy(&batch_state.materialPtr[material_offset], env_mats.data(),
               env_mats.size() * sizeof(uint32_t));

        packed_env.data.y = material_offset;
        material_offset += env_mats.size();

        memcpy(&batch_state.lightPtr[light_offset], env_backend.lights.data(),
               env_backend.lights.size() * sizeof(PackedLight));

        packed_env.data.z = light_offset;
        packed_env.data.w = env_backend.lights.size();
        light_offset += env_backend.lights.size();

        packed_env.tlasAddr = env_backend.tlas.tlasStorageDevAddr;
        packed_env.reservoirGridAddr = env_backend.reservoirGrid.devAddr;
    }

    shared_scene_state_.lock.lock();
    dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelines_.rt.layout, 1, 1,
                                 &shared_scene_state_.descSet, 0, nullptr);
    shared_scene_state_.lock.unlock();

    dev.dt.cmdDispatch(
        render_cmd,
        launch_size_.x,
        launch_size_.y,
        launch_size_.z);

    VkMemoryBarrier comp_barrier;
    comp_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    comp_barrier.pNext = nullptr;
    comp_barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT;
    comp_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    if (cfg_.tonemap) {
        dev.dt.cmdPipelineBarrier(render_cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
            &comp_barrier, 0, nullptr, 0, nullptr);

        dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipelines_.exposure.hdl);

        dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipelines_.exposure.layout, 0, 1,
                                     &batch_state.exposureSet,
                                     0, nullptr);

        dev.dt.cmdDispatch(
            render_cmd,
            1,
            1,
            cfg_.batchSize);

        dev.dt.cmdPipelineBarrier(render_cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
            &comp_barrier, 0, nullptr, 0, nullptr);

        dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipelines_.tonemap.hdl);

        dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipelines_.tonemap.layout, 0, 1,
                                     &batch_state.tonemapSet,
                                     0, nullptr);

        dev.dt.cmdDispatch(
            render_cmd,
            launch_size_.x,
            launch_size_.y,
            launch_size_.z);
    }

    REQ_VK(dev.dt.endCommandBuffer(render_cmd));

    batch_backend.renderInputBuffer.flush(dev);

    VkSubmitInfo render_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &batch_state.renderCmd,
        0,
        nullptr,
    };

    uint32_t swapchain_idx = 0;
    if (present_.has_value()) {
        render_submit.pSignalSemaphores = &batch_state.renderSignal;
        render_submit.signalSemaphoreCount = 1;

        swapchain_idx = present_->acquireNext(dev, batch_state.swapchainReady);
    }

    compute_queues_[cur_queue_].submit(
        dev, 1, &render_submit, batch_state.fence);

    if (present_.has_value()) {
        array present_wait_semas {
            batch_state.swapchainReady,
            batch_state.renderSignal,
        };
        present_->present(dev, swapchain_idx, compute_queues_[cur_queue_],
                          present_wait_semas.size(),
                          present_wait_semas.data());
    }

    frame_counter_ += cfg_.batchSize;

    batch_backend.curBuffer = (batch_backend.curBuffer + 1) & 1;
    cur_queue_ = (cur_queue_ + 1) & 1;
}

void VulkanBackend::waitForBatch(RenderBatch &batch)
{
    auto &batch_backend = *getVkBatch(batch);

    VkFence fence = batch_backend.state.fence;
    assert(fence != VK_NULL_HANDLE);
    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);
}

half *VulkanBackend::getOutputPointer(RenderBatch &batch)
{
    auto &batch_backend = *getVkBatch(batch);

    return batch_backend.state.outputBuffer;
}

AuxiliaryOutputs VulkanBackend::getAuxiliaryOutputs(RenderBatch &batch)
{
    auto &batch_backend = *getVkBatch(batch);
    return {
        batch_backend.state.normalBuffer,
        batch_backend.state.albedoBuffer,
    };
}

}
}
