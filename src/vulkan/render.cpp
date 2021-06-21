#include "render.hpp"

#include "scene.hpp"

#include <iostream>

using namespace std;

namespace RLpbr {
namespace vk {

static BackendConfig getBackendConfig(const RenderConfig &cfg, bool validate)
{
    bool use_zsobol =  uint64_t(cfg.spp) * uint64_t(cfg.imgWidth) *
        uint64_t(cfg.imgHeight) < uint64_t(~0u);
    return BackendConfig {
        cfg.doubleBuffered ? 2u : 1u,
        use_zsobol,
        false,
        validate,
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
    cfg.totalParamBytes =
        alloc.alignStorageBufferOffset(cur_offset);

    return cfg;
}

static FramebufferConfig getFramebufferConfig(const RenderConfig &cfg,
                                              const BackendConfig &backend_cfg)
{
    uint32_t batch_size = cfg.batchSize;
    uint32_t num_batches = backend_cfg.numBatches;

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

    uint32_t total_fb_width = batch_fb_width * num_batches;
    uint32_t total_fb_height = batch_fb_height;

    uint64_t output_linear_bytes =
        4 * sizeof(uint16_t) * batch_fb_width * batch_fb_height;
    assert(output_linear_bytes > 0);

    uint64_t normal_linear_bytes =
        3 * sizeof(uint16_t) * batch_fb_width * batch_fb_height;

    uint64_t albedo_linear_bytes =
        4 * sizeof(uint8_t) * batch_fb_width * batch_fb_height;

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
        total_fb_width,
        total_fb_height,
        output_linear_bytes,
        normal_linear_bytes,
        albedo_linear_bytes,
        output_linear_bytes * num_batches,
        normal_linear_bytes * num_batches,
        albedo_linear_bytes * num_batches,
    };
}

static VkSampler makeImmutableSampler(const DeviceState &dev)
{
    VkSampler sampler;

    VkSamplerCreateInfo sampler_info;
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.pNext = nullptr;
    sampler_info.flags = 0;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.mipLodBias = 0;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 0;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    REQ_VK(dev.dt.createSampler(dev.hdl, &sampler_info, nullptr, &sampler));

    return sampler;
}

static RenderState makeRenderState(const DeviceState &dev,
                                   const RenderConfig &cfg,
                                   const BackendConfig &backend_cfg)
{
    VkSampler texture_sampler = makeImmutableSampler(dev);

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

    vector<string> shader_defines {
        string("SPP (") + to_string(cfg.spp) + "u)",
        string("MAX_DEPTH (") + to_string(cfg.maxDepth) + "u)",
        string("RES_X (") + to_string(cfg.imgWidth) + "u)",
        string("RES_Y (") + to_string(cfg.imgHeight) + "u)",
        string("BATCH_SIZE (") + to_string(cfg.batchSize) + "u)",
        sampling_define,
        string("ZSOBOL_NUM_BASE4 (") + to_string(num_index_digits_base4) + "u)",
        string("ZSOBOL_INDEX_SHIFT (") + to_string(index_shift) + "u)",
    };

    if (is_odd_power2) {
        shader_defines.push_back("ZSOBOL_ODD_POWER");
    }

    if (cfg.spp == 1) {
        shader_defines.push_back("ONE_SAMPLE");
    }

    if (cfg.maxDepth == 1) {
        shader_defines.push_back("PRIMARY_ONLY");
    }

    if (cfg.auxiliaryOutputs) {
         shader_defines.emplace_back("AUXILIARY_OUTPUTS");
    }

    if (cfg.clampThreshold > 0.f) {
        shader_defines.emplace_back(string("-DINDIRECT_CLAMP (") +
            to_string(cfg.clampThreshold) + "f)");
    }

    ShaderPipeline::initCompiler();

    ShaderPipeline shader(dev, {"pathtracer.comp"},
        {
            {0, 4, texture_sampler, 1, 0},
            {1, 2, VK_NULL_HANDLE,
                VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
        },
        shader_defines);

    FixedDescriptorPool desc_pool(dev, shader, 0, backend_cfg.numBatches);

    return RenderState {
        texture_sampler,
        move(shader),
        move(desc_pool),
    };
}

static PipelineState makePipeline(const DeviceState &dev,
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

    VkComputePipelineCreateInfo pt_compute_info;
    pt_compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pt_compute_info.pNext = nullptr;
    pt_compute_info.flags = 0;
    pt_compute_info.stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        render_state.rt.getShader(0),
        "main",
        nullptr,
    };
    pt_compute_info.layout = pt_layout;
    pt_compute_info.basePipelineHandle = VK_NULL_HANDLE;
    pt_compute_info.basePipelineIndex = -1;

    VkPipeline pt_pipeline;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache, 1,
                                         &pt_compute_info, nullptr,
                                         &pt_pipeline));

    return PipelineState {
        pipeline_cache,
        RTPipelineState {
            pt_layout,
            pt_pipeline,
        },
    };
}

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        const RenderConfig &cfg,
                                        const FramebufferConfig &fb_cfg,
                                        MemoryAllocator &alloc)
{
    vector<LocalBuffer> outputs;
    vector<VkDeviceMemory> backings;
    vector<CudaImportedBuffer> exported;

    if (cfg.auxiliaryOutputs) {
        outputs.reserve(3);
        backings.reserve(3);
        exported.reserve(3);
    } else {
        outputs.reserve(1);
        backings.reserve(1);
        exported.reserve(1);
    }

    auto [main_buffer, main_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.totalLinearOutputBytes);

    outputs.emplace_back(move(main_buffer));
    backings.emplace_back(move(main_mem));
    exported.emplace_back(dev, cfg.gpuID, backings.back(),
                          fb_cfg.totalLinearOutputBytes);

    if (cfg.auxiliaryOutputs) {
        auto [normal_buffer, normal_mem] =
            alloc.makeDedicatedBuffer(fb_cfg.totalLinearNormalBytes);

        outputs.emplace_back(move(normal_buffer));
        backings.emplace_back(move(normal_mem));
        exported.emplace_back(dev, cfg.gpuID, backings.back(),
                              fb_cfg.totalLinearNormalBytes);

        auto [albedo_buffer, albedo_mem] =
            alloc.makeDedicatedBuffer(fb_cfg.totalLinearAlbedoBytes);

        outputs.emplace_back(move(albedo_buffer));
        backings.emplace_back(move(albedo_mem));
        exported.emplace_back(dev, cfg.gpuID, backings.back(),
                              fb_cfg.totalLinearAlbedoBytes);
    }

    return FramebufferState {
        move(outputs),
        move(backings),
        move(exported),
    };
}

static PerBatchState makePerBatchState(const DeviceState &dev,
                                       const FramebufferConfig &fb_cfg,
                                       const FramebufferState &fb,
                                       const ParamBufferConfig &param_cfg,
                                       VkCommandPool cmd_pool,
                                       HostBuffer &param_buffer,
                                       VkDescriptorSet rt_set,
                                       const BSDFPrecomputed &precomp_tex,
                                       bool auxiliary_outputs,
                                       uint32_t global_batch_idx)
{
    VkCommandBuffer render_cmd = makeCmdBuffer(dev, cmd_pool);

    VkDeviceSize output_buffer_offset =
        global_batch_idx * fb_cfg.linearOutputBytesPerBatch;

    VkDeviceSize normal_buffer_offset =
        global_batch_idx * fb_cfg.linearNormalBytesPerBatch;

    VkDeviceSize albedo_buffer_offset =
        global_batch_idx * fb_cfg.linearAlbedoBytesPerBatch;

    half *output_buffer = (half *)((char *)fb.exported[0].getDevicePointer() +
                                   output_buffer_offset);

    half *normal_buffer = nullptr, *albedo_buffer = nullptr;
    if (auxiliary_outputs) {
        normal_buffer = (half *)((char *)fb.exported[1].getDevicePointer() +
                                       normal_buffer_offset);

        albedo_buffer = (half *)((char *)fb.exported[2].getDevicePointer() +
                                       albedo_buffer_offset);
    }

    vector<VkWriteDescriptorSet> desc_set_updates;

    VkDeviceSize base_offset = global_batch_idx * param_cfg.totalParamBytes;

    uint8_t *base_ptr =
        reinterpret_cast<uint8_t *>(param_buffer.ptr) + base_offset;

    InstanceTransform *transform_ptr =
        reinterpret_cast<InstanceTransform *>(base_ptr);

    uint32_t *material_ptr = reinterpret_cast<uint32_t *>(
        base_ptr + param_cfg.materialIndicesOffset);

    PackedLight *light_ptr =
        reinterpret_cast<PackedLight *>(base_ptr + param_cfg.lightsOffset);

    PackedEnv *env_ptr =
        reinterpret_cast<PackedEnv *>(base_ptr + param_cfg.envOffset);

    DescriptorUpdates desc_updates(7);

    VkDescriptorBufferInfo transform_info {
        param_buffer.buffer,
        base_offset,
        param_cfg.totalTransformBytes,
    };

    desc_updates.buffer(rt_set, &transform_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo mat_info {
        param_buffer.buffer,
        base_offset + param_cfg.materialIndicesOffset,
        param_cfg.totalMaterialIndexBytes,
    };
    desc_updates.buffer(rt_set, &mat_info, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo light_info {
        param_buffer.buffer,
        base_offset + param_cfg.lightsOffset,
        param_cfg.totalLightParamBytes,
    };

    desc_updates.buffer(rt_set, &light_info, 2,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo env_info {
        param_buffer.buffer,
        base_offset + param_cfg.envOffset,
        param_cfg.totalEnvParamBytes,
    };

    desc_updates.buffer(rt_set, &env_info, 3,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorImageInfo diffuse_avg_info {
        VK_NULL_HANDLE,
        precomp_tex.msDiffuseAverage.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    desc_updates.textures(rt_set, &diffuse_avg_info, 1, 5);

    VkDescriptorImageInfo diffuse_dir_info {
        VK_NULL_HANDLE,
        precomp_tex.msDiffuseDirectional.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    desc_updates.textures(rt_set, &diffuse_dir_info, 1, 6);

    VkDescriptorImageInfo ggx_avg_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXAverage.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    desc_updates.textures(rt_set, &ggx_avg_info, 1, 7);

    VkDescriptorImageInfo ggx_dir_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXDirectional.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    desc_updates.textures(rt_set, &ggx_dir_info, 1, 8);

    VkDescriptorImageInfo ggx_inv_info {
        VK_NULL_HANDLE,
        precomp_tex.msGGXInverse.second,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    desc_updates.textures(rt_set, &ggx_inv_info, 1, 9);

    VkDescriptorBufferInfo out_info {
        fb.outputs[0].buffer,
        output_buffer_offset,
        fb_cfg.linearOutputBytesPerBatch,
    };

    desc_updates.buffer(rt_set, &out_info, 10,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo normal_info;
    VkDescriptorBufferInfo albedo_info;

    if (auxiliary_outputs) {
        normal_info = {
            fb.outputs[1].buffer,
            normal_buffer_offset,
            fb_cfg.linearNormalBytesPerBatch,
        };

        albedo_info = {
            fb.outputs[2].buffer,
            albedo_buffer_offset,
            fb_cfg.linearAlbedoBytesPerBatch,
        };

        desc_updates.buffer(rt_set, &normal_info, 11,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        desc_updates.buffer(rt_set, &albedo_info, 12,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    desc_updates.update(dev);

    return PerBatchState {
        makeFence(dev),
        render_cmd,
        output_buffer,
        normal_buffer,
        albedo_buffer,
        rt_set,
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
        { "ggx_avg_albedo.bin", { 16 } },
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

VulkanBackend::VulkanBackend(const RenderConfig &cfg, bool validate)
    : VulkanBackend(cfg, getBackendConfig(cfg, validate))
{}

VulkanBackend::VulkanBackend(const RenderConfig &cfg,
                             const BackendConfig &backend_cfg)
    : batch_size_(cfg.batchSize),
      inst(backend_cfg.validate, false, {}),
      dev(inst.makeDevice(getUUIDFromCudaID(cfg.gpuID),
                          1,
                          2,
                          cfg.numLoaders,
                          nullptr)),
      alloc(dev, inst),
      fb_cfg_(getFramebufferConfig(cfg, backend_cfg)),
      param_cfg_(getParamBufferConfig(cfg.batchSize, alloc)),
      render_state_(makeRenderState(dev, cfg, backend_cfg)),
      pipeline_(makePipeline(dev, render_state_)),
      fb_(makeFramebuffer(dev,
                          cfg,
                          fb_cfg_,
                          alloc)),
      transfer_queues_(initTransferQueues(cfg, dev)),
      graphics_queues_(initGraphicsQueues(dev)),
      compute_queues_(initComputeQueues(dev)),
      render_input_buffer_(alloc.makeParamBuffer(param_cfg_.totalParamBytes *
                                                 backend_cfg.numBatches)),
      cmd_pool_(makeCmdPool(dev, dev.computeQF)),
      bsdf_precomp_(loadPrecomputedTextures(dev, alloc, compute_queues_[0],
                    dev.computeQF)),
      num_loaders_(0),
      max_loaders_(cfg.numLoaders),
      max_texture_resolution_(cfg.maxTextureResolution == 0 ? ~0u :
                              cfg.maxTextureResolution),
      batch_states_(),
      cur_batch_(0),
      batch_mask_(backend_cfg.numBatches == 2 ? 1 : 0),
      frame_counter_(0)
{
    batch_states_.reserve(backend_cfg.numBatches);
    for (int i = 0; i < (int)backend_cfg.numBatches; i++) {
        batch_states_.emplace_back(makePerBatchState(
            dev, fb_cfg_, fb_, param_cfg_, cmd_pool_,
            render_input_buffer_, render_state_.rtPool.makeSet(),
            bsdf_precomp_, cfg.auxiliaryOutputs, i));
    }
}

LoaderImpl VulkanBackend::makeLoader()
{
    int loader_idx = num_loaders_.fetch_add(1, memory_order_acq_rel);
    assert(loader_idx < max_loaders_);

    auto loader = new VulkanLoader(
        dev, alloc, transfer_queues_[loader_idx % transfer_queues_.size()],
        compute_queues_.back(), render_state_.rt, dev.computeQF,
        max_texture_resolution_);

    return makeLoaderImpl<VulkanLoader>(loader);
}

EnvironmentImpl VulkanBackend::makeEnvironment(const shared_ptr<Scene> &scene)
{
    const VulkanScene &vk_scene = *static_cast<VulkanScene *>(scene.get());
    VulkanEnvironment *environment = new VulkanEnvironment(vk_scene);
    return makeEnvironmentImpl<VulkanEnvironment>(environment);
}

static PackedCamera packCamera(const Camera &cam)
{
    PackedCamera packed;

    glm::vec3 scaled_up = -cam.tanFOV * cam.up;
    glm::vec3 scaled_right = cam.aspectRatio * cam.tanFOV * cam.right;

    packed.data[0] = glm::vec4(cam.position.x, cam.position.y,
        cam.position.z, cam.view.x);
    packed.data[1] = glm::vec4(cam.view.y, cam.view.z,
        scaled_up.x, scaled_up.y);
    packed.data[2] = glm::vec4(scaled_up.z, scaled_right.x,
        scaled_right.y, scaled_right.z);

    return packed;
}

uint32_t VulkanBackend::render(const Environment *envs)
{
    PerBatchState &batch_state = batch_states_[cur_batch_];

    VkCommandBuffer render_cmd = batch_state.renderCmd;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd, &begin_info));

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_.rtState.hdl);

    RTPushConstant push_const {
        frame_counter_,
    };

    dev.dt.cmdPushConstants(render_cmd, pipeline_.rtState.layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(RTPushConstant),
                            &push_const);

    dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline_.rtState.layout, 0, 1,
                                 &batch_state.rtSet, 0, nullptr);

    // Hack
    {
        const VulkanScene &scene =
            *static_cast<const VulkanScene *>(envs[0].getScene().get());
        dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline_.rtState.layout, 1, 1,
                                     &scene.descSet.hdl, 0, nullptr);
    }

    // TLAS build
    for (int batch_idx = 0; batch_idx < (int)batch_size_; batch_idx++) {
        const Environment &env = envs[batch_idx];

        if (env.isDirty()) {
            VulkanEnvironment &env_backend =
                *(VulkanEnvironment *)(env.getBackend());
            const VulkanScene &scene =
                *static_cast<const VulkanScene *>(env.getScene().get());

            env_backend.tlas.build(dev, alloc, env.getInstances(),
                                   env.getTransforms(), env.getInstanceFlags(),
                                   scene.objectInfo, scene.blases, render_cmd);
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
    for (int batch_idx = 0; batch_idx < (int)batch_size_; batch_idx++) {
        const Environment &env = envs[batch_idx];
        const VulkanEnvironment &env_backend =
            *static_cast<const VulkanEnvironment *>(env.getBackend());

        PackedEnv &packed_env = batch_state.envPtr[batch_idx];

        packed_env.cam = packCamera(env.getCamera());

        const auto &env_transforms = env.getTransforms();
        uint32_t num_instances = env.getNumInstances();
        memcpy(&batch_state.transformPtr[inst_offset], env_transforms.data(),
               sizeof(InstanceTransform) * num_instances);

        packed_env.data.x = inst_offset;
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
    }

    dev.dt.cmdDispatch(
        render_cmd,
        getWorkgroupSize(fb_cfg_.imgWidth),
        getWorkgroupSize(fb_cfg_.imgHeight),
        getWorkgroupSize(batch_size_));

    REQ_VK(dev.dt.endCommandBuffer(render_cmd));

    render_input_buffer_.flush(dev);

    uint32_t rendered_batch_idx = cur_batch_;

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

    compute_queues_[0].submit(dev, 1, &render_submit, batch_state.fence);

    frame_counter_ += batch_size_;

    cur_batch_ = (cur_batch_ + 1) & batch_mask_;

    return rendered_batch_idx;
}

void VulkanBackend::waitForFrame(uint32_t batch_idx)
{
    VkFence fence = batch_states_[batch_idx].fence;
    assert(fence != VK_NULL_HANDLE);
    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);
}

half *VulkanBackend::getOutputPointer(uint32_t batch_idx)
{
    return batch_states_[batch_idx].outputBuffer;
}

AuxiliaryOutputs VulkanBackend::getAuxiliaryOutputs(uint32_t batch_idx)
{
    return {
        batch_states_[batch_idx].normalBuffer,
        batch_states_[batch_idx].albedoBuffer,
    };
}

}
}
