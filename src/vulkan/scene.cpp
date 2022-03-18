#include "scene.hpp"
#include <vulkan/vulkan_core.h>

#include "rlpbr_core/utils.hpp"
#include "shader.hpp"
#include "utils.hpp"
#include "vulkan/core.hpp"
#include "vulkan/memory.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

namespace RLpbr {
namespace vk {

namespace InternalConfig {
constexpr float reservoirCellSize = 1.f;
}

static ReservoirGrid makeReservoirGrid(
    const DeviceState &dev,
    MemoryAllocator &alloc,
    const VulkanScene &scene)
{
    AABB bbox = scene.envInit.defaultBBox;
    // Round bbox size out to reservoirCellSize
    glm::vec3 min_remainder = glm::vec3(
        fmodf(bbox.pMin.x, InternalConfig::reservoirCellSize),
        fmodf(bbox.pMin.y, InternalConfig::reservoirCellSize),
        fmodf(bbox.pMin.z, InternalConfig::reservoirCellSize));

    bbox.pMin -= min_remainder;

    glm::vec3 max_remainder = glm::vec3(
        fmodf(bbox.pMax.x, InternalConfig::reservoirCellSize),
        fmodf(bbox.pMax.y, InternalConfig::reservoirCellSize),
        fmodf(bbox.pMax.z, InternalConfig::reservoirCellSize));

    bbox.pMax += 1.f - max_remainder;

    glm::vec3 bbox_size = bbox.pMax - bbox.pMin;
    glm::vec3 num_cells_frac = bbox_size / InternalConfig::reservoirCellSize;
    glm::u32vec3 num_cells = glm::ceil(num_cells_frac);

    uint32_t total_cells = num_cells.x * num_cells.y * num_cells.z;

    total_cells = 1; // FIXME
    auto [grid_buffer, grid_memory] =
        alloc.makeDedicatedBuffer(total_cells * sizeof(Reservoir), true);

    VkDeviceAddress dev_addr;
    VkBufferDeviceAddressInfo addr_info;
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.pNext = nullptr;
    addr_info.buffer = grid_buffer.buffer;
    dev_addr = dev.dt.getBufferDeviceAddress(dev.hdl, &addr_info);

    return ReservoirGrid {
        bbox,
        grid_memory,
        dev_addr,
        move(grid_buffer),
    };
}

DomainRandomization randomizeDomain(mt19937 &rand_gen,
                                    uint32_t num_env_maps,
                                    bool enable_randomization)
{
    if (!enable_randomization) {
        return DomainRandomization {
            glm::quat(1, 0, 0, 0),
            glm::vec3(1.f, 1.f, 1.f),
            0,
        };
    }

    uniform_real_distribution<float> rand_dist(0, 1.f);

    float env_angle = rand_dist(rand_gen) * 2.f * M_PI;

    glm::quat env_rot = glm::angleAxis(env_angle, glm::vec3(0.f, 1.f, 0.f));

    glm::vec3 light_filter(rand_dist(rand_gen),
                           rand_dist(rand_gen),
                           rand_dist(rand_gen));

    light_filter = glm::normalize(light_filter);

    // FIXME: Not very useful currently
    light_filter = glm::vec3(1.f);

    uniform_int_distribution<uint32_t> env_map_dist(0, num_env_maps - 1);

    return DomainRandomization {
        env_rot,
        light_filter,
        env_map_dist(rand_gen),
    };
}

VulkanEnvironment::VulkanEnvironment(const DeviceState &d,
                                     const VulkanScene &scene,
                                     const Camera &cam,
                                     mt19937 &rand_gen,
                                     bool should_randomize)
    : EnvironmentBackend {},
      lights(),
      dev(d),
      tlas(),
      prevCam(cam),
      domainRandomization(randomizeDomain(rand_gen, 1, should_randomize))
{
    for (const LightProperties &light : scene.envInit.lights) {
        PackedLight packed;
        memcpy(&packed.data.x, &light.type, sizeof(uint32_t));
        if (light.type == LightType::Sphere) {
            packed.data.y = glm::uintBitsToFloat(light.sphereVertIdx);
            packed.data.z = glm::uintBitsToFloat(light.sphereMatIdx);
            packed.data.w = light.radius;
        } else if (light.type == LightType::Triangle) {
            packed.data.y = glm::uintBitsToFloat(light.triIdxOffset);
            packed.data.z = glm::uintBitsToFloat(light.triMatIdx);
        } else if (light.type == LightType::Portal) {
            packed.data.y = glm::uintBitsToFloat(light.portalIdxOffset);
        }

        lights.push_back(packed);
    }
}

VulkanEnvironment::~VulkanEnvironment()
{
    tlas.free(dev);
}

uint32_t VulkanEnvironment::addLight(const glm::vec3 &position,
                                     const glm::vec3 &color)
{
    // FIXME
    (void)position;
    (void)color;
    lights.push_back(PackedLight {
    });

    return lights.size() - 1;
}

void VulkanEnvironment::removeLight(uint32_t idx)
{
    lights[idx] = lights.back();
    lights.pop_back();
}

VulkanLoader::VulkanLoader(const DeviceState &d,
                           MemoryAllocator &alc,
                           const QueueState &transfer_queue,
                           const QueueState &render_queue,
                           VkDescriptorSet scene_set,
                           DescriptorManager &&env_map_pool,
                           uint32_t render_qf,
                           uint32_t max_texture_resolution)
    : VulkanLoader(d, alc, transfer_queue, render_queue, nullptr,
                   scene_set, move(env_map_pool), render_qf,
                   max_texture_resolution)
{}

VulkanLoader::VulkanLoader(const DeviceState &d,
                           MemoryAllocator &alc,
                           const QueueState &transfer_queue,
                           const QueueState &render_queue,
                           SharedSceneState &shared_scene_state,
                           DescriptorManager &&env_map_pool,
                           uint32_t render_qf,
                           uint32_t max_texture_resolution)
    : VulkanLoader(d, alc, transfer_queue, render_queue, &shared_scene_state,
                   shared_scene_state.descSet,
                   move(env_map_pool), render_qf,
                   max_texture_resolution)
{}

static VkQueryPool makeQueryPool(const DeviceState &dev,
                                 uint32_t max_num_queries,
                                 VkQueryType query_type)
{
    VkQueryPoolCreateInfo pool_info;
    pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_info.pNext = nullptr;
    pool_info.flags = 0;
    pool_info.queryType = query_type;
    pool_info.queryCount = max_num_queries;
    pool_info.pipelineStatistics = 0;

    VkQueryPool query_pool;
    REQ_VK(dev.dt.createQueryPool(dev.hdl, &pool_info, nullptr, &query_pool));

    return query_pool;
}

VulkanLoader::VulkanLoader(const DeviceState &d,
                           MemoryAllocator &alc,
                           const QueueState &transfer_queue,
                           const QueueState &render_queue,
                           SharedSceneState *shared_scene_state,
                           VkDescriptorSet scene_set,
                           DescriptorManager &&env_map_pool,
                           uint32_t render_qf,
                           uint32_t max_texture_resolution)
    : dev(d),
      alloc(alc),
      transfer_queue_(transfer_queue),
      render_queue_(render_queue),
      shared_scene_state_(shared_scene_state),
      scene_set_(scene_set),
      env_map_pool_(move(env_map_pool)),
      transfer_cmd_pool_(makeCmdPool(d, d.transferQF)),
      transfer_cmd_(makeCmdBuffer(dev, transfer_cmd_pool_)),
      render_cmd_pool_(makeCmdPool(d, render_qf)),
      render_cmd_(makeCmdBuffer(dev, render_cmd_pool_)),
      transfer_sema_(makeBinarySemaphore(dev)),
      fence_(makeFence(dev)),
      max_queries_(256),
      compacted_query_pool_(makeQueryPool(dev, max_queries_,
          VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)),
      serialized_query_pool_(makeQueryPool(dev, max_queries_,
          VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR)),
      render_qf_(render_qf),
      max_texture_resolution_(max_texture_resolution)
{}

TextureData::TextureData(const DeviceState &d, MemoryAllocator &a)
    : dev(d),
      alloc(a),
      memory(VK_NULL_HANDLE),
      textures(),
      views()
{}

TextureData::TextureData(TextureData &&o)
    : dev(o.dev),
      alloc(o.alloc),
      memory(o.memory),
      textures(move(o.textures)),
      views(move(o.views))
{
    o.memory = VK_NULL_HANDLE;
}

TextureData::~TextureData()
{
    if (memory == VK_NULL_HANDLE) return;

    for (auto view : views) {
        dev.dt.destroyImageView(dev.hdl, view, nullptr);
    }

    for (auto &texture : textures) {
        alloc.destroyTexture(move(texture));
    }

    dev.dt.freeMemory(dev.hdl, memory, nullptr);
}

static tuple<uint8_t *, uint32_t, glm::u32vec2, uint32_t, float>
loadTextureFromDisk(const string &tex_path, uint32_t texel_bytes,
                    uint32_t max_texture_resolution)
{
    ifstream tex_file(tex_path, ios::in | ios::binary);
    auto read_uint = [&tex_file]() {
        uint32_t v;
        tex_file.read((char *)&v, sizeof(uint32_t));
        return v;
    };

    auto magic = read_uint();
    if (magic != 0x50505050) {
        cerr << "Invalid texture file" << endl;
        abort();
    }
    auto total_num_levels = read_uint();
    
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t num_compressed_bytes = 0;
    uint32_t num_decompressed_bytes = 0;
    uint32_t skip_bytes = 0;
    vector<pair<uint32_t, uint32_t>> png_pos;
    png_pos.reserve(total_num_levels);

    uint32_t level_alignment = max(texel_bytes, 4u);

    uint32_t num_skip_levels = 0;
    for (int i = 0; i < (int)total_num_levels; i++) {
        uint32_t level_x = read_uint();
        uint32_t level_y = read_uint();
        uint32_t offset = read_uint();
        uint32_t lvl_compressed_bytes = read_uint();

        if (level_x > max_texture_resolution &&
            level_y > max_texture_resolution) {
            skip_bytes += lvl_compressed_bytes;
            num_skip_levels++;
            continue;
        }

        num_decompressed_bytes = alignOffset(num_decompressed_bytes,
                                             level_alignment);

        if (x == 0 && y == 0) {
            x = level_x;
            y = level_y;
        }

        png_pos.emplace_back(offset - skip_bytes,
                             lvl_compressed_bytes);
        num_decompressed_bytes +=
            level_x * level_y * texel_bytes;
        num_compressed_bytes += lvl_compressed_bytes;
    }

    int num_levels = total_num_levels - num_skip_levels;

    uint8_t *img_data = (uint8_t *)malloc(num_decompressed_bytes);
    tex_file.ignore(skip_bytes);

    uint8_t *compressed_data = (uint8_t *)malloc(num_compressed_bytes);
    tex_file.read((char *)compressed_data, num_compressed_bytes);

    uint32_t cur_offset = 0;
    for (int i = 0; i < (int)num_levels; i++) {
        auto [offset, num_bytes] = png_pos[i];
        int lvl_x, lvl_y, tmp_n;
        uint8_t *decompressed = stbi_load_from_memory(
            compressed_data + offset, num_bytes,
            &lvl_x, &lvl_y, &tmp_n, 4);

        cur_offset = alignOffset(cur_offset, level_alignment);

        for (int pix_idx = 0; pix_idx < int(lvl_x * lvl_y);
             pix_idx++) {
            uint8_t *decompressed_offset =
                decompressed + pix_idx * 4;
            uint8_t *out_offset =
                img_data + cur_offset + pix_idx * texel_bytes;

            for (int byte_idx = 0; byte_idx < (int)texel_bytes;
                 byte_idx++) {
                out_offset[byte_idx] =
                    decompressed_offset[byte_idx];
            }
        }
        free(decompressed);

        cur_offset += lvl_x * lvl_y * texel_bytes;
    }

    assert(cur_offset == num_decompressed_bytes);

    free(compressed_data);

    return make_tuple(img_data, num_decompressed_bytes, glm::u32vec2(x, y),
                      num_levels, -float(num_skip_levels));
}

static tuple<void *, uint64_t, glm::u32vec3,
             void *, uint64_t, glm::u32vec3>
    loadEnvironmentMapFromDisk(const string &env_path)
{
    ifstream tex_file(env_path, ios::in | ios::binary);

    auto read_uint = [&tex_file]() {
        uint32_t v;
        tex_file.read((char *)&v, sizeof(uint32_t));
        return v;
    };

    uint32_t num_env_mips = read_uint();
    uint32_t env_width = read_uint();
    uint32_t env_height = read_uint();
    uint64_t env_bytes;
    tex_file.read((char *)&env_bytes, sizeof(uint64_t));

    void *env_staging = malloc(env_bytes);
    tex_file.read((char *)env_staging, env_bytes);

    uint32_t num_imp_mips = read_uint();
    uint32_t imp_width = read_uint();
    uint32_t imp_height = read_uint();

    assert(imp_width == imp_height);
    uint64_t imp_bytes;
    tex_file.read((char *)&imp_bytes, sizeof(uint64_t));

    void *imp_staging = malloc(imp_bytes);
    tex_file.read((char *)imp_staging, imp_bytes);

    return {
        env_staging,
        env_bytes,
        { env_width, env_height, num_env_mips },
        imp_staging,
        imp_bytes,
        { imp_width, imp_height, num_imp_mips },
    };
}

struct StagedTextures {
    HostBuffer stageBuffer;
    VkDeviceMemory texMemory;
    vector<size_t> stageOffsets;
    vector<LocalTexture> textures;
    vector<VkImageView> textureViews;
    vector<uint32_t> textureTexelBytes;

    vector<uint32_t> base;
    vector<uint32_t> metallicRoughness;
    vector<uint32_t> specular;
    vector<uint32_t> normal;
    vector<uint32_t> emittance;
    vector<uint32_t> transmission;
    vector<uint32_t> clearcoat;
    vector<uint32_t> anisotropic;
};

static optional<StagedTextures> prepareSceneTextures(const DeviceState &dev,
                                           const TextureInfo &texture_info,
                                           uint32_t max_texture_resolution,
                                           MemoryAllocator &alloc)
{
    uint32_t num_textures =
        texture_info.base.size() + texture_info.metallicRoughness.size() +
        texture_info.specular.size() + texture_info.normal.size() +
        texture_info.emittance.size() + texture_info.transmission.size() +
        texture_info.clearcoat.size() + texture_info.anisotropic.size();

    if (num_textures == 0) {
        return optional<StagedTextures>();
    }

    vector<void *> host_ptrs;
    vector<uint32_t> host_sizes;
    vector<size_t> stage_offsets;
    vector<LocalTexture> gpu_textures;
    vector<VkFormat> texture_formats;
    vector<uint32_t> texture_texel_bytes;
    vector<size_t> texture_offsets;
    vector<VkImageView> texture_views;

    host_ptrs.reserve(num_textures);
    host_sizes.reserve(num_textures);
    stage_offsets.reserve(num_textures);
    gpu_textures.reserve(num_textures);
    texture_formats.reserve(num_textures);
    texture_texel_bytes.reserve(num_textures);
    texture_offsets.reserve(num_textures);
    texture_views.reserve(num_textures);

    size_t cur_tex_offset = 0;
    auto stageTexture = [&](void *img_data, uint32_t img_bytes,
                            uint32_t width, uint32_t height,
                            uint32_t num_levels, VkFormat fmt,
                            uint32_t texel_bytes) {
         host_ptrs.push_back(img_data);
         host_sizes.push_back(img_bytes);

         auto [gpu_tex, tex_reqs] =
             alloc.makeTexture2D(width, height, num_levels, fmt);

         gpu_textures.emplace_back(move(gpu_tex));
         texture_formats.push_back(fmt);
         texture_texel_bytes.push_back(texel_bytes);

         cur_tex_offset = alignOffset(cur_tex_offset, tex_reqs.alignment);
         texture_offsets.push_back(cur_tex_offset);
         cur_tex_offset += tex_reqs.size;

         return gpu_textures.size() - 1;
    };

    auto stageTextureList = [&](const vector<string> &texture_names,
                                TextureFormat orig_fmt) {
        vector<uint32_t> tex_locs;
        tex_locs.reserve(texture_names.size());

        uint32_t texel_bytes = getTexelBytes(orig_fmt);
        VkFormat fmt = alloc.getTextureFormat(orig_fmt);

        for (const string &tex_name : texture_names) {
            auto [img_data, num_stage_bytes, dims, num_levels, bias] =
                loadTextureFromDisk(texture_info.textureDir + tex_name,
                                    texel_bytes, max_texture_resolution);

            tex_locs.push_back(
                stageTexture(img_data, num_stage_bytes, dims.x, dims.y,
                             num_levels, fmt, texel_bytes));
        }

        return tex_locs;
    };

    TextureFormat fourCompSRGB = TextureFormat::R8G8B8A8_SRGB;

    TextureFormat twoCompUnorm = TextureFormat::R8G8_UNORM;

    auto base_locs = stageTextureList(texture_info.base, fourCompSRGB);
                                   
    auto mr_locs = stageTextureList(texture_info.metallicRoughness,
                                    twoCompUnorm);
    auto specular_locs = stageTextureList(texture_info.specular,
                                          fourCompSRGB);
    auto normal_locs = stageTextureList(texture_info.normal,
                                        twoCompUnorm);
    auto emittance_locs = stageTextureList(texture_info.emittance,
                                           fourCompSRGB);
    auto transmission_locs = stageTextureList(texture_info.transmission,
                                              TextureFormat::R8_UNORM);
    auto clearcoat_locs = stageTextureList(texture_info.clearcoat,
                                           twoCompUnorm);
    auto anisotropic_locs = stageTextureList(texture_info.anisotropic,
                                             twoCompUnorm);

    size_t num_device_bytes = cur_tex_offset;

    size_t num_staging_bytes = 0;
    for (int i = 0; i < (int)host_sizes.size(); i++) {
        uint32_t num_bytes = host_sizes[i];
        uint32_t texel_bytes = texture_texel_bytes[i];

        uint32_t alignment = max(texel_bytes, 4u);
        num_staging_bytes = alignOffset(num_staging_bytes, alignment);

        stage_offsets.push_back(num_staging_bytes);

        num_staging_bytes += num_bytes;
    }

    HostBuffer texture_staging = alloc.makeStagingBuffer(num_staging_bytes);

    for (int i = 0 ; i < (int)num_textures; i++) {
        char *cur_ptr = (char *)texture_staging.ptr + stage_offsets[i];
        memcpy(cur_ptr, host_ptrs[i], host_sizes[i]);
        free(host_ptrs[i]);
    }

    texture_staging.flush(dev);

    optional<VkDeviceMemory> tex_mem_opt = alloc.alloc(num_device_bytes);
    if (!tex_mem_opt.has_value()) {
        cerr << "Out of memory, failed to allocate texture memory" << endl;
        fatalExit();
    }

    VkDeviceMemory tex_mem = tex_mem_opt.value();

    // Bind image memory and create views
    for (uint32_t i = 0; i < num_textures; i++) {
        LocalTexture &gpu_texture = gpu_textures[i];
        VkDeviceSize offset = texture_offsets[i];

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, gpu_texture.image,
                                      tex_mem, offset));

        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = gpu_texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = texture_formats[i];
        view_info.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A,
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            gpu_texture.mipLevels, 
            0,
            1,
        };

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        texture_views.push_back(view);
    }

    return StagedTextures {
        move(texture_staging),
        tex_mem,
        move(stage_offsets),
        move(gpu_textures),
        move(texture_views),
        move(texture_texel_bytes),
        move(base_locs),
        move(mr_locs),
        move(specular_locs),
        move(normal_locs),
        move(emittance_locs),
        move(transmission_locs),
        move(clearcoat_locs),
        move(anisotropic_locs),
    };
}

BLASData::BLASData(const DeviceState &d, vector<BLAS> &&as,
                   LocalBuffer &&buf)
    : dev(&d),
      accelStructs(move(as)),
      storage(move(buf))
{}

BLASData::BLASData(BLASData &&o)
    : dev(o.dev),
      accelStructs(move(o.accelStructs)),
      storage(move(o.storage))
{}

static void freeBLASes(const DeviceState &dev, const vector<BLAS> &blases)
{
    for (const auto &blas : blases) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, blas.hdl,
                                                nullptr);
    }
}

BLASData &BLASData::operator=(BLASData &&o)
{
    freeBLASes(*dev, accelStructs);

    dev = o.dev;
    accelStructs = move(o.accelStructs);
    storage = move(o.storage);

    return *this;
}

BLASData::~BLASData()
{
    freeBLASes(*dev, accelStructs);
}

static DynArray<uint64_t> getBLASProperties(const DeviceState &dev,
                                            const BLASData &blas_data,
                                            VkCommandPool cmd_pool,
                                            VkCommandBuffer query_cmd,
                                            const QueueState &query_queue,
                                            VkFence fence,
                                            VkQueryPool query_pool,
                                            uint32_t max_num_queries,
                                            VkQueryType query_type)
{
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    DynArray<VkAccelerationStructureKHR> hdls(blas_data.accelStructs.size());

    for (int i = 0; i < (int)hdls.size(); i++) {
        hdls[i] = blas_data.accelStructs[i].hdl;
    }
    DynArray<uint64_t> blas_props(hdls.size());

    for (int query_offset = 0; query_offset < (int)hdls.size();
         query_offset += max_num_queries) {
        uint32_t query_size =
            min<uint32_t>(max_num_queries, hdls.size() - query_offset);

        REQ_VK(dev.dt.resetCommandPool(dev.hdl, cmd_pool, 0));
        REQ_VK(dev.dt.beginCommandBuffer(query_cmd, &begin_info));

        dev.dt.cmdResetQueryPool(query_cmd, query_pool,
                                 0, query_size);

        dev.dt.cmdWriteAccelerationStructuresPropertiesKHR(query_cmd,
            query_size, hdls.data() + query_offset,
            query_type,
            query_pool, 0);

        REQ_VK(dev.dt.endCommandBuffer(query_cmd));

        VkSubmitInfo query_submit {};
        query_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        query_submit.waitSemaphoreCount = 0;
        query_submit.pWaitSemaphores = nullptr;
        query_submit.pWaitDstStageMask = nullptr;
        query_submit.commandBufferCount = 1;
        query_submit.pCommandBuffers = &query_cmd;

        query_queue.submit(dev, 1, &query_submit, fence);

        waitForFenceInfinitely(dev, fence);
        resetFence(dev, fence);

        REQ_VK(dev.dt.getQueryPoolResults(dev.hdl,
            query_pool,
            0,
            query_size,
            sizeof(uint64_t) * query_size,
            blas_props.data() + query_offset,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    }

    return blas_props;
}

struct BLASBuildResults {
    BLASData blases;
    optional<LocalBuffer> scratch;
    optional<HostBuffer> staging;
    VkDeviceSize totalBLASBytes;
    bool blasesRebuilt;
};

static optional<BLASBuildResults> makeBLASes(
    const DeviceState &dev,
    MemoryAllocator &alloc, 
    const vector<MeshInfo> &meshes,
    const vector<ObjectInfo> &objects,
    uint32_t max_num_vertices,
    VkDeviceAddress vert_base,
    VkDeviceAddress index_base,
    VkCommandBuffer build_cmd)
{
    vector<VkAccelerationStructureGeometryKHR> geo_infos;
    vector<uint32_t> num_triangles;
    vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos;

    geo_infos.reserve(meshes.size());
    num_triangles.reserve(meshes.size());
    range_infos.reserve(meshes.size());

    vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos;
    vector<tuple<VkDeviceSize, VkDeviceSize, VkDeviceSize>> memory_locs;

    build_infos.reserve(objects.size());
    memory_locs.reserve(objects.size());

    VkDeviceSize total_scratch_bytes = 0;
    VkDeviceSize total_accel_bytes = 0;

    for (const ObjectInfo &object : objects) {
        for (int mesh_idx = 0; mesh_idx < (int)object.numMeshes; mesh_idx++) {
            const MeshInfo &mesh = meshes[object.meshIndex + mesh_idx];

            VkDeviceAddress vert_addr = vert_base;
            VkDeviceAddress index_addr =
                index_base + mesh.indexOffset * sizeof(uint32_t);

            VkAccelerationStructureGeometryKHR geo_info;
            geo_info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geo_info.pNext = nullptr;
            geo_info.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geo_info.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto &tri_info = geo_info.geometry.triangles;
            tri_info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            tri_info.pNext = nullptr;
            tri_info.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            tri_info.vertexData.deviceAddress = vert_addr;
            tri_info.vertexStride = sizeof(Vertex);
            tri_info.maxVertex = max_num_vertices;
            tri_info.indexType = VK_INDEX_TYPE_UINT32;
            tri_info.indexData.deviceAddress = index_addr;
            tri_info.transformData.deviceAddress = 0;

            geo_infos.push_back(geo_info);
            num_triangles.push_back(mesh.numTriangles);

            VkAccelerationStructureBuildRangeInfoKHR range_info;
            range_info.primitiveCount = mesh.numTriangles;
            range_info.primitiveOffset = 0;
            range_info.firstVertex = 0;
            range_info.transformOffset = 0;
            range_infos.push_back(range_info);
        }

        VkAccelerationStructureBuildGeometryInfoKHR build_info;
        build_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.pNext = nullptr;
        build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        build_info.mode =
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.srcAccelerationStructure = VK_NULL_HANDLE;
        build_info.dstAccelerationStructure = VK_NULL_HANDLE;
        build_info.geometryCount = object.numMeshes;
        build_info.pGeometries = &geo_infos[object.meshIndex];
        build_info.ppGeometries = nullptr;
        // Set device address to 0 before space calculation 
        build_info.scratchData.deviceAddress = 0;
        build_infos.push_back(build_info);

        VkAccelerationStructureBuildSizesInfoKHR size_info;
        size_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        size_info.pNext = nullptr;

        dev.dt.getAccelerationStructureBuildSizesKHR(
            dev.hdl, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_infos.back(),
            &num_triangles[object.meshIndex],
            &size_info);

         // Must be aligned to 256 as per spec
        total_accel_bytes = alignOffset(total_accel_bytes, 256);

        memory_locs.emplace_back(total_scratch_bytes, total_accel_bytes,
                                 size_info.accelerationStructureSize);

        total_scratch_bytes += size_info.buildScratchSize;
        total_accel_bytes += size_info.accelerationStructureSize;
    }

    optional<LocalBuffer> scratch_mem_opt =
        alloc.makeLocalBuffer(total_scratch_bytes, true);

    optional<LocalBuffer> accel_mem_opt =
        alloc.makeLocalBuffer(total_accel_bytes, true);

    if (!scratch_mem_opt.has_value() || !accel_mem_opt.has_value()) {
        return {};
    }

    LocalBuffer &scratch_mem = scratch_mem_opt.value();
    LocalBuffer &accel_mem = accel_mem_opt.value();

    VkBufferDeviceAddressInfoKHR scratch_addr_info;
    scratch_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    scratch_addr_info.pNext = nullptr;
    scratch_addr_info.buffer = scratch_mem.buffer;
    VkDeviceAddress scratch_base_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &scratch_addr_info);

    vector<BLAS> accel_structs;
    vector<VkAccelerationStructureBuildRangeInfoKHR *> range_info_ptrs;
    accel_structs.reserve(objects.size());
    range_info_ptrs.reserve(objects.size());

    for (int obj_idx = 0; obj_idx < (int)objects.size(); obj_idx++) {
        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = accel_mem.buffer;
        create_info.offset = get<1>(memory_locs[obj_idx]);
        create_info.size = get<2>(memory_locs[obj_idx]);
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.deviceAddress = 0;

        VkAccelerationStructureKHR blas;
        REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                     nullptr, &blas));

        auto &build_info = build_infos[obj_idx];
        build_info.dstAccelerationStructure = blas;
        build_info.scratchData.deviceAddress =
            scratch_base_addr + get<0>(memory_locs[obj_idx]);

        VkAccelerationStructureDeviceAddressInfoKHR addr_info;
        addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.pNext = nullptr;
        addr_info.accelerationStructure = blas;

        VkDeviceAddress dev_addr = 
            dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl, &addr_info);

        accel_structs.push_back({
            blas,
            dev_addr,
        });

        range_info_ptrs.push_back(&range_infos[objects[obj_idx].meshIndex]);
    }

    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd,
        build_infos.size(), build_infos.data(), range_info_ptrs.data());

    return BLASBuildResults {
        BLASData(dev, move(accel_structs), move(accel_mem)),
        move(scratch_mem),
        {},
        total_accel_bytes,
        true,
    };
}

static optional<BLASBuildResults> loadCachedBLASes(
        const DeviceState &dev, MemoryAllocator &alloc,
        string_view blas_path, VkCommandBuffer build_cmd)
{
    ifstream blases_file(filesystem::path(blas_path), ios::binary);

    uint32_t num_blases;
    blases_file.read((char *)&num_blases, sizeof(uint32_t));
    uint64_t total_serialized_bytes;
    blases_file.read((char *)&total_serialized_bytes, sizeof(uint64_t));
    HostBuffer staging_buf =
        alloc.makeHostBuffer(total_serialized_bytes, true);
    blases_file.read((char *)staging_buf.ptr, total_serialized_bytes);
    staging_buf.flush(dev);

    uint8_t *staging_ptr = (uint8_t *)staging_buf.ptr;

    bool version_match = true;
    uint64_t total_deserialized_bytes = 0;
    {
        int64_t cur_offset = 0;
        for (int i = 0; i < (int)num_blases; i++) {
            VkAccelerationStructureVersionInfoKHR blas_version;
            blas_version.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
            blas_version.pNext = nullptr;
            blas_version.pVersionData = staging_ptr + cur_offset;

            VkAccelerationStructureCompatibilityKHR compat_result;
            dev.dt.getDeviceAccelerationStructureCompatibilityKHR(
                dev.hdl, 
                &blas_version,
                &compat_result);

            if (compat_result ==
                VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR) {
                version_match = false;
                break;
            }

            uint64_t serialized_bytes;
            memcpy(&serialized_bytes, staging_ptr + cur_offset +
                   2 * VK_UUID_SIZE, sizeof(uint64_t));

            uint64_t deserialized_bytes;
            memcpy(&deserialized_bytes, staging_ptr + cur_offset +
                   2 * VK_UUID_SIZE + sizeof(uint64_t), sizeof(uint64_t));

            cur_offset += serialized_bytes;
            cur_offset = alignOffset(cur_offset, 256);

            total_deserialized_bytes += deserialized_bytes;
            total_deserialized_bytes =
                alignOffset(total_deserialized_bytes, 256);
        }
    }

    if (!version_match) {
        cerr <<
            "WARNING: cached BLAS found but version check failed - rebuilding\n"
            << endl;
        return {};
    }

    optional<LocalBuffer> accel_buf_opt =
        alloc.makeLocalBuffer(total_deserialized_bytes, true);
    LocalBuffer accel_buf = move(*accel_buf_opt);

    VkBufferDeviceAddressInfo staging_addr_info;
    staging_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    staging_addr_info.pNext = nullptr;
    staging_addr_info.buffer = staging_buf.buffer;
    VkDeviceAddress staging_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &staging_addr_info);

    uint64_t serialized_offset = 0;
    uint64_t deserialized_offset = 0;
    vector<BLAS> blases;
    for (int i = 0; i < (int)num_blases; i++) {
        uint64_t serialized_size;
        memcpy(&serialized_size, staging_ptr + serialized_offset +
               2 * VK_UUID_SIZE, sizeof(uint64_t));
        uint64_t deserialized_size;
        memcpy(&deserialized_size, staging_ptr + serialized_offset +
               2 * VK_UUID_SIZE + sizeof(uint64_t), sizeof(uint64_t));

        VkAccelerationStructureCreateInfoKHR as_info;
        as_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        as_info.pNext = nullptr;
        as_info.createFlags = 0;
        as_info.buffer = accel_buf.buffer;
        as_info.offset = deserialized_offset;
        as_info.size = deserialized_size;
        as_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        as_info.deviceAddress = 0;

        VkAccelerationStructureKHR blas_hdl;
        REQ_VK(dev.dt.createAccelerationStructureKHR(
                dev.hdl, &as_info, nullptr, &blas_hdl));

        VkAccelerationStructureDeviceAddressInfoKHR blas_addr_info;
        blas_addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        blas_addr_info.pNext = nullptr;
        blas_addr_info.accelerationStructure = blas_hdl;

        VkDeviceAddress blas_addr =
            dev.dt.getAccelerationStructureDeviceAddressKHR(
                dev.hdl, &blas_addr_info);

        blases.push_back({
            blas_hdl,
            blas_addr,
        });

        VkCopyMemoryToAccelerationStructureInfoKHR copy_info;
        copy_info.sType =
            VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
        copy_info.pNext = nullptr;
        copy_info.src.deviceAddress = staging_addr + serialized_offset;
        copy_info.dst = blas_hdl;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;

        dev.dt.cmdCopyMemoryToAccelerationStructureKHR(
            build_cmd, &copy_info);

        serialized_offset += serialized_size;
        serialized_offset = alignOffset(serialized_offset, 256);
        deserialized_offset += deserialized_size;
        deserialized_offset = alignOffset(deserialized_offset, 256);
    }

    return BLASBuildResults {
        BLASData(dev, move(blases), move(accel_buf)),
        {},
        move(staging_buf),
        total_deserialized_bytes,
        false,
    };
}

static void cacheBLASes(const DeviceState &dev, MemoryAllocator &alloc,
                        string_view blas_path, const BLASData &blas_data,
                        VkCommandPool cmd_pool, VkCommandBuffer build_cmd,
                        const QueueState &build_queue, VkFence fence,
                        VkQueryPool serialization_size_query_pool,
                        uint32_t max_num_queries)
{
    DynArray<uint64_t> blas_serialized_sizes = getBLASProperties(
        dev, blas_data, cmd_pool, build_cmd, build_queue, fence,
        serialization_size_query_pool, max_num_queries,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR);

    const int num_blases = blas_data.accelStructs.size();

    uint64_t total_serialized_bytes = 0;
    for (int i = 0; i < num_blases; i++) {
        total_serialized_bytes += blas_serialized_sizes[i];
        total_serialized_bytes = alignOffset(total_serialized_bytes, 256);
    }

    HostBuffer serialized_buffer =
        alloc.makeHostBuffer(total_serialized_bytes, true);

    VkBufferDeviceAddressInfo serialized_addr_info;
    serialized_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    serialized_addr_info.pNext = nullptr;
    serialized_addr_info.buffer = serialized_buffer.buffer;
    VkDeviceAddress serialized_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &serialized_addr_info);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, cmd_pool, 0));
    REQ_VK(dev.dt.beginCommandBuffer(build_cmd, &begin_info));

    int cur_offset = 0;
    for (int i = 0; i < num_blases; i++) {
        VkCopyAccelerationStructureToMemoryInfoKHR copy_info;
        copy_info.sType =
            VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
        copy_info.pNext = nullptr;
        copy_info.src = blas_data.accelStructs[i].hdl;
        copy_info.dst.deviceAddress = serialized_addr + cur_offset;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;

        dev.dt.cmdCopyAccelerationStructureToMemoryKHR(build_cmd, &copy_info);

        cur_offset += blas_serialized_sizes[i];
        cur_offset = alignOffset(cur_offset, 256);
    }

    REQ_VK(dev.dt.endCommandBuffer(build_cmd));

    VkSubmitInfo cache_submit {};
    cache_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    cache_submit.waitSemaphoreCount = 0;
    cache_submit.pWaitSemaphores = nullptr;
    cache_submit.pWaitDstStageMask = nullptr;
    cache_submit.commandBufferCount = 1;
    cache_submit.pCommandBuffers = &build_cmd;

    build_queue.submit(dev, 1, &cache_submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);

    ofstream cache_file(filesystem::path(blas_path), ios::binary);

    uint32_t num_blases_u32 = num_blases;
    cache_file.write((char *)&num_blases_u32, sizeof(uint32_t));
    cache_file.write((char *)&total_serialized_bytes, sizeof(uint64_t));
    cache_file.write((char *)serialized_buffer.ptr, total_serialized_bytes);
}

static BLASData compactBLASes(const DeviceState &dev,
                              MemoryAllocator &alloc,
                              const BLASData &blases,
                              VkCommandPool cmd_pool,
                              VkCommandBuffer compact_cmd,
                              const QueueState &compact_queue,
                              VkFence fence,
                              VkQueryPool compact_query_pool,
                              uint32_t max_queries)
{
    DynArray<uint64_t> blas_compacted_sizes = getBLASProperties(
        dev, blases, cmd_pool, compact_cmd, compact_queue, fence,
        compact_query_pool, max_queries,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR);

    const int num_blases = blases.accelStructs.size();

    uint64_t total_compacted_size = 0;
    for (int i = 0; i < num_blases; i++) {
        total_compacted_size += blas_compacted_sizes[i];
        total_compacted_size = alignOffset(total_compacted_size, 256);
    }

    auto compact_buffer_opt =
        alloc.makeLocalBuffer(total_compacted_size, true);

    if (!compact_buffer_opt.has_value()) {
        cerr << "OOM while compacting BLAS\n" << endl;
        fatalExit();
    }

    auto compact_buffer = move(*compact_buffer_opt);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, cmd_pool, 0));
    REQ_VK(dev.dt.beginCommandBuffer(compact_cmd, &begin_info));

    vector<BLAS> compacted_blases;
    compacted_blases.reserve(num_blases);

    uint64_t cur_offset = 0;
    for (int i = 0; i < num_blases; i++) {
        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = compact_buffer.buffer;
        create_info.offset = cur_offset;
        create_info.size = blas_compacted_sizes[i];
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.deviceAddress = 0;

        VkAccelerationStructureKHR blas;
        dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info, nullptr,
                                              &blas);

        VkAccelerationStructureDeviceAddressInfoKHR addr_info;
        addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.pNext = nullptr;
        addr_info.accelerationStructure = blas;

        VkDeviceAddress dev_addr = 
            dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl, &addr_info);

        compacted_blases.push_back({
            blas,
            dev_addr,
        });

        VkCopyAccelerationStructureInfoKHR copy_info;
        copy_info.sType =
            VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
        copy_info.pNext = nullptr;
        copy_info.src = blases.accelStructs[i].hdl;
        copy_info.dst = blas;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

        dev.dt.cmdCopyAccelerationStructureKHR(compact_cmd, &copy_info);

        cur_offset += blas_compacted_sizes[i];
        cur_offset = alignOffset(cur_offset, 256);
    }

    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = compact_buffer.buffer;
    barrier.offset = 0;
    barrier.size = total_compacted_size;

    dev.dt.cmdPipelineBarrier(
        compact_cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
        0, 0, nullptr,
        1, &barrier,
        0, nullptr);

    REQ_VK(dev.dt.endCommandBuffer(compact_cmd));

    VkSubmitInfo compact_submit {};
    compact_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    compact_submit.waitSemaphoreCount = 0;
    compact_submit.pWaitSemaphores = nullptr;
    compact_submit.pWaitDstStageMask = nullptr;
    compact_submit.commandBufferCount = 1;
    compact_submit.pCommandBuffers = &compact_cmd;

    compact_queue.submit(dev, 1, &compact_submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);

    return BLASData {
        dev,
        move(compacted_blases),
        move(compact_buffer),
    };
}

static optional<BLASBuildResults> getBLASes(
    const DeviceState &dev,
    MemoryAllocator &alloc, 
    const string_view blas_path,
    const vector<MeshInfo> &meshes,
    const vector<ObjectInfo> &objects,
    uint32_t max_num_vertices,
    VkDeviceAddress vert_base,
    VkDeviceAddress index_base,
    VkCommandBuffer build_cmd)
{
    optional<BLASBuildResults> blas_results;

    if (filesystem::exists(blas_path)) {
        blas_results =
            loadCachedBLASes(dev, alloc, blas_path, build_cmd);
    }

    if (!blas_results.has_value()) {
        blas_results = makeBLASes(dev, alloc, meshes, objects,
                                  max_num_vertices, vert_base,
                                  index_base, build_cmd);
    }

    if (blas_results.has_value()) {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = blas_results->blases.storage.buffer;
        barrier.offset = 0;
        barrier.size = blas_results->totalBLASBytes;

        dev.dt.cmdPipelineBarrier(
            build_cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
            0, 0, nullptr,
            1, &barrier,
            0, nullptr);
    }

    return blas_results;
}

void TLAS::build(const DeviceState &dev,
                 MemoryAllocator &alloc,
                 const vector<ObjectInstance> &instances,
                 const vector<InstanceTransform> &instance_transforms,
                 const vector<InstanceFlags> &instance_flags,
                 const vector<ObjectInfo> &objects,
                 const BLASData &blases,
                 VkCommandBuffer build_cmd)
{
    int new_num_instances = instances.size();
    if ((int)numBuildInstances < new_num_instances) {
        numBuildInstances = new_num_instances;

        buildStorage = alloc.makeHostBuffer(
            sizeof(VkAccelerationStructureInstanceKHR) * numBuildInstances,
            true);
    }

    VkAccelerationStructureInstanceKHR *accel_insts =
        reinterpret_cast<VkAccelerationStructureInstanceKHR  *>(
            buildStorage->ptr);

    for (int inst_idx = 0; inst_idx < new_num_instances; inst_idx++) {
        const ObjectInstance &inst = instances[inst_idx];
        const InstanceTransform &txfm = instance_transforms[inst_idx];

        VkAccelerationStructureInstanceKHR &inst_info =
            accel_insts[inst_idx];

        memcpy(&inst_info.transform,
               glm::value_ptr(glm::transpose(txfm.mat)),
               sizeof(VkTransformMatrixKHR));

        if (instance_flags[inst_idx] & InstanceFlags::Transparent) {
            inst_info.mask = 2;
        } else {
            inst_info.mask = 1;
        }
        inst_info.instanceCustomIndex = inst.materialOffset;
        inst_info.instanceShaderBindingTableRecordOffset = 
            objects[inst.objectIndex].meshIndex;
        inst_info.flags = 0;
        inst_info.accelerationStructureReference =
            blases.accelStructs[inst.objectIndex].devAddr;
    }

    buildStorage->flush(dev);

    VkBufferDeviceAddressInfo inst_build_addr_info {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        nullptr,
        buildStorage->buffer,
    };
    VkDeviceAddress inst_build_data_addr = 
        dev.dt.getBufferDeviceAddress(dev.hdl, &inst_build_addr_info);

    VkAccelerationStructureGeometryKHR tlas_geometry;
    tlas_geometry.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geometry.pNext = nullptr;
    tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geometry.flags = 0;
    auto &tlas_instances = tlas_geometry.geometry.instances;
    tlas_instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlas_instances.pNext = nullptr;
    tlas_instances.arrayOfPointers = false;
    tlas_instances.data.deviceAddress = inst_build_data_addr;

    VkAccelerationStructureBuildGeometryInfoKHR build_info;
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.pNext = nullptr;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.srcAccelerationStructure = VK_NULL_HANDLE;
    build_info.dstAccelerationStructure = VK_NULL_HANDLE;
    build_info.geometryCount = 1;
    build_info.pGeometries = &tlas_geometry;
    build_info.ppGeometries = nullptr;
    build_info.scratchData.deviceAddress = 0;

    VkAccelerationStructureBuildSizesInfoKHR size_info;
    size_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    size_info.pNext = nullptr;

    dev.dt.getAccelerationStructureBuildSizesKHR(dev.hdl,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &numBuildInstances, &size_info);

    size_t new_storage_bytes = size_info.accelerationStructureSize +
        size_info.buildScratchSize;

    if (new_storage_bytes > numStorageBytes) {
        numStorageBytes = new_storage_bytes;

        tlasStorage = alloc.makeLocalBuffer(numStorageBytes, true);

        if (!tlasStorage.has_value()) {
            cerr << "Failed to allocate TLAS storage" << endl;
            fatalExit();
        }
    }

    VkAccelerationStructureCreateInfoKHR create_info;
    create_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.pNext = nullptr;
    create_info.createFlags = 0;
    create_info.buffer = tlasStorage->buffer;
    create_info.offset = 0;
    create_info.size = size_info.accelerationStructureSize;
    create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    create_info.deviceAddress = 0;

    REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                 nullptr, &hdl));

    VkAccelerationStructureDeviceAddressInfoKHR accel_addr_info;
    accel_addr_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accel_addr_info.pNext = nullptr;
    accel_addr_info.accelerationStructure = hdl;

    tlasStorageDevAddr = dev.dt.getAccelerationStructureDeviceAddressKHR(
        dev.hdl, &accel_addr_info);

    VkBufferDeviceAddressInfoKHR storage_addr_info;
    storage_addr_info.sType =
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    storage_addr_info.pNext = nullptr;
    storage_addr_info.buffer = tlasStorage->buffer;
    VkDeviceAddress storage_base =
        dev.dt.getBufferDeviceAddress(dev.hdl, &storage_addr_info);

    build_info.dstAccelerationStructure = hdl;
    build_info.scratchData.deviceAddress =
        storage_base + size_info.accelerationStructureSize;

    VkAccelerationStructureBuildRangeInfoKHR range_info;
    range_info.primitiveCount = numBuildInstances;
    range_info.primitiveOffset = 0;
    range_info.firstVertex = 0;
    range_info.transformOffset = 0;
    const auto *range_info_ptr = &range_info;

    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd, 1, &build_info,
                                             &range_info_ptr);
}

void TLAS::free(const DeviceState &dev)
{
    dev.dt.destroyAccelerationStructureKHR(dev.hdl, hdl, nullptr);
}

SharedSceneState::SharedSceneState(const DeviceState &dev,
                                   VkDescriptorPool scene_pool,
                                   VkDescriptorSetLayout scene_layout,
                                   MemoryAllocator &alloc)
    : lock(),
      descSet(makeDescriptorSet(dev, scene_pool, scene_layout)),
      addrData([&]() {
          size_t num_addr_bytes = sizeof(SceneAddresses) * VulkanConfig::max_scenes;
          HostBuffer addr_data = alloc.makeParamBuffer(num_addr_bytes);

          VkDescriptorBufferInfo addr_buf_info {
              addr_data.buffer,
              0,
              num_addr_bytes,
          };

          DescriptorUpdates desc_update(1);
          desc_update.uniform(descSet, &addr_buf_info, 0);
          desc_update.update(dev);

          return addr_data;
      }()),
      freeSceneIDs(),
      numSceneIDs(0)
{}

SceneID::SceneID(SharedSceneState &shared)
    : shared_(&shared),
      id_([&]() {
          if (shared_->freeSceneIDs.size() > 0) {
              uint32_t id = shared_->freeSceneIDs.back();
              shared_->freeSceneIDs.pop_back();

              return id;
          } else {
              return shared_->numSceneIDs++;
          }
      }())
{}

SceneID::SceneID(SceneID &&o)
    : shared_(o.shared_),
      id_(o.id_)
{
    o.shared_ = nullptr;
}

SceneID::~SceneID()
{
    if (shared_ == nullptr) return;

    lock_guard<mutex> lock(shared_->lock);

    shared_->freeSceneIDs.push_back(id_);
}

shared_ptr<Scene> VulkanLoader::loadScene(SceneLoadData &&load_info)
{
    TextureData texture_store(dev, alloc);

    vector<LocalTexture> &gpu_textures = texture_store.textures;
    vector<VkImageView> &texture_views = texture_store.views;

    optional<StagedTextures> staged_textures = prepareSceneTextures(dev,
        load_info.textureInfo, max_texture_resolution_, alloc);

    uint32_t num_textures = staged_textures.has_value() ?
        staged_textures->textures.size() : 0;

    if (num_textures > 0) {
        texture_store.memory = staged_textures->texMemory;
        gpu_textures = move(staged_textures->textures);
        texture_views = move(staged_textures->textureViews);
    }

    // Copy all geometry into single buffer
    optional<LocalBuffer> data_opt =
        alloc.makeLocalBuffer(load_info.hdr.totalBytes, true);

    if (!data_opt.has_value()) {
        cerr << "Out of memory, failed to allocate geometry storage" << endl;
        fatalExit();
    }

    LocalBuffer data = move(data_opt.value());

    HostBuffer data_staging =
        alloc.makeStagingBuffer(load_info.hdr.totalBytes);

    if (holds_alternative<ifstream>(load_info.data)) {
        ifstream &file = *get_if<ifstream>(&load_info.data);
        file.read((char *)data_staging.ptr, load_info.hdr.totalBytes);
    } else {
        char *data_src = get_if<vector<char>>(&load_info.data)->data();
        memcpy(data_staging.ptr, data_src, load_info.hdr.totalBytes);
    }

    // Reset command buffers
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, transfer_cmd_pool_, 0));
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, render_cmd_pool_, 0));

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transfer_cmd_, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = load_info.hdr.totalBytes;
    dev.dt.cmdCopyBuffer(transfer_cmd_, data_staging.buffer, data.buffer,
                         1, &copy_settings);

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> texture_barriers(num_textures);
    for (size_t i = 0; i < num_textures; i++) {
        const LocalTexture &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = texture_barriers[i];

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = gpu_texture.image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, gpu_texture.mipLevels, 0, 1,
        };
    }

    if (num_textures > 0) {
        dev.dt.cmdPipelineBarrier(
            transfer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());

        // Record cpu -> gpu copies
        vector<VkBufferImageCopy> copy_infos;
        for (size_t i = 0; i < num_textures; i++) {
            const LocalTexture &gpu_texture = gpu_textures[i];
            uint32_t base_width = gpu_texture.width;
            uint32_t base_height = gpu_texture.height;
            uint32_t num_levels = gpu_texture.mipLevels;
            uint32_t texel_bytes = staged_textures->textureTexelBytes[i];
            copy_infos.resize(num_levels);
            uint32_t level_alignment = max(texel_bytes, 4u);

            size_t cur_lvl_offset = staged_textures->stageOffsets[i];

            for (uint32_t level = 0; level < num_levels; level++) {
                uint32_t level_div = 1 << level;
                uint32_t level_width = max(1U, base_width / level_div);
                uint32_t level_height = max(1U, base_height / level_div);

                cur_lvl_offset = alignOffset(cur_lvl_offset,
                                             level_alignment);

                // Set level copy
                VkBufferImageCopy copy_info {};
                copy_info.bufferOffset = cur_lvl_offset;
                copy_info.imageSubresource.aspectMask =
                    VK_IMAGE_ASPECT_COLOR_BIT;
                copy_info.imageSubresource.mipLevel = level;
                copy_info.imageSubresource.baseArrayLayer = 0;
                copy_info.imageSubresource.layerCount = 1;
                copy_info.imageExtent = {
                    level_width,
                    level_height,
                    1,
                };

                copy_infos[level] = copy_info;

                cur_lvl_offset += level_width * level_height *
                    texel_bytes;
            }

            dev.dt.cmdCopyBufferToImage(
                transfer_cmd_, staged_textures->stageBuffer.buffer,
                gpu_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                copy_infos.size(), copy_infos.data());
        }

        // Transfer queue relinquish texture barriers
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = render_qf_;
        }
    }

    // Transfer queue relinquish geometry
    VkBufferMemoryBarrier geometry_barrier;
    geometry_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barrier.pNext = nullptr;
    geometry_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barrier.dstAccessMask = 0;
    geometry_barrier.srcQueueFamilyIndex = dev.transferQF;
    geometry_barrier.dstQueueFamilyIndex = render_qf_;

    geometry_barrier.buffer = data.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = load_info.hdr.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(
        transfer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &geometry_barrier,
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transfer_cmd_));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.waitSemaphoreCount = 0;
    copy_submit.pWaitSemaphores = nullptr;
    copy_submit.pWaitDstStageMask = nullptr;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transfer_cmd_;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &transfer_sema_;

    transfer_queue_.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    // Start recording for transferring to rendering queue
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd_, &begin_info));

    // Finish moving geometry onto render queue family
    // geometry and textures need separate barriers due to different
    // dependent stages
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkPipelineStageFlags dst_geo_render_stage =
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    dev.dt.cmdPipelineBarrier(render_cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              dst_geo_render_stage, 0, 0, nullptr, 1,
                              &geometry_barrier, 0, nullptr);

    if (num_textures > 0) {
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = render_qf_;
        }

        // Finish acquiring mips on render queue and transition layout
        dev.dt.cmdPipelineBarrier(
            render_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());
    }

    VkBufferDeviceAddressInfo addr_info;
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.pNext = nullptr;
    addr_info.buffer = data.buffer;
    VkDeviceAddress geometry_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &addr_info);

    string blas_path =
        filesystem::path(load_info.scenePath).replace_extension("blas_cache");

    auto blas_result = getBLASes(dev, alloc,
                                 blas_path,
                                 load_info.meshInfo,
                                 load_info.objectInfo,
                                 load_info.hdr.numVertices,
                                 geometry_addr,
                                 geometry_addr + load_info.hdr.indexOffset,
                                 render_cmd_);

    if (!blas_result.has_value()) {
        cerr <<
            "OOM while constructing bottom level acceleration structures"
            << endl;
    }

    auto [blases, blas_scratch, blas_staging, total_blas_bytes,
          blases_rebuilt] =
        move(*blas_result);

    REQ_VK(dev.dt.endCommandBuffer(render_cmd_));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 1;
    render_submit.pWaitSemaphores = &transfer_sema_;
    VkPipelineStageFlags sema_wait_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    render_submit.pWaitDstStageMask = &sema_wait_mask;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &render_cmd_;

    render_queue_.submit(dev, 1, &render_submit, fence_);

    waitForFenceInfinitely(dev, fence_);
    resetFence(dev, fence_);

    // Free BLAS temporaries as early as possible
    blas_scratch.reset();
    blas_staging.reset();

    if (blases_rebuilt) {
        blases = compactBLASes(dev, alloc, blases, render_cmd_pool_,
            render_cmd_, render_queue_, fence_,
            compacted_query_pool_, max_queries_);

        cacheBLASes(dev, alloc, blas_path, blases, render_cmd_pool_,
                    render_cmd_, render_queue_, fence_,
                    serialized_query_pool_, max_queries_);
    }

    // Set Layout
    // 0: Scene addresses uniform
    // 1: textures

    DescriptorUpdates desc_updates(1);
    vector<VkDescriptorImageInfo> descriptor_views;
    descriptor_views.reserve(load_info.hdr.numMaterials * 8);

    VkDescriptorImageInfo null_img {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    for (int mat_idx = 0; mat_idx < (int)load_info.hdr.numMaterials;
         mat_idx++) {
        const MaterialTextures &tex_indices =
            load_info.textureIndices[mat_idx];

        auto appendDescriptor = [&](uint32_t idx, const auto &texture_list) {
            if (idx != ~0u) {
                VkImageView tex_view = texture_views[texture_list[idx]];

                descriptor_views.push_back({
                    VK_NULL_HANDLE,
                    tex_view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
            } else {
                descriptor_views.push_back(null_img);
            }
        };

        appendDescriptor(tex_indices.baseColorIdx, staged_textures->base);
        appendDescriptor(tex_indices.metallicRoughnessIdx,
                         staged_textures->metallicRoughness);
        appendDescriptor(tex_indices.specularIdx, staged_textures->specular);
        appendDescriptor(tex_indices.normalIdx, staged_textures->normal);
        appendDescriptor(tex_indices.emittanceIdx, staged_textures->emittance);
        appendDescriptor(tex_indices.transmissionIdx,
                         staged_textures->transmission);
        appendDescriptor(tex_indices.clearcoatIdx, staged_textures->clearcoat);
        appendDescriptor(tex_indices.anisoIdx, staged_textures->anisotropic);
    }

    optional<SceneID> scene_id_tracker;
    uint32_t scene_id;
    VkDescriptorBufferInfo vert_info;
    VkDescriptorBufferInfo mat_info;
    if (shared_scene_state_) {
        shared_scene_state_->lock.lock();
        scene_id_tracker.emplace(*shared_scene_state_);
        scene_id = scene_id_tracker->getID();
    } else {
        // FIXME, this entire special codepath for the editor needs to be
        // removed
        scene_id = 0;

        vert_info.buffer = data.buffer;
        vert_info.offset = 0;
        vert_info.range =
            load_info.hdr.numVertices * sizeof(PackedVertex);

        desc_updates.storage(scene_set_, &vert_info, 0);

        mat_info.buffer = data.buffer;
        mat_info.offset = load_info.hdr.materialOffset;
        mat_info.range = load_info.hdr.numMaterials * sizeof(MaterialParams);

        desc_updates.storage(scene_set_, &mat_info, 2);
    }

    if (load_info.hdr.numMaterials > 0) {
        assert(load_info.hdr.numMaterials < VulkanConfig::max_materials);

        uint32_t texture_offset = scene_id * VulkanConfig::max_materials *
             VulkanConfig::textures_per_material;
        desc_updates.textures(scene_set_,
                              descriptor_views.data(),
                              descriptor_views.size(), 1,
                              texture_offset);
    }

    desc_updates.update(dev);

    if (shared_scene_state_) {
        SceneAddresses &scene_dev_addrs = 
            ((SceneAddresses *)shared_scene_state_->addrData.ptr)[scene_id];
        scene_dev_addrs.vertAddr = geometry_addr;
        scene_dev_addrs.idxAddr = geometry_addr + load_info.hdr.indexOffset;
        scene_dev_addrs.matAddr = geometry_addr + load_info.hdr.materialOffset;
        scene_dev_addrs.meshAddr = geometry_addr + load_info.hdr.meshOffset;
        shared_scene_state_->addrData.flush(dev);
        shared_scene_state_->lock.unlock();
    }

    uint32_t num_meshes = load_info.meshInfo.size();

    return make_shared<VulkanScene>(VulkanScene {
        {
            move(load_info.meshInfo),
            move(load_info.objectInfo),
            move(load_info.envInit),
        },
        move(texture_store),
        move(data),
        load_info.hdr.indexOffset,
        num_meshes,
        move(scene_id_tracker),
        move(blases),
    });
}

shared_ptr<EnvironmentMapGroup> VulkanLoader::loadEnvironmentMaps(
    const char **paths, uint32_t num_paths)
{
    vector<void *> host_ptrs;
    vector<uint64_t> num_host_bytes;
    vector<uint64_t> host_offsets;
    vector<uint64_t> dev_offsets;
    vector<LocalTexture> gpu_textures;
    vector<VkImageView> views;
    host_ptrs.reserve(num_paths * 2);
    num_host_bytes.reserve(num_paths * 2);
    host_offsets.reserve(num_paths * 2);
    dev_offsets.reserve(num_paths * 2);
    gpu_textures.reserve(num_paths * 2);
    views.reserve(num_paths * 2);
    
    auto env_fmt = alloc.getTextureFormat(TextureFormat::R32G32B32A32_SFLOAT);
    auto imp_fmt = alloc.getTextureFormat(TextureFormat::R32_SFLOAT);

    auto env_texel_bytes = getTexelBytes(TextureFormat::R32G32B32A32_SFLOAT);
    auto imp_texel_bytes = getTexelBytes(TextureFormat::R32_SFLOAT);

    uint64_t cur_host_offset = 0;
    uint64_t cur_dev_offset = 0;
    for (int i = 0; i < (int)num_paths; i++) {
        auto [env_data, env_data_bytes, env_dims,
              imp_data, imp_data_bytes, imp_dims] =
            loadEnvironmentMapFromDisk(paths[i]);

        auto [env_gpu_tex, env_tex_reqs] = alloc.makeTexture2D(
            env_dims.x, env_dims.y, env_dims.z, env_fmt);

        auto [imp_gpu_tex, imp_tex_reqs] = alloc.makeTexture2D(
            imp_dims.x, imp_dims.y, imp_dims.z, imp_fmt);

        host_ptrs.push_back(env_data);
        host_ptrs.push_back(imp_data);
        num_host_bytes.push_back(env_data_bytes);
        num_host_bytes.push_back(imp_data_bytes);
        gpu_textures.emplace_back(move(env_gpu_tex));
        gpu_textures.emplace_back(move(imp_gpu_tex));

        cur_host_offset =
            alignOffset(cur_host_offset, max(env_texel_bytes, 4u));
        host_offsets.push_back(cur_host_offset);
        cur_host_offset += env_data_bytes;

        cur_host_offset =
            alignOffset(cur_host_offset, max(imp_texel_bytes, 4u));
        host_offsets.push_back(cur_host_offset);
        cur_host_offset += imp_data_bytes;

        cur_dev_offset = alignOffset(cur_dev_offset, env_tex_reqs.alignment);
        dev_offsets.push_back(cur_dev_offset);
        cur_dev_offset += env_tex_reqs.size;

        cur_dev_offset = alignOffset(cur_dev_offset, imp_tex_reqs.alignment);
        dev_offsets.push_back(cur_dev_offset);
        cur_dev_offset += env_tex_reqs.size;
    }

    HostBuffer staging_buffer = alloc.makeStagingBuffer(cur_host_offset);

    const uint64_t num_device_bytes = cur_dev_offset;

    optional<VkDeviceMemory> tex_mem_opt = alloc.alloc(num_device_bytes);
    if (!tex_mem_opt.has_value()) {
        cerr << "Out of device memory for environment maps" << endl;
        fatalExit();
    }

    VkDeviceMemory tex_mem = tex_mem_opt.value();

    // Bind image memory and create views
    for (int pair_idx = 0; pair_idx < (int)num_paths; pair_idx++) {
        int env_idx = pair_idx * 2;
        int imp_idx = env_idx + 1;
        LocalTexture &env_texture = gpu_textures[env_idx];
        VkDeviceSize env_offset = dev_offsets[env_idx];

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, env_texture.image,
                                      tex_mem, env_offset));

        LocalTexture &imp_texture = gpu_textures[imp_idx];
        VkDeviceSize imp_offset = dev_offsets[imp_idx];

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, imp_texture.image,
                                      tex_mem, imp_offset));

        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = env_texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = env_fmt;
        view_info.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A,
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            env_texture.mipLevels, 
            0,
            1,
        };

        VkImageView env_view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr,
                                      &env_view));

        views.push_back(env_view);

        view_info.image = imp_texture.image;
        view_info.format = imp_fmt;
        view_info.subresourceRange.levelCount = imp_texture.mipLevels;

        VkImageView imp_view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr,
                                      &imp_view));

        views.push_back(imp_view);
    }

    int num_textures = (int)views.size();

    for (int i = 0; i < num_textures; i++) {
        memcpy((char *)staging_buffer.ptr + host_offsets[i],
               host_ptrs[i], num_host_bytes[i]);

        free(host_ptrs[i]);
    }

    staging_buffer.flush(dev);

    // Reset command buffers
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, transfer_cmd_pool_, 0));
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, render_cmd_pool_, 0));

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transfer_cmd_, &begin_info));

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> texture_barriers(num_textures);
    for (int i = 0; i < num_textures; i++) {
        const LocalTexture &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = texture_barriers[i];

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = gpu_texture.image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, gpu_texture.mipLevels, 0, 1,
        };
    }

    dev.dt.cmdPipelineBarrier(
        transfer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
        texture_barriers.size(), texture_barriers.data());

    // Record cpu -> gpu copies
    vector<VkBufferImageCopy> copy_infos;

    auto recordCopy = [&](const LocalTexture &gpu_texture,
                          uint64_t base_staging_offset,
                          uint32_t texel_bytes) {
        uint32_t base_width = gpu_texture.width;
        uint32_t base_height = gpu_texture.height;
        uint32_t num_levels = gpu_texture.mipLevels;
        copy_infos.resize(num_levels);
        uint32_t level_alignment = max(texel_bytes, 4u);

        size_t cur_lvl_offset = base_staging_offset;

        for (uint32_t level = 0; level < num_levels; level++) {
            uint32_t level_div = 1 << level;
            uint32_t level_width = max(1U, base_width / level_div);
            uint32_t level_height = max(1U, base_height / level_div);

            cur_lvl_offset = alignOffset(cur_lvl_offset,
                                         level_alignment);

            // Set level copy
            VkBufferImageCopy copy_info {};
            copy_info.bufferOffset = cur_lvl_offset;
            copy_info.imageSubresource.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;
            copy_info.imageSubresource.mipLevel = level;
            copy_info.imageSubresource.baseArrayLayer = 0;
            copy_info.imageSubresource.layerCount = 1;
            copy_info.imageExtent = {
                level_width,
                level_height,
                1,
            };

            copy_infos[level] = copy_info;

            cur_lvl_offset += level_width * level_height *
                texel_bytes;
        }

        dev.dt.cmdCopyBufferToImage(
            transfer_cmd_, staging_buffer.buffer,
            gpu_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            copy_infos.size(), copy_infos.data());
    };

    for (int i = 0; i < (int)num_paths; i++) {
        int env_idx = 2 * i;
        int imp_idx = 2 * i + 1;
        recordCopy(gpu_textures[env_idx], host_offsets[env_idx],
                   env_texel_bytes);
        recordCopy(gpu_textures[imp_idx], host_offsets[imp_idx],
                   imp_texel_bytes);
    }

    // Transfer queue relinquish texture barriers
    for (VkImageMemoryBarrier &barrier : texture_barriers) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = dev.transferQF;
        barrier.dstQueueFamilyIndex = render_qf_;
    }

    dev.dt.cmdPipelineBarrier(
        transfer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transfer_cmd_));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.waitSemaphoreCount = 0;
    copy_submit.pWaitSemaphores = nullptr;
    copy_submit.pWaitDstStageMask = nullptr;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transfer_cmd_;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &transfer_sema_;

    transfer_queue_.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    // Start recording for transferring to rendering queue
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd_, &begin_info));
    for (VkImageMemoryBarrier &barrier : texture_barriers) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = dev.transferQF;
        barrier.dstQueueFamilyIndex = render_qf_;
    }

    // Finish acquiring mips on render queue and transition layout
    dev.dt.cmdPipelineBarrier(
        render_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(render_cmd_));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 1;
    render_submit.pWaitSemaphores = &transfer_sema_;
    VkPipelineStageFlags sema_wait_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    render_submit.pWaitDstStageMask = &sema_wait_mask;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &render_cmd_;

    render_queue_.submit(dev, 1, &render_submit, fence_);
    waitForFenceInfinitely(dev, fence_);
    resetFence(dev, fence_);

    vector<VkDescriptorImageInfo> desc_views;
    desc_views.reserve(views.size());

    for (VkImageView view : views) {
        desc_views.push_back({
            VK_NULL_HANDLE,
            view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
    }

    DescriptorSet desc_set = env_map_pool_.makeSet();

    DescriptorUpdates desc_updates(1);

    desc_updates.textures(desc_set.hdl,
                          desc_views.data(),
                          desc_views.size(), 0, 0);

    desc_updates.update(dev);

    auto hdl = std::shared_ptr<VulkanEnvMapGroup>(new VulkanEnvMapGroup {
        {},
        TextureData(dev, alloc),
        move(desc_set),
    });

    hdl->texData.memory = tex_mem;
    hdl->texData.textures = move(gpu_textures);
    hdl->texData.views = move(views);

    return hdl;
}

}
}
