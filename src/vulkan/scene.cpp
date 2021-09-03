#include "scene.hpp"
#include <vulkan/vulkan_core.h>

#include "rlpbr_core/utils.hpp"
#include "shader.hpp"
#include "utils.hpp"

#include <glm/gtc/type_ptr.hpp>
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

VulkanEnvironment::VulkanEnvironment(const DeviceState &d,
                                     MemoryAllocator &alloc,
                                     const VulkanScene &scene,
                                     const Camera &cam)
    : EnvironmentBackend {},
      lights(),
      dev(d),
      tlas(),
      reservoirGrid(makeReservoirGrid(dev, alloc, scene)),
      prevCam(cam)
{
    for (const LightProperties &light : scene.envInit.lights) {
        PackedLight packed;
        memcpy(&packed.data[0].x, &light.type, sizeof(uint32_t));
        if (light.type == LightType::Point) {
            packed.data[0].y = light.color[0];
            packed.data[0].z = light.color[1];
            packed.data[0].w = light.color[2];
            packed.data[1].x = light.position[0];
            packed.data[1].y = light.position[1];
            packed.data[1].z = light.position[2];
        } else if (light.type == LightType::Portal) {
            packed.data[1].x = light.corners[0][0];
            packed.data[1].y = light.corners[0][1];
            packed.data[1].z = light.corners[0][2];
            packed.data[1].w = light.corners[1][0];
            packed.data[2].x = light.corners[1][1];
            packed.data[2].y = light.corners[1][2];
            packed.data[2].z = light.corners[2][0];
            packed.data[2].w = light.corners[2][1];
            packed.data[3].x = light.corners[2][2];
            packed.data[3].y = light.corners[3][0];
            packed.data[3].z = light.corners[3][1];
            packed.data[3].w = light.corners[3][2];
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
    lights.push_back(PackedLight {
        glm::vec4(position, 1.f),
        glm::vec4(color, 1.f),
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
                           const ShaderPipeline &shader,
                           const SceneDescriptorBindings &binding_defn,
                           uint32_t render_qf,
                           uint32_t max_texture_resolution)
    : dev(d),
      alloc(alc),
      transfer_queue_(transfer_queue),
      render_queue_(render_queue),
      transfer_cmd_pool_(makeCmdPool(d, d.transferQF)),
      transfer_cmd_(makeCmdBuffer(dev, transfer_cmd_pool_)),
      render_cmd_pool_(makeCmdPool(d, render_qf)),
      render_cmd_(makeCmdBuffer(dev, render_cmd_pool_)),
      transfer_sema_(makeBinarySemaphore(dev)),
      fence_(makeFence(dev)),
      desc_mgr_(dev, shader, 1),
      binding_defn_(binding_defn),
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

        cur_offset = alignOffset(cur_offset, texel_bytes);

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

    free(compressed_data);

    return make_tuple(img_data, num_decompressed_bytes, glm::u32vec2(x, y),
                      num_levels, -float(num_skip_levels));
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
    optional<uint32_t> envMap;
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

    if (!texture_info.envMap.empty()) {
        num_textures += 1;
    }

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

    optional<uint32_t> env_loc;

    if (!texture_info.envMap.empty()) {
        int x, y, n;
        float *img_data = stbi_loadf(
            (texture_info.textureDir + texture_info.envMap).c_str(),
            &x, &y, &n, 4);

        env_loc = stageTexture(img_data, x * y * 4 * sizeof(float), x, y, 1,
            alloc.getTextureFormat(TextureFormat::R32G32B32A32_SFLOAT),
            getTexelBytes(TextureFormat::R32G32B32A32_SFLOAT));
    }

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
        stbi_image_free(host_ptrs[i]);
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
        move(env_loc),
    };
}

BLASData::~BLASData()
{
    for (const auto &blas : accelStructs) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, blas.hdl,
                                               nullptr);
    }
}

static optional<tuple<BLASData, LocalBuffer, VkDeviceSize>> makeBLASes(
    const DeviceState &dev,
    MemoryAllocator &alloc, 
    const std::vector<MeshInfo> &meshes,
    const std::vector<ObjectInfo> &objects,
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
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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

    return make_tuple(
        BLASData {
            dev,
            move(accel_structs),
            move(accel_mem),
        },
        move(scratch_mem),
        total_accel_bytes);
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
    tlas_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
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

            size_t cur_lvl_offset = staged_textures->stageOffsets[i];

            for (uint32_t level = 0; level < num_levels; level++) {
                uint32_t level_div = 1 << level;
                uint32_t level_width = max(1U, base_width / level_div);
                uint32_t level_height = max(1U, base_height / level_div);


                cur_lvl_offset = alignOffset(cur_lvl_offset, texel_bytes);

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

    auto blas_result = makeBLASes(dev, alloc, 
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

    auto [blases, scratch, total_blas_bytes] = move(*blas_result);

    // Repurpose geometry_barrier for blas barrier
    geometry_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    geometry_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT;
    geometry_barrier.srcQueueFamilyIndex = render_qf_;
    geometry_barrier.dstQueueFamilyIndex = render_qf_;
    geometry_barrier.buffer = blases.storage.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = total_blas_bytes;

    dev.dt.cmdPipelineBarrier(
        render_cmd_, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        0, 0, nullptr,
        1, &geometry_barrier,
        0, nullptr);

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

    DescriptorSet scene_set = desc_mgr_.makeSet();

    DescriptorUpdates desc_updates(5);

    // Set Layout
    // 0: Vertex buffer
    // 1: Index buffer
    // 2: textures
    // 3: material params
    // 4: mesh infos

    VkDescriptorBufferInfo vertex_buffer_info {
        data.buffer,
        0,
        load_info.hdr.numVertices * sizeof(Vertex),
    };

    if (binding_defn_.vertexBinding != ~0u) {
        desc_updates.storage(scene_set.hdl, &vertex_buffer_info,
                             binding_defn_.vertexBinding);
    }

    VkDescriptorBufferInfo index_buffer_info {
        data.buffer,
        load_info.hdr.indexOffset,
        load_info.hdr.numIndices * sizeof(uint32_t),
    };

    if (binding_defn_.indexBinding != ~0u) {
        desc_updates.storage(scene_set.hdl, &index_buffer_info,
                             binding_defn_.indexBinding);
    }

    vector<VkDescriptorImageInfo> descriptor_views;
    descriptor_views.reserve(load_info.hdr.numMaterials * 8 + 1);

    if (staged_textures->envMap.has_value()) {
        descriptor_views.push_back({
            VK_NULL_HANDLE,
            texture_views[staged_textures->envMap.value()],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
    } else {
        descriptor_views.push_back({
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
    }

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
                descriptor_views.push_back({
                    VK_NULL_HANDLE,
                    VK_NULL_HANDLE,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
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

    VkDescriptorBufferInfo material_buffer_info;

    if (load_info.hdr.numMaterials > 0) {
        assert(load_info.hdr.numMaterials < VulkanConfig::max_materials);

        if (binding_defn_.textureBinding != ~0u) {
            desc_updates.textures(scene_set.hdl, descriptor_views.data(),
                descriptor_views.size(), binding_defn_.textureBinding);
        }

        material_buffer_info.buffer = data.buffer;
        material_buffer_info.offset = load_info.hdr.materialOffset;
        material_buffer_info.range =
            load_info.hdr.numMaterials * sizeof(MaterialParams);

        if (binding_defn_.materialBinding != ~0u) {
            desc_updates.storage(scene_set.hdl, &material_buffer_info,
                                 binding_defn_.materialBinding);
        }
    }

    VkDescriptorBufferInfo mesh_buffer_info {
        data.buffer,
        load_info.hdr.meshOffset,
        load_info.hdr.numMeshes * sizeof(MeshInfo),
    };

    if (binding_defn_.meshInfoBinding != ~0u) {
        desc_updates.storage(scene_set.hdl, &mesh_buffer_info,
                             binding_defn_.meshInfoBinding);
    }

    desc_updates.update(dev);

    uint32_t num_meshes = load_info.meshInfo.size();

    return make_shared<VulkanScene>(VulkanScene {
        {
            move(load_info.meshInfo),
            move(load_info.objectInfo),
            move(load_info.envInit),
        },
        move(texture_store),
        move(scene_set),
        move(data),
        load_info.hdr.indexOffset,
        num_meshes,
        move(blases),
    });
}

}
}
