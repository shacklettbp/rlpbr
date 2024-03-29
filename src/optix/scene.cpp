#include "scene.hpp"
#include "utils.hpp"

#include <optix_stubs.h>
#include <iostream>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

namespace RLpbr {
namespace optix {

OptixScene::~OptixScene()
{
    defaultTLAS.free();

    for (auto blas_ptr : blasStorage) {
        cudaFree((void *)blas_ptr);
    }

    cudaFree((void *)sceneStorage);
}

static void assignInstanceTransform(OptixInstance &inst,
                                    const glm::mat4x3 &mat)
{
    for (uint32_t col = 0; col < 4; col++) {
        for (uint32_t row = 0; row < 3; row++) {
            inst.transform[row * 4 + col] = mat[col][row];
        }
    }
}

void TLAS::build(
    OptixDeviceContext ctx,
    const vector<ObjectInstance> &instances,
    const vector<InstanceTransform> &instance_transforms,
    const vector<InstanceFlags> &instance_flags,
    const OptixTraversableHandle *blases,
    cudaStream_t build_stream)
{
    bool update = hdl != 0;

    uint32_t new_num_instances = instances.size();

    if (true || numBuildInstances != new_num_instances) {
        update = false;
        hdl = 0;
    }

    if (numBuildInstances < new_num_instances) {
        if (instanceBuildBuffer != nullptr) {
            REQ_CUDA(cudaFree(instanceBuildBuffer));
        }
        if (instanceBLASes != nullptr) {
            REQ_CUDA(cudaFree(instanceBLASes));
        }

        REQ_CUDA(cudaMalloc(&instanceBuildBuffer,
                            sizeof(OptixInstance) * new_num_instances));

        REQ_CUDA(cudaMalloc(&instanceBLASes,
                            sizeof(OptixTraversableHandle) * new_num_instances));

        numBuildInstances = new_num_instances;
    }

    instanceStaging.clear();
    instanceBLASStaging.clear();
    for (int inst_id = 0; inst_id < (int)new_num_instances; inst_id++) {
        const ObjectInstance &inst = instances[inst_id];
        const InstanceTransform &txfm = instance_transforms[inst_id];
        OptixInstance cur_inst;

        assignInstanceTransform(cur_inst, txfm.mat);
        cur_inst.instanceId = 0;
        cur_inst.sbtOffset = 0;
        if (instance_flags[inst_id] & InstanceFlags::Transparent) {
            cur_inst.visibilityMask = 2;
        } else {
            cur_inst.visibilityMask = 1;
        }
        cur_inst.flags = 0;
        cur_inst.traversableHandle = blases[inst.objectIndex];

        instanceStaging.push_back(cur_inst);
        instanceBLASStaging.push_back(blases[inst.objectIndex]);
    }

    cudaMemcpyAsync(instanceBuildBuffer, instanceStaging.data(),
                    new_num_instances * sizeof(OptixInstance),
                    cudaMemcpyHostToDevice, build_stream);

    cudaMemcpyAsync(instanceBLASes, instanceBLASStaging.data(),
                    new_num_instances * sizeof(OptixTraversableHandle),
                    cudaMemcpyHostToDevice, build_stream);

    OptixBuildInput tlas_build {};
    tlas_build.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray &tlas_instances = tlas_build.instanceArray;
    tlas_instances.instances = (CUdeviceptr)instanceBuildBuffer;
    tlas_instances.numInstances = new_num_instances;

    OptixAccelBuildOptions tlas_options;
    tlas_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    tlas_options.motionOptions = {};

    if (update) {
        tlas_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        tlas_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    OptixAccelBufferSizes tlas_buffer_sizes;
    REQ_OPTIX(optixAccelComputeMemoryUsage(ctx, &tlas_options, &tlas_build, 1,
                                           &tlas_buffer_sizes));
    
    if (numTLASBytes < tlas_buffer_sizes.outputSizeInBytes) {
        if (tlasStorage != nullptr) {
            REQ_CUDA(cudaFree(tlasStorage));
        }
        REQ_CUDA(cudaMalloc(&tlasStorage, tlas_buffer_sizes.outputSizeInBytes));

        numTLASBytes = tlas_buffer_sizes.outputSizeInBytes;
    }

    size_t new_num_scratch_bytes;
    if (update) {
        new_num_scratch_bytes = tlas_buffer_sizes.tempUpdateSizeInBytes;
    } else {
        new_num_scratch_bytes = tlas_buffer_sizes.tempSizeInBytes;
    }

    if (numScratchBytes < new_num_scratch_bytes) {
        if (scratchStorage != nullptr) {
            REQ_CUDA(cudaFree(scratchStorage));
        }
        REQ_CUDA(cudaMalloc(&scratchStorage, new_num_scratch_bytes));

        numScratchBytes = new_num_scratch_bytes;
    }

    REQ_OPTIX(optixAccelBuild(ctx, build_stream, &tlas_options, &tlas_build,
        1, (CUdeviceptr)scratchStorage,
        update ? tlas_buffer_sizes.tempUpdateSizeInBytes :
                 tlas_buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)tlasStorage, tlas_buffer_sizes.outputSizeInBytes, &hdl,
        nullptr, 0));
}

void TLAS::free()
{
    cudaFree(instanceBLASes);
    cudaFree(instanceBuildBuffer);

    cudaFree(scratchStorage);
    cudaFree(tlasStorage);
}

OptixEnvironment OptixEnvironment::make(OptixDeviceContext ctx,
                                        cudaStream_t build_stream,
                                        const OptixScene &scene)
{
    uint32_t num_instances = scene.envInit.defaultInstances.size();

    TLAS new_tlas {};

#if 0
    new_tlas.tlasStorage = allocCU(scene.defaultTLAS.numTLASBytes);
    cudaMemcpyAsync(new_tlas.tlasStorage,
                    (void *)scene.defaultTLAS.tlasStorage,
                    scene.defaultTLAS.numTLASBytes, cudaMemcpyDeviceToDevice,
                    build_stream);
    new_tlas.numTLASBytes = scene.defaultTLAS.numTLASBytes;

    OptixAccelRelocationInfo tlas_reloc_info {};
    REQ_OPTIX(optixAccelGetRelocationInfo(ctx, scene.defaultTLAS.hdl,
                                          &tlas_reloc_info));

    REQ_OPTIX(optixAccelRelocate(ctx, build_stream, &tlas_reloc_info,
        (CUdeviceptr)scene.defaultTLAS.instanceBLASes,
        num_instances, (CUdeviceptr)new_tlas.tlasStorage,
        scene.defaultTLAS.numTLASBytes, &new_tlas.hdl));
#endif
    (void)ctx;
    (void)num_instances;

    // FIXME, pre-pack this in envInit somehow... backend
    // env init?
    vector<PackedLight> lights;
    lights.reserve(scene.envInit.lights.size());

    for (const auto &light : scene.envInit.lights) {
        PackedLight packed;
        memcpy(&packed.data[0].x, &light.type, sizeof(uint32_t));
        if (light.type == LightType::Sphere) {
            // FIXME
        } 
        lights.push_back(packed);
    }

    PackedLight *light_buffer =
        (PackedLight *)allocCU(sizeof(PackedLight) * lights.size());
    cudaMemcpyAsync(light_buffer, lights.data(),
                    sizeof(PackedLight) * lights.size(),
                    cudaMemcpyHostToDevice, build_stream);

    REQ_CUDA(cudaStreamSynchronize(build_stream));

    optional<PhysicsEnvironment> physics;
    if (scene.physics.has_value()) {
        physics.emplace(*scene.physics, build_stream);
    }

    return OptixEnvironment {
        {},
        new_tlas,
        light_buffer,
        uint32_t(lights.size()),
        move(physics),
    };
}

OptixEnvironment::~OptixEnvironment()
{
    tlas.free();
    cudaFree(lights);
}

uint32_t OptixEnvironment::addLight(const glm::vec3 &position,
                  const glm::vec3 &color)
{
    (void)position;
    (void)color;
    return 0;
#if 0
    PackedLight packed;
    LightType type = LightType::Point;
    memcpy(&packed.data[0].x, &type, sizeof(LightType));
    packed.data[0].y = color.r;
    packed.data[0].z = color.g;
    packed.data[0].w = color.b;
    packed.data[1].x = position.x;
    packed.data[1].y = position.y;
    packed.data[1].z = position.z;

    lights.push_back(packed);

    return lights.size() - 1;
#endif
}

void OptixEnvironment::removeLight(uint32_t light_idx)
{
    (void)light_idx;
#if 0
    lights[light_idx] = lights.back();

    lights.pop_back();
#endif
}

void OptixEnvironment::queueTLASRebuild(const Environment &env,
    OptixDeviceContext ctx, cudaStream_t strm)
{
    const OptixScene &scene = 
        *static_cast<const OptixScene *>(env.getScene().get());

    tlas.build(ctx, env.getInstances(),
        env.getTransforms(), env.getInstanceFlags(),
        scene.blases.data(), strm);
}

OptixLoader::OptixLoader(OptixDeviceContext ctx, TextureManager &texture_mgr,
                         uint32_t max_texture_resolution, bool need_physics)
    : stream_([]() {
          cudaStream_t strm;
          REQ_CUDA(cudaStreamCreate(&strm));
          return strm;
      }()),
      ctx_(ctx),
      texture_mgr_(texture_mgr),
      max_texture_resolution_(max_texture_resolution),
      need_physics_(need_physics)
{}

static LoadedTextures loadTextures(const TextureInfo &texture_info,
                                   cudaStream_t cpy_strm,
                                   uint32_t max_texture_resolution,
                                   TextureManager &mgr)
{
    vector<void *> host_tex_data;

    auto loadTextureList = [&](const vector<string> &texture_names,
                               TextureFormat fmt) {
        vector<Texture> gpu_textures;
        gpu_textures.reserve(texture_names.size());

        for (const auto &tex_name : texture_names) {
            string full_path = texture_info.textureDir + tex_name;

            Texture tex = mgr.load(full_path, fmt, cudaAddressModeWrap,
                cpy_strm, [&](const string &tex_path) {
                    ifstream tex_file(tex_path, ios::in | ios::binary);

                    auto read_uint = [&tex_file]() {
                        uint32_t v;
                        tex_file.read((char *)&v, sizeof(uint32_t));
                        return v;
                    };

                    uint32_t bytes_per_pixel;
                    if (fmt == TextureFormat::R8G8B8A8_SRGB) {
                        bytes_per_pixel = 4;
                    } else if (fmt == TextureFormat::R8G8B8A8_UNORM) {
                        bytes_per_pixel = 4;
                    } else if (fmt == TextureFormat::R8G8_UNORM) {
                        bytes_per_pixel = 2;
                    } else {
                        cerr << "Invalid texture format" << endl;
                        abort();
                    }

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
                            level_x * level_y * bytes_per_pixel;
                        num_compressed_bytes += lvl_compressed_bytes;
                    }

                    int num_levels = total_num_levels - num_skip_levels;

                    uint8_t *img_data = (uint8_t *)malloc(num_decompressed_bytes);
                    tex_file.ignore(skip_bytes);

                    uint8_t *compressed_data =
                        (uint8_t *)malloc(num_compressed_bytes);
                    tex_file.read((char *)compressed_data, num_compressed_bytes);

                    uint8_t *cur_ptr = img_data;
                    for (int i = 0; i < (int)num_levels; i++) {
                        auto [offset, num_bytes] = png_pos[i];
                        int lvl_x, lvl_y, tmp_n;
                        uint8_t *decompressed = stbi_load_from_memory(
                            compressed_data + offset, num_bytes,
                            &lvl_x, &lvl_y, &tmp_n, 4);

                        for (int pix_idx = 0; pix_idx < int(lvl_x * lvl_y);
                             pix_idx++) {
                            uint8_t *decompressed_offset =
                                decompressed + pix_idx * 4;
                            uint8_t *out_offset =
                                cur_ptr + pix_idx * bytes_per_pixel;

                            for (int byte_idx = 0; byte_idx < (int)bytes_per_pixel;
                                 byte_idx++) {
                                out_offset[byte_idx] =
                                    decompressed_offset[byte_idx];
                            }
                        }
                        free(decompressed);

                        cur_ptr += lvl_x * lvl_y * bytes_per_pixel;
                    }

                    free(compressed_data);
                    host_tex_data.push_back(img_data);

                    return make_tuple(img_data, glm::u32vec2(x, y),
                                      num_levels, -float(num_skip_levels));
                });

            gpu_textures.emplace_back(move(tex));
        }

        return gpu_textures;
    };

    LoadedTextures loaded;
    loaded.base = loadTextureList(texture_info.base,
                                  TextureFormat::R8G8B8A8_SRGB);
    loaded.metallicRoughness = loadTextureList(texture_info.metallicRoughness,
                                               TextureFormat::R8G8_UNORM);
    loaded.specular = loadTextureList(texture_info.specular,
                                      TextureFormat::R8G8B8A8_SRGB);
    loaded.normal = loadTextureList(texture_info.normal,
                                    TextureFormat::R8G8_UNORM);
    loaded.emittance = loadTextureList(texture_info.emittance,
                                       TextureFormat::R8G8B8A8_SRGB);
    loaded.transmission = loadTextureList(texture_info.transmission,
                                          TextureFormat::R8_UNORM);
    loaded.clearcoat = loadTextureList(texture_info.clearcoat,
                                       TextureFormat::R8G8_UNORM);
    loaded.anisotropic = loadTextureList(texture_info.anisotropic,
                                         TextureFormat::R8G8_UNORM);

    if (!texture_info.envMap.empty()) {
        loaded.envMap.emplace(mgr.load(
            texture_info.textureDir + texture_info.envMap,
            TextureFormat::R32G32B32A32_SFLOAT, cudaAddressModeWrap,
            cpy_strm, [&](const string &tex_path) {
                int x, y, n;
                float *img_data =
                    stbi_loadf(tex_path.c_str(), &x, &y, &n, 4);
                host_tex_data.push_back(img_data);
                return make_tuple(img_data, glm::u32vec2(x, y), 1, 0.f);
            }));
    }

    REQ_CUDA(cudaStreamSynchronize(cpy_strm));

    for (void *ptr : host_tex_data) {
        free(ptr);
    }

    return loaded;
}

shared_ptr<Scene> OptixLoader::loadScene(SceneLoadData &&load_info)
{
    auto textures = loadTextures(load_info.textureInfo, stream_,
                                 max_texture_resolution_, texture_mgr_);

    size_t num_texture_hdls = load_info.hdr.numMaterials *
        TextureConstants::numTexturesPerMaterial;
    if (textures.envMap.has_value()) {
        num_texture_hdls++;
    }
    size_t texture_ptr_offset = alignOffset(load_info.hdr.totalBytes, 16);
    size_t texture_ptr_bytes = num_texture_hdls * sizeof(cudaTextureObject_t);
    size_t texture_dim_offset =
        alignOffset(texture_ptr_offset + texture_ptr_bytes, 16);
    size_t texture_dim_bytes = num_texture_hdls * sizeof(TextureSize);
    size_t sdf_ptr_offset = 0;
    size_t sdf_ptr_bytes = 0;
    size_t total_device_bytes = 0;
    if (need_physics_) {
        sdf_ptr_offset =
            alignOffset(texture_dim_offset + texture_dim_bytes, 16);
        sdf_ptr_bytes =
            load_info.physics.sdfPaths.size() * sizeof(cudaTextureObject_t);

        total_device_bytes = sdf_ptr_offset + sdf_ptr_bytes;
    } else {
        total_device_bytes = texture_dim_offset + texture_dim_bytes;
    }

    char *scene_storage = (char *)allocCU(total_device_bytes);

    char *data_src = nullptr;
    bool cuda_staging = false;

    if (holds_alternative<ifstream>(load_info.data)) {
        REQ_CUDA(cudaHostAlloc((void **)&data_src, load_info.hdr.totalBytes,
                               cudaHostAllocWriteCombined));
        cuda_staging = true;

        ifstream &file = *get_if<ifstream>(&load_info.data);
        file.read(data_src, load_info.hdr.totalBytes);
    } else {
        data_src = get_if<vector<char>>(&load_info.data)->data();
    }

    cudaMemcpyAsync(scene_storage, data_src, load_info.hdr.totalBytes,
                    cudaMemcpyHostToDevice, stream_);

    // Build BLASes
    uint32_t num_objects = load_info.objectInfo.size();

    vector<CUdeviceptr> blas_storage;
    blas_storage.reserve(num_objects);

    vector<OptixTraversableHandle> blases;
    blases.reserve(num_objects);

    // Improves performance slightly
    unsigned int tri_build_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixAccelBuildOptions accel_options {
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_OPERATION_BUILD,
        {},
    };

    CUdeviceptr scene_storage_dev = (CUdeviceptr)scene_storage;
    const DevicePackedVertex *base_vertex_ptr =
        (const DevicePackedVertex *)scene_storage;
    const uint32_t *base_index_ptr = 
        (const uint32_t *)(scene_storage + load_info.hdr.indexOffset);

    static_assert(sizeof(PackedVertex) == sizeof(Vertex));

    for (int obj_idx = 0; obj_idx < (int)num_objects; obj_idx++) {
        const ObjectInfo &object_info = load_info.objectInfo[obj_idx];
        vector<OptixBuildInput> geometry_infos;
        geometry_infos.reserve(object_info.numMeshes);

        for (int obj_mesh_idx = 0; obj_mesh_idx < (int)object_info.numMeshes;
             obj_mesh_idx++) {
            uint32_t mesh_idx = object_info.meshIndex + obj_mesh_idx;
            const MeshInfo &mesh_info = load_info.meshInfo[mesh_idx];

            OptixBuildInput geometry_info {};
            geometry_info.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            CUdeviceptr index_ptr =
                (CUdeviceptr)(base_index_ptr + mesh_info.indexOffset);
            auto &tri_info = geometry_info.triangleArray;
            tri_info.vertexBuffers = &scene_storage_dev;
            tri_info.numVertices = load_info.hdr.numVertices;
            tri_info.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            tri_info.vertexStrideInBytes = sizeof(Vertex);
            tri_info.indexBuffer = index_ptr;
            tri_info.numIndexTriplets = mesh_info.numTriangles;
            tri_info.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            tri_info.indexStrideInBytes = 0;
            tri_info.preTransform = 0;
            tri_info.flags = &tri_build_flag;
            tri_info.numSbtRecords = 1;
            tri_info.sbtIndexOffsetBuffer = 0;
            tri_info.sbtIndexOffsetSizeInBytes = 0;
            tri_info.sbtIndexOffsetStrideInBytes = 0;
            tri_info.primitiveIndexOffset = 0;
            tri_info.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

            geometry_infos.push_back(geometry_info);
        }

        OptixAccelBufferSizes buffer_sizes;
        REQ_OPTIX(optixAccelComputeMemoryUsage(ctx_, &accel_options,
                                               geometry_infos.data(),
                                               geometry_infos.size(),
                                               &buffer_sizes));

        CUdeviceptr scratch_storage =
            (CUdeviceptr)allocCU(buffer_sizes.tempSizeInBytes);

        CUdeviceptr accel_storage =
            (CUdeviceptr)allocCU(buffer_sizes.outputSizeInBytes);

        OptixTraversableHandle blas;
        REQ_OPTIX(optixAccelBuild(ctx_, stream_, &accel_options,
                                  geometry_infos.data(), geometry_infos.size(),
                                  scratch_storage, buffer_sizes.tempSizeInBytes,
                                  accel_storage, buffer_sizes.outputSizeInBytes,
                                  &blas, nullptr, 0));

        REQ_CUDA(cudaStreamSynchronize(stream_));

        REQ_CUDA(cudaFree((void *)scratch_storage));

        blases.push_back(blas);
        blas_storage.push_back(accel_storage);
    }

    TLAS tlas {};
    tlas.build(ctx_, load_info.envInit.defaultInstances,
        load_info.envInit.defaultTransforms,
        load_info.envInit.defaultInstanceFlags,
        blases.data(), stream_);

    InstanceTransform *default_transforms =
        (InstanceTransform *)allocCU(sizeof(InstanceTransform) *
                                  load_info.envInit.defaultTransforms.size());

    cudaMemcpyAsync(default_transforms, load_info.envInit.defaultTransforms.data(),
        sizeof(InstanceTransform) * load_info.envInit.defaultTransforms.size(),
        cudaMemcpyHostToDevice, stream_);

    // Based on indices in MaterialTextures build indexable list of handles
    vector<cudaTextureObject_t> texture_hdls;
    vector<TextureSize> texture_dims;
    texture_hdls.reserve(num_texture_hdls);
    texture_dims.reserve(num_texture_hdls);
    if (textures.envMap.has_value()) {
        texture_hdls.push_back(textures.envMap->getHandle());
        texture_dims.push_back({
            textures.envMap->getWidth(),
            textures.envMap->getHeight(),
        });
    }
    for (int mat_idx = 0; mat_idx < (int)load_info.hdr.numMaterials;
         mat_idx++) {
        const MaterialTextures &tex_indices =
            load_info.textureIndices[mat_idx];

        auto appendHandle = [&](uint32_t idx, const auto &texture_list) {
            if (idx != ~0u) {
                texture_hdls.push_back(texture_list[idx].getHandle());
                texture_dims.push_back({
                    texture_list[idx].getWidth(),
                    texture_list[idx].getHeight(),
                });
            } else {
                texture_hdls.push_back(0);
                texture_dims.push_back({
                    0, 0,
                });
            }
        };

        appendHandle(tex_indices.baseColorIdx, textures.base);
        appendHandle(tex_indices.metallicRoughnessIdx,
                     textures.metallicRoughness);
        appendHandle(tex_indices.specularIdx, textures.specular);
        appendHandle(tex_indices.normalIdx, textures.normal);
        appendHandle(tex_indices.emittanceIdx, textures.emittance);
        appendHandle(tex_indices.transmissionIdx, textures.transmission);
        appendHandle(tex_indices.clearcoatIdx, textures.clearcoat);
        appendHandle(tex_indices.anisoIdx, textures.anisotropic);
    }

    cudaTextureObject_t *tex_gpu_ptr = (cudaTextureObject_t *)(
        scene_storage + texture_ptr_offset);
    cudaMemcpyAsync(tex_gpu_ptr, texture_hdls.data(), texture_ptr_bytes,
                    cudaMemcpyHostToDevice, stream_);

    TextureSize *tex_dims_ptr = (TextureSize *)(
        scene_storage + texture_dim_offset);
    cudaMemcpyAsync(tex_dims_ptr, texture_dims.data(), texture_dim_bytes,
                    cudaMemcpyHostToDevice, stream_);

    REQ_CUDA(cudaStreamSynchronize(stream_));
    
    if (cuda_staging) {
        REQ_CUDA(cudaFreeHost(data_src));
    }

    optional<ScenePhysicsData> physics;
    if (need_physics_) {
        physics.emplace(ScenePhysicsData::make(load_info.physics,
                                               load_info.hdr, 
                                               scene_storage,
                                               scene_storage + sdf_ptr_offset,
                                               stream_, texture_mgr_));
    }

    return shared_ptr<OptixScene>(new OptixScene {
        {
            move(load_info.meshInfo),
            move(load_info.objectInfo),
            move(load_info.envInit),
        },
        scene_storage_dev,
        base_vertex_ptr,
        base_index_ptr,
        reinterpret_cast<const PackedMaterial *>(
            scene_storage + load_info.hdr.materialOffset),
        reinterpret_cast<const PackedMeshInfo *>(
            scene_storage + load_info.hdr.meshOffset),
        move(blas_storage),
        move(blases),
        move(tlas),
        default_transforms,
        move(textures),
        tex_gpu_ptr,
        tex_dims_ptr,
        move(physics),
    });
}

}
}
