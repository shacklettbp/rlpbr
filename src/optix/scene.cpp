#include "scene.hpp"
#include "utils.hpp"

#include <rlpbr_backend/shader.hpp>
#include <optix_stubs.h>
#include <iostream>

#include <glm/gtc/type_ptr.hpp>

using namespace std;

namespace RLpbr {
namespace optix {

OptixScene::~OptixScene()
{
    cudaFreeHost(defaultTLAS.instanceBLASes);

    cudaFree((void *)defaultTLAS.storage);

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

OptixEnvironment OptixEnvironment::make(OptixDeviceContext ctx,
                                        cudaStream_t build_stream,
                                        const OptixScene &scene)
{
    void *tlas_storage = allocCU(scene.defaultTLAS.numBytes);
    cudaMemcpyAsync(tlas_storage, (void *)scene.defaultTLAS.storage,
                    scene.defaultTLAS.numBytes, cudaMemcpyDeviceToDevice,
                    build_stream);

    OptixAccelRelocationInfo tlas_reloc_info {};
    REQ_OPTIX(optixAccelGetRelocationInfo(ctx, scene.defaultTLAS.hdl,
                                          &tlas_reloc_info));

    OptixTraversableHandle new_tlas;
    REQ_OPTIX(optixAccelRelocate(ctx, build_stream, &tlas_reloc_info,
        (CUdeviceptr)scene.defaultTLAS.instanceBLASes,
        scene.envInit.indexMap.size(), (CUdeviceptr)tlas_storage,
        scene.defaultTLAS.numBytes, &new_tlas));

    REQ_CUDA(cudaStreamSynchronize(build_stream));

    return OptixEnvironment {
        {},
        (CUdeviceptr)tlas_storage,
        new_tlas,
    };
}

OptixEnvironment::~OptixEnvironment()
{
    cudaFree((void *)tlasStorage);
}

uint32_t OptixEnvironment::addLight(const glm::vec3 &position,
                  const glm::vec3 &color)
{
    (void)position;
    (void)color;
    return 0;
}

void OptixEnvironment::removeLight(uint32_t light_idx)
{
    (void)light_idx;
}

OptixLoader::OptixLoader(OptixDeviceContext ctx)
    : stream_([]() {
          cudaStream_t strm;
          REQ_CUDA(cudaStreamCreate(&strm));
          return strm;
      }()),
      ctx_(ctx)
{}

struct TLASIntermediate {
    void *instanceTransforms;
    void *buildScratch;
};

static void freeTLASIntermediate(TLASIntermediate &&inter)
{
    REQ_CUDA(cudaFreeHost(inter.instanceTransforms));
    REQ_CUDA(cudaFree(inter.buildScratch));
}

static pair<TLAS, TLASIntermediate> buildTLAS(
    OptixDeviceContext ctx,
    const vector<vector<glm::mat4x3>> &instance_transforms,
    uint32_t num_instances,
    const vector<MeshInfo> &mesh_infos,
    OptixTraversableHandle *blases,
    cudaStream_t build_stream)
{
    OptixInstance *instance_ptr;
    REQ_CUDA(cudaHostAlloc(&instance_ptr, sizeof(OptixInstance) * num_instances,
                           cudaHostAllocMapped));

    OptixTraversableHandle *instance_blases;
    REQ_CUDA(cudaHostAlloc(&instance_blases,
                           sizeof(OptixTraversableHandle) * num_instances,
                           cudaHostAllocMapped));

    uint32_t cur_instance_idx = 0;
    for (uint32_t model_idx = 0; model_idx < instance_transforms.size();
         model_idx++) {
        const auto &model_instances = instance_transforms[model_idx];
        for (const glm::mat4x3 &txfm : model_instances) {
            OptixInstance &cur_inst = instance_ptr[cur_instance_idx];

            assignInstanceTransform(cur_inst, txfm);
            cur_inst.instanceId = mesh_infos[model_idx].indexOffset;
            cur_inst.sbtOffset = 0;
            cur_inst.visibilityMask = 0xff;
            cur_inst.flags = 0;
            cur_inst.traversableHandle = blases[model_idx];

            instance_blases[cur_instance_idx] = blases[model_idx];

            cur_instance_idx++;
        }
    }

    OptixBuildInput tlas_build {};
    tlas_build.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray &tlas_instances = tlas_build.instanceArray;
    tlas_instances.instances = (CUdeviceptr)instance_ptr;
    tlas_instances.numInstances = num_instances;

    OptixAccelBuildOptions tlas_options;
    tlas_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    tlas_options.motionOptions = {};
    tlas_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlas_buffer_sizes;
    REQ_OPTIX(optixAccelComputeMemoryUsage(ctx, &tlas_options, &tlas_build, 1,
                                           &tlas_buffer_sizes));

    void *tlas_storage;
    REQ_CUDA(cudaMalloc(&tlas_storage, tlas_buffer_sizes.outputSizeInBytes));
    void *scratch_storage;
    REQ_CUDA(cudaMalloc(&scratch_storage, tlas_buffer_sizes.tempSizeInBytes));

    OptixTraversableHandle tlas;
    REQ_OPTIX(optixAccelBuild(ctx, build_stream, &tlas_options, &tlas_build,
        1, (CUdeviceptr)scratch_storage, tlas_buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)tlas_storage, tlas_buffer_sizes.outputSizeInBytes, &tlas,
        nullptr, 0));

    return {
        TLAS {
            tlas,
            (CUdeviceptr)tlas_storage,
            tlas_buffer_sizes.outputSizeInBytes,
            instance_blases,
        },
        TLASIntermediate {
            instance_ptr,
            scratch_storage,
        },
    };
}

shared_ptr<Scene> OptixLoader::loadScene(SceneLoadData &&load_info)
{
    char *scene_storage = (char *)allocCU(load_info.hdr.totalBytes);

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
    uint32_t num_meshes = load_info.meshInfo.size();

    vector<CUdeviceptr> blas_storage;
    blas_storage.reserve(num_meshes);

    vector<OptixTraversableHandle> blases;
    blases.reserve(num_meshes);

    // Improves performance slightly
    unsigned int tri_build_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixAccelBuildOptions accel_options {
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_OPERATION_BUILD,
        {},
    };

    CUdeviceptr scene_storage_dev = (CUdeviceptr)scene_storage;
    const PackedVertex *base_vertex_ptr = (const PackedVertex *)scene_storage;
    const uint32_t *base_index_ptr = 
        (const uint32_t *)(scene_storage + load_info.hdr.indexOffset);

    static_assert(sizeof(PackedVertex) == sizeof(Vertex));

    for (uint32_t mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
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

        OptixAccelBufferSizes buffer_sizes;
        REQ_OPTIX(optixAccelComputeMemoryUsage(ctx_, &accel_options,
                                               &geometry_info, 1,
                                               &buffer_sizes));

        CUdeviceptr scratch_storage =
            (CUdeviceptr)allocCU(buffer_sizes.tempSizeInBytes);

        CUdeviceptr accel_storage =
            (CUdeviceptr)allocCU(buffer_sizes.outputSizeInBytes);

        REQ_OPTIX(optixAccelBuild(ctx_, stream_, &accel_options,
                                  &geometry_info, 1,
                                  scratch_storage, buffer_sizes.tempSizeInBytes,
                                  accel_storage, buffer_sizes.outputSizeInBytes,
                                  &blases[mesh_idx], nullptr, 0));

        REQ_CUDA(cudaStreamSynchronize(stream_));

        REQ_CUDA(cudaFree((void *)scratch_storage));

        blas_storage.push_back(accel_storage);
    }

    auto [tlas, tlas_inter] = buildTLAS(ctx_, load_info.envInit.transforms,
        load_info.envInit.indexMap.size(), load_info.meshInfo,
        blases.data(), stream_);

    REQ_CUDA(cudaStreamSynchronize(stream_));
    freeTLASIntermediate(move(tlas_inter));
    
    if (cuda_staging) {
        REQ_CUDA(cudaFreeHost(data_src));
    }

    return shared_ptr<OptixScene>(new OptixScene {
        {
            move(load_info.meshInfo),
            move(load_info.envInit),
        },
        scene_storage_dev,
        base_vertex_ptr,
        base_index_ptr,
        move(blas_storage),
        move(blases),
        move(tlas),
    });
}

}
}
