#include "optix_scene.hpp"
#include "optix_utils.hpp"
#include "shader.hpp"

#include <optix_stubs.h>
#include <iostream>

#include <glm/gtc/type_ptr.hpp>

using namespace std;

namespace RLpbr {
namespace optix {

OptixScene::~OptixScene()
{
    cudaFree((void *)blasStorage);
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
    uint32_t num_instances = 0;
    for (const auto &transforms : scene.envInit.transforms) {
        num_instances += transforms.size();
    }

    OptixInstance *instance_ptr;
    REQ_CUDA(cudaHostAlloc(&instance_ptr, sizeof(OptixInstance) * num_instances,
                           cudaHostAllocMapped));

    uint32_t cur_instance_idx = 0;
    for (uint32_t model_idx = 0; model_idx < scene.envInit.transforms.size();
         model_idx++) {
        const auto &transforms = scene.envInit.transforms[model_idx];
        for (const glm::mat4x3 &txfm : transforms) {
            OptixInstance &cur_inst =
                instance_ptr[cur_instance_idx];

            assignInstanceTransform(cur_inst, txfm);
            cur_inst.instanceId = scene.meshInfo[model_idx].indexOffset;
            cur_inst.sbtOffset = 0;
            cur_inst.visibilityMask = 0xff;
            cur_inst.flags = 0;
            cur_inst.traversableHandle = scene.blases[model_idx];

            cur_instance_idx++;
        }
    }

    OptixBuildInput tlas_build;
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

    cudaStreamSynchronize(build_stream);
 
    REQ_CUDA(cudaFree(scratch_storage));
    REQ_CUDA(cudaFreeHost(instance_ptr));

    return OptixEnvironment {
        {},
        (CUdeviceptr)tlas_storage,
        tlas,
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

shared_ptr<Scene> OptixLoader::loadScene(SceneLoadData &&load_info)
{
    CUdeviceptr scene_storage = (CUdeviceptr)allocCU(load_info.hdr.totalBytes);

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

    cudaMemcpyAsync((void *)scene_storage, data_src, load_info.hdr.totalBytes,
                    cudaMemcpyHostToDevice, stream_);

    OptixBuildInput geometry_info;
    geometry_info.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    unsigned int tri_info_flag = OPTIX_GEOMETRY_FLAG_NONE;
    auto &tri_info = geometry_info.triangleArray;
    tri_info.vertexBuffers = &scene_storage;
    tri_info.numVertices = load_info.hdr.numVertices;
    tri_info.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_info.vertexStrideInBytes = sizeof(Vertex);
    tri_info.indexBuffer = scene_storage + load_info.hdr.indexOffset;
    tri_info.numIndexTriplets = load_info.meshInfo[0].numTriangles;
    tri_info.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri_info.indexStrideInBytes = 0;
    tri_info.preTransform = 0;
    tri_info.flags = &tri_info_flag;
    tri_info.numSbtRecords = 1;
    tri_info.sbtIndexOffsetBuffer = 0;
    tri_info.sbtIndexOffsetSizeInBytes = 0;
    tri_info.sbtIndexOffsetStrideInBytes = 0;
    tri_info.primitiveIndexOffset = 0;
    tri_info.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    OptixAccelBuildOptions accel_options {
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_OPERATION_BUILD,
        {},
    };

    OptixAccelBufferSizes buffer_sizes;
    REQ_OPTIX(optixAccelComputeMemoryUsage(ctx_, &accel_options,
                                           &geometry_info, 1,
                                           &buffer_sizes));

    CUdeviceptr scratch_storage =
        (CUdeviceptr)allocCU(buffer_sizes.tempSizeInBytes);

    CUdeviceptr accel_storage =
        (CUdeviceptr)allocCU(buffer_sizes.outputSizeInBytes);

    OptixTraversableHandle accel_struct;
    REQ_OPTIX(optixAccelBuild(ctx_, stream_, &accel_options,
                              &geometry_info, 1,
                              scratch_storage, buffer_sizes.tempSizeInBytes,
                              accel_storage, buffer_sizes.outputSizeInBytes,
                              &accel_struct, nullptr, 0));

    REQ_CUDA(cudaStreamSynchronize(stream_));

    REQ_CUDA(cudaFree((void *)scratch_storage));
    
    if (cuda_staging) {
        REQ_CUDA(cudaFreeHost(data_src));
    }

    return shared_ptr<OptixScene>(new OptixScene {
        {
            move(load_info.meshInfo),
            move(load_info.envInit),
        },
        scene_storage,
        accel_storage,
        { accel_struct },
    });
}

}
}
