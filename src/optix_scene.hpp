#pragma once

#include "common.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "shader.hpp"
#include "optix_shader.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace RLpbr {
namespace optix {

struct TLAS {
    OptixTraversableHandle hdl;
    CUdeviceptr storage;
    size_t numBytes;
    OptixTraversableHandle *instanceBLASes;
};

struct OptixScene : public Scene {
    OptixScene(const OptixScene &) = delete;
    OptixScene(OptixScene &&) = delete;
    ~OptixScene();

    CUdeviceptr sceneStorage;
    const PackedVertex *vertexPtr;
    const uint32_t *indexPtr;

    std::vector<CUdeviceptr> blasStorage;
    std::vector<OptixTraversableHandle> blases;
    TLAS defaultTLAS;
};

class OptixEnvironment : public EnvironmentBackend {
public:
    static OptixEnvironment make(OptixDeviceContext ctx,
                                 cudaStream_t build_stream,
                                 const OptixScene &scene);

    OptixEnvironment(const OptixEnvironment &) = delete;
    OptixEnvironment(OptixEnvironment &&) = delete;
    ~OptixEnvironment();

    uint32_t addLight(const glm::vec3 &position,
                      const glm::vec3 &color);

    void removeLight(uint32_t light_idx);

    CUdeviceptr tlasStorage;
    OptixTraversableHandle tlas;
};

class OptixLoader : public LoaderBackend {
public:
    OptixLoader(OptixDeviceContext ctx);

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_info);

private:
    cudaStream_t stream_;
    OptixDeviceContext ctx_;
};

}
}
