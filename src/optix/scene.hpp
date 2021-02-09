#pragma once

#include <rlpbr_backend/common.hpp>
#include <rlpbr_backend/scene.hpp>
#include <rlpbr_backend/utils.hpp>
#include <rlpbr_backend/shader.hpp>

#include "shader.hpp"

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

struct Texture {
    cudaArray_t mem;
    cudaTextureObject_t hdl;
};

struct OptixScene : public Scene {
    OptixScene(const OptixScene &) = delete;
    OptixScene(OptixScene &&) = delete;
    ~OptixScene();

    CUdeviceptr sceneStorage;
    const PackedVertex *vertexPtr;
    const uint32_t *indexPtr;
    const PackedMaterial *materialPtr;

    std::vector<CUdeviceptr> blasStorage;
    std::vector<OptixTraversableHandle> blases;
    TLAS defaultTLAS;

    std::vector<Texture> textures;
    const cudaTextureObject_t *texturePtr;
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

    std::vector<PackedLight> lights;
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
