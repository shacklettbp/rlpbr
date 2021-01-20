#pragma once

#include "common.hpp"
#include "scene.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace RLpbr {
namespace optix {

class OptixEnvironment : public EnvironmentBackend {
public:
    uint32_t addLight(const glm::vec3 &position,
                      const glm::vec3 &color);

    void removeLight(uint32_t light_idx);
};

struct OptixScene : public Scene {
    CUdeviceptr sceneStorage;
    CUdeviceptr accelStorage;
    OptixTraversableHandle accelStructure;
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
