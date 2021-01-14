#pragma once

#include <rlpbr/environment.hpp>

#include <cstdint>
#include <vector>

namespace RLpbr {

struct SceneLoadData;

struct HostRenderData {
    uint64_t backendIdx;
    void *ptr;
};

struct LoaderBackend {
    ~LoaderBackend();
    void (LoaderBackend::*destroy)();

    std::shared_ptr<Scene> (LoaderBackend::*loadScene)(
        const SceneLoadData &scene_data);

    HostRenderData (LoaderBackend::*allocHostData)(uint64_t num_bytes);
};

struct EnvironmentState {
    ~EnvironmentState();
    void (EnvironmentState::*destroy)();

    void (EnvironmentState::*addLight)(const glm::vec3 &position,
                                       const glm::vec3 &color);

    void (EnvironmentState::*deleteLight)(uint32_t idx);
};

struct RenderBackend {
    ~RenderBackend();
    void (RenderBackend::*destroy)();

    Handle<LoaderBackend> (RenderBackend::*makeLoader)();

    Handle<EnvironmentState> (RenderBackend::*makeEnvironment)(
        const std::shared_ptr<Scene> &scene);

    void (RenderBackend::*render)(const Environment *envs);
};

}
