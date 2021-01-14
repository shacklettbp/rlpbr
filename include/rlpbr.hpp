#pragma once

#include <rlpbr/fwd.hpp>
#include <rlpbr/config.hpp>
#include <rlpbr/utils.hpp>
#include <rlpbr/environment.hpp>

#include <string_view>

namespace RLpbr {

class AssetLoader {
public:
    AssetLoader(Handle<LoaderBackend> &&backend);

    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

private:
    Handle<LoaderBackend> backend_;

friend class BatchRenderer;
};

class Renderer {
public:
    Renderer(const RenderConfig &cfg);

    AssetLoader makeLoader();

    Environment makeEnvironment(const std::shared_ptr<Scene> &scene);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::vec3 &eye, const glm::vec3 &look,
                                const glm::vec3 &up);
    
    void render(const Environment *envs);

private:
    Handle<RenderBackend> backend_;
};

}
