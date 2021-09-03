#pragma once

#include <rlpbr/fwd.hpp>
#include <rlpbr/config.hpp>
#include <rlpbr/backend.hpp>
#include <rlpbr/utils.hpp>
#include <rlpbr/environment.hpp>
#include <rlpbr/render.hpp>

#include <string_view>

namespace RLpbr {

class AssetLoader {
public:
    AssetLoader(LoaderImpl &&backend);

    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

private:
    LoaderImpl backend_;

friend class BatchRenderer;
};

class Renderer {
public:
    Renderer(const RenderConfig &cfg);

    AssetLoader makeLoader();

    Environment makeEnvironment(const std::shared_ptr<Scene> &scene);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::vec3 &eye, const glm::vec3 &target,
                                const glm::vec3 &up, float vertical_fov = 90.f,
                                float aspect_ratio = 0.f);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::mat4 &camera_to_world,
                                float vertical_fov = 90.f,
                                float aspect_ratio = 0.f);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::vec3 &pos,
                                const glm::vec3 &fwd,
                                const glm::vec3 &up,
                                const glm::vec3 &right,
                                float vertical_fov = 90.f,
                                float aspect_ratio = 0.f);

    RenderBatch makeRenderBatch();

    void render(RenderBatch &batch);

    void waitForBatch(RenderBatch &batch);

    half *getOutputPointer(RenderBatch &batch) const;

    AuxiliaryOutputs getAuxiliaryOutputs(RenderBatch &batch) const;

private:
    RendererImpl backend_;
    float aspect_ratio_;
    uint32_t batch_size_;
};

}
